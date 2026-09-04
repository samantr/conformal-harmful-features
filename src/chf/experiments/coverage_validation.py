from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


GROUP_COVERAGE_COLUMNS = (
    "group_coverage_count",
    "group_macro_coverage",
    "group_coverage_standard_deviation",
    "group_coverage_standard_error",
    "group_coverage_ci_lower",
    "group_coverage_ci_upper",
    "group_coverage_confidence_level",
    "group_coverage_status",
    "group_coverages",
    "group_covered_counts",
    "group_sample_counts",
)


@dataclass(frozen=True)
class CoverageValidation:
    check_name: str
    hard_pass: bool
    scientific_status: str
    policy: dict[str, Any]
    diagnostics: dict[str, Any]


def coverage_validation_mode(config: Mapping[str, Any]) -> str:
    """Return the explicit coverage policy, with a split-aware default."""
    configured = config.get("coverage_validation", {})
    default = "grouped" if config["split"].get("unit", "rows") == "groups" else "fixed"
    mode = str(configured.get("mode", default))
    if mode not in {"fixed", "grouped"}:
        raise ValueError("coverage_validation.mode must be 'fixed' or 'grouped'")
    if mode == "grouped" and config["split"].get("unit", "rows") != "groups":
        raise ValueError("grouped coverage validation requires split.unit: groups")
    return mode


def grouped_coverage_metrics(
    included: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    target: float,
    confidence_level: float,
) -> dict[str, Any]:
    """Summarize coverage using independent groups as the sampling unit."""
    included_values = np.asarray(included, dtype=bool)
    label_values = np.asarray(labels)
    group_values = np.asarray(groups)
    if included_values.ndim != 2:
        raise ValueError("included must be a two-dimensional boolean matrix")
    if label_values.shape != (len(included_values),):
        raise ValueError("labels must contain one value per included row")
    if group_values.shape != (len(included_values),):
        raise ValueError("groups must contain one value per included row")
    if not 0 < target < 1:
        raise ValueError("coverage target must lie strictly between zero and one")
    if not 0 < confidence_level < 1:
        raise ValueError("coverage confidence level must lie between zero and one")

    covered = included_values[np.arange(len(label_values)), label_values]
    unique_groups = np.unique(group_values)
    if len(unique_groups) < 2:
        raise ValueError("grouped coverage validation requires at least two test groups")

    group_coverages: dict[str, float] = {}
    group_covered_counts: dict[str, int] = {}
    group_sample_counts: dict[str, int] = {}
    for group in unique_groups:
        mask = group_values == group
        key = str(group.item() if isinstance(group, np.generic) else group)
        group_sample_counts[key] = int(mask.sum())
        group_covered_counts[key] = int(covered[mask].sum())
        group_coverages[key] = float(covered[mask].mean())

    coverage_values = np.fromiter(group_coverages.values(), dtype=float)
    macro_coverage = float(coverage_values.mean())
    standard_deviation = float(coverage_values.std(ddof=1))
    standard_error = float(standard_deviation / np.sqrt(len(coverage_values)))
    critical_value = float(
        student_t.ppf(
            0.5 + confidence_level / 2.0,
            df=len(coverage_values) - 1,
        )
    )
    half_width = critical_value * standard_error
    lower = float(max(0.0, macro_coverage - half_width))
    upper = float(min(1.0, macro_coverage + half_width))
    if upper < target:
        status = "UNDERCOVERAGE"
    elif lower > target:
        status = "OVERCONSERVATIVE"
    else:
        status = "WITHIN_GROUPED_UNCERTAINTY"

    return {
        "group_coverage_count": int(len(coverage_values)),
        "group_macro_coverage": macro_coverage,
        "group_coverage_standard_deviation": standard_deviation,
        "group_coverage_standard_error": standard_error,
        "group_coverage_ci_lower": lower,
        "group_coverage_ci_upper": upper,
        "group_coverage_confidence_level": float(confidence_level),
        "group_coverage_status": status,
        "group_coverages": group_coverages,
        "group_covered_counts": group_covered_counts,
        "group_sample_counts": group_sample_counts,
    }


def validate_coverage_results(
    results: pd.DataFrame,
    config: Mapping[str, Any],
) -> CoverageValidation:
    """Apply the configured hard gate and return explicit scientific diagnostics."""
    if results.empty:
        raise ValueError("coverage validation requires at least one result row")
    target = 1.0 - float(config["conformal"]["alpha"])
    mode = coverage_validation_mode(config)
    if mode == "fixed":
        tolerance = float(
            config.get("stability", {}).get("max_coverage_deviation", 0.03)
        )
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("fixed coverage tolerance must be finite and non-negative")
        if not np.isfinite(results["coverage"]).all():
            raise ValueError("coverage values must be finite")
        deviations = results["coverage"].sub(target).abs()
        hard_pass = bool(deviations.le(tolerance + 1e-12).all())
        return CoverageValidation(
            check_name="coverage_within_stability_tolerance",
            hard_pass=hard_pass,
            scientific_status=(
                "WITHIN_FIXED_TOLERANCE"
                if hard_pass
                else "OUTSIDE_FIXED_TOLERANCE"
            ),
            policy={
                "mode": "fixed",
                "unit": "rows",
                "target": target,
                "absolute_tolerance": tolerance,
                "hard_failure_rule": "any row outside the fixed absolute tolerance",
            },
            diagnostics={
                "evaluated_rows": int(len(results)),
                "failing_rows": int(deviations.gt(tolerance + 1e-12).sum()),
                "maximum_absolute_deviation": float(deviations.max()),
            },
        )

    missing = set(GROUP_COVERAGE_COLUMNS).difference(results.columns)
    if missing:
        raise ValueError(
            "grouped coverage validation requires saved group metrics; "
            f"missing columns: {sorted(missing)}"
        )
    configured = config.get("coverage_validation", {})
    confidence_level = float(configured.get("confidence_level", 0.95))
    numeric_columns = [
        column
        for column in GROUP_COVERAGE_COLUMNS
        if column
        not in {
            "group_coverage_status",
            "group_coverages",
            "group_covered_counts",
            "group_sample_counts",
        }
    ]
    if not np.isfinite(results[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("saved grouped coverage metrics must be finite")
    observed_levels = results["group_coverage_confidence_level"].astype(float)
    if not np.allclose(observed_levels, confidence_level, atol=0.0, rtol=0.0):
        raise ValueError("saved group coverage confidence level differs from config")
    if results["group_coverage_status"].isna().any():
        raise ValueError("saved grouped coverage statuses must not be missing")
    lower = results["group_coverage_ci_lower"].astype(float)
    upper = results["group_coverage_ci_upper"].astype(float)
    if not ((0 <= lower) & (lower <= upper) & (upper <= 1)).all():
        raise ValueError("saved grouped coverage intervals are invalid")
    expected_statuses = np.where(
        upper < target,
        "UNDERCOVERAGE",
        np.where(
            lower > target,
            "OVERCONSERVATIVE",
            "WITHIN_GROUPED_UNCERTAINTY",
        ),
    )
    if not np.array_equal(
        results["group_coverage_status"].to_numpy(), expected_statuses
    ):
        raise ValueError("saved grouped coverage statuses disagree with intervals")
    statuses = results["group_coverage_status"].value_counts().to_dict()
    unknown = set(statuses).difference(
        {
            "UNDERCOVERAGE",
            "WITHIN_GROUPED_UNCERTAINTY",
            "OVERCONSERVATIVE",
        }
    )
    if unknown:
        raise ValueError(f"unknown grouped coverage statuses: {sorted(unknown)}")
    undercoverage_count = int(statuses.get("UNDERCOVERAGE", 0))
    overconservative_count = int(statuses.get("OVERCONSERVATIVE", 0))
    within_count = int(statuses.get("WITHIN_GROUPED_UNCERTAINTY", 0))
    if undercoverage_count:
        scientific_status = "UNDERCOVERAGE"
    elif overconservative_count:
        scientific_status = "OVERCONSERVATIVE"
    else:
        scientific_status = "WITHIN_GROUPED_UNCERTAINTY"
    group_counts = results["group_coverage_count"].astype(int)
    if group_counts.lt(2).any():
        raise ValueError("grouped coverage validation requires at least two groups")
    return CoverageValidation(
        check_name="no_grouped_undercoverage",
        hard_pass=undercoverage_count == 0,
        scientific_status=scientific_status,
        policy={
            "mode": "grouped",
            "unit": "groups",
            "estimand": "unweighted mean of per-group window coverage",
            "confidence_interval": "two-sided Student t interval across groups",
            "confidence_level": confidence_level,
            "target": target,
            "hard_failure_rule": (
                "any row whose grouped confidence-interval upper bound is below target"
            ),
            "overcoverage_rule": (
                "record OVERCONSERVATIVE when the grouped interval lower bound "
                "is above target; do not abort computation"
            ),
        },
        diagnostics={
            "evaluated_rows": int(len(results)),
            "test_groups_per_row_minimum": int(group_counts.min()),
            "test_groups_per_row_maximum": int(group_counts.max()),
            "undercoverage_rows": undercoverage_count,
            "within_grouped_uncertainty_rows": within_count,
            "overconservative_rows": overconservative_count,
            "all_rows_within_grouped_uncertainty": within_count == len(results),
        },
    )
