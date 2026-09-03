import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from chf.conformal import calibrate_threshold, evaluate_prediction_sets
from chf.data import (
    ControlledSyntheticDataset,
    DRY_BEAN_ARCHIVE_SHA256,
    DRY_BEAN_URL,
    RealTabularDataset,
    load_dry_bean,
    make_controlled_multiclass,
)
from chf.metrics import classification_metrics, conditional_coverage_metrics
from chf.scaling import (
    probabilities_from_logits,
    probability_diagnostics,
    tune_confts,
    tune_temperature,
)


REFERENCE_METRICS = (
    "accuracy",
    "macro_f1",
    "nll",
    "ece",
    "coverage",
    "mean_size",
    "median_size",
    "size_p90",
    "empty_set_rate",
    "full_set_rate",
    "sscv",
    "class_coverage_gap",
    "class_coverage_max_deviation",
)


def split_id(split: Any) -> str:
    """Return a stable identifier for a four-way split."""
    digest = hashlib.sha256()
    for name in ("train", "tune", "calibration", "test"):
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(getattr(split, name), dtype=np.int64).tobytes())
    return digest.hexdigest()[:16]


def code_version(repository_root: Path) -> str:
    """Return the current Git revision, marking an uncommitted worktree."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return f"{commit}-dirty" if dirty else commit
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def dataset_from_config(
    config: Mapping[str, Any], repository_root: Path | None = None
) -> ControlledSyntheticDataset | RealTabularDataset:
    """Load the configured dataset without fitting any preprocessing."""
    dataset_config = config["dataset"]
    dataset_kind = str(dataset_config.get("kind", "controlled_synthetic"))
    if dataset_kind == "dry_bean":
        archive_path = Path(
            dataset_config.get(
                "archive_path", "outputs/_datasets/dry_bean_dataset.zip"
            )
        )
        if not archive_path.is_absolute():
            archive_path = (repository_root or Path.cwd()) / archive_path
        return load_dry_bean(
            archive_path,
            source_url=str(dataset_config.get("source_url", DRY_BEAN_URL)),
            expected_sha256=str(
                dataset_config.get("archive_sha256", DRY_BEAN_ARCHIVE_SHA256)
            ),
        )
    if dataset_kind != "controlled_synthetic":
        raise ValueError(f"unsupported dataset kind: {dataset_kind}")
    feature_config = dataset_config["features"]
    return make_controlled_multiclass(
        n_samples=int(dataset_config["n_samples"]),
        n_classes=int(dataset_config["n_classes"]),
        n_strong=int(feature_config["strong"]),
        n_weak=int(feature_config["weak"]),
        n_redundant=int(feature_config["redundant"]),
        n_noise=int(feature_config["noise"]),
        seed=int(config["seed"]),
        strong_signal=float(dataset_config.get("strong_signal", 1.0)),
        weak_signal=float(dataset_config.get("weak_signal", 0.25)),
        redundant_noise=float(dataset_config.get("redundant_noise", 0.15)),
    )


def dataset_name(dataset: ControlledSyntheticDataset | RealTabularDataset) -> str:
    """Return the stable result-table identifier for a loaded dataset."""
    return str(getattr(dataset, "name", "controlled_multiclass"))


def attach_reference_deltas(
    results: pd.DataFrame,
    *,
    key_columns: tuple[str, ...] = ("model", "scaling", "score"),
) -> pd.DataFrame:
    """Attach matched reference values and signed intervention deltas.

    ``key_columns`` identifies the experimental conditions that must be paired.
    Phase 2 uses model/scaling/score, while resampled Phase 3 evidence also adds
    the tuning-resample identifier. Positive ``mean_size_reduction`` means that
    the intervention made prediction sets smaller.
    """
    required = {
        "intervention",
        "alpha",
        *key_columns,
        *REFERENCE_METRICS,
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"results are missing required columns: {sorted(missing)}")

    reference = results.loc[results["intervention"] == "reference"].copy()
    keys = list(key_columns)
    if reference.duplicated(keys).any():
        raise RuntimeError(f"reference rows are not unique by {keys}")
    reference = reference[keys + list(REFERENCE_METRICS)].rename(
        columns={metric: f"reference_{metric}" for metric in REFERENCE_METRICS}
    )
    merged = results.merge(reference, on=keys, how="left", validate="many_to_one")
    reference_columns = [f"reference_{metric}" for metric in REFERENCE_METRICS]
    if merged[reference_columns].isna().any().any():
        raise RuntimeError("an intervention row has no matching reference")
    for metric in REFERENCE_METRICS:
        merged[f"{metric}_delta"] = merged[metric] - merged[f"reference_{metric}"]
    merged["accuracy_loss"] = -merged["accuracy_delta"]
    merged["macro_f1_loss"] = -merged["macro_f1_delta"]
    merged["mean_size_reduction"] = -merged["mean_size_delta"]
    merged["coverage_target_deviation"] = (
        merged["coverage"] - (1.0 - merged["alpha"])
    ).abs()
    return merged


def evaluate_logits(
    *,
    logits_tune: np.ndarray,
    logits_calibration: np.ndarray,
    logits_test: np.ndarray,
    labels_tune: np.ndarray,
    labels_calibration: np.ndarray,
    labels_test: np.ndarray,
    config: Mapping[str, Any],
    calibration_uniforms: np.ndarray,
    test_uniforms: np.ndarray,
    seed: int,
    included_scores: tuple[str, ...] | None = None,
    included_scalings: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate one fitted-model/data combination under the shared CP protocol.

    Temperature choices use only ``tune``. Each frozen choice receives a fresh
    threshold from ``calibration`` and is evaluated once on ``test``. Explicit
    uniforms make every randomized APS/RAPS comparison paired.
    """
    conformal_config = config["conformal"]
    alpha = float(conformal_config["alpha"])
    temperature_grid = [float(value) for value in config["temperature"]["grid"]]
    ece_bins = int(config.get("metrics", {}).get("ece_bins", 15))
    min_sscv_size = int(
        config.get("metrics", {}).get("min_sscv_stratum_size", 25)
    )
    raps_lambda = float(conformal_config["raps_lambda"])
    raps_k_reg = int(conformal_config["raps_k_reg"])
    score_names = included_scores or tuple(conformal_config["scores"])
    unknown_scores = set(score_names).difference(conformal_config["scores"])
    if unknown_scores:
        raise ValueError(f"unknown requested scores: {sorted(unknown_scores)}")
    scaling_names = included_scalings or ("base", "ts", "confts")
    unknown_scalings = set(scaling_names).difference({"base", "ts", "confts"})
    if unknown_scalings:
        raise ValueError(
            f"unknown requested scaling methods: {sorted(unknown_scalings)}"
        )
    ts_result = (
        tune_temperature(logits_tune, labels_tune, temperature_grid)
        if "ts" in scaling_names
        else None
    )
    rows: list[dict[str, Any]] = []

    for score_name in score_names:
        confts_result = (
            tune_confts(
                logits_tune,
                labels_tune,
                alpha,
                score_name,
                temperature_grid,
                seed=seed + 30_001,
                threshold_fraction=float(
                    config["temperature"].get("confts_threshold_fraction", 0.5)
                ),
                k_reg=raps_k_reg,
                lambda_reg=raps_lambda,
                reject_zero_probabilities=bool(
                    config["temperature"].get("reject_zero_probabilities", True)
                ),
                reject_saturated_probabilities=bool(
                    config["temperature"].get(
                        "reject_saturated_probabilities", False
                    )
                ),
            )
            if "confts" in scaling_names
            else None
        )
        candidates = {
            "base": ("base", 1.0, np.nan, np.nan, ()),
            "ts": (
                "ts", ts_result.temperature, ts_result.nll, np.nan, ()
            ) if ts_result is not None else None,
            "confts": (
                "confts", confts_result.temperature, np.nan, confts_result.loss,
                confts_result.rejected_temperatures,
            ) if confts_result is not None else None,
        }
        scaling_candidates = tuple(candidates[name] for name in scaling_names)
        for (
            scaling_name,
            temperature,
            tuning_nll,
            confts_loss,
            rejected,
        ) in scaling_candidates:
            probability_calibration = probabilities_from_logits(
                logits_calibration, temperature
            )
            probability_test = probabilities_from_logits(logits_test, temperature)
            tau = calibrate_threshold(
                probability_calibration,
                labels_calibration,
                alpha,
                score_name,
                u_values=calibration_uniforms,
                k_reg=raps_k_reg,
                lambda_reg=raps_lambda,
            )
            prediction_set_result = evaluate_prediction_sets(
                probability_test,
                labels_test,
                tau,
                score_name,
                u_values=test_uniforms,
                k_reg=raps_k_reg,
                lambda_reg=raps_lambda,
            )
            classification = classification_metrics(
                probability_test, labels_test, ece_bins=ece_bins
            )
            conditional = conditional_coverage_metrics(
                prediction_set_result.included,
                labels_test,
                target_coverage=1 - alpha,
                min_sscv_stratum_size=min_sscv_size,
            )
            calibration_diagnostics = probability_diagnostics(
                probability_calibration
            )
            test_diagnostics = probability_diagnostics(probability_test)
            rows.append(
                {
                    "scaling": scaling_name,
                    "temperature": temperature,
                    "score": score_name,
                    "alpha": alpha,
                    "raps_lambda": raps_lambda,
                    "raps_k_reg": raps_k_reg,
                    "threshold": tau,
                    "accuracy": classification.accuracy,
                    "macro_f1": classification.macro_f1,
                    "nll": classification.nll,
                    "ece": classification.ece,
                    "coverage": prediction_set_result.coverage,
                    "mean_size": prediction_set_result.mean_size,
                    "median_size": prediction_set_result.median_size,
                    "size_p90": prediction_set_result.size_p90,
                    "empty_set_rate": prediction_set_result.empty_set_rate,
                    "full_set_rate": prediction_set_result.full_set_rate,
                    "sscv": conditional.sscv,
                    "class_coverage_gap": conditional.class_coverage_gap,
                    "class_coverage_max_deviation": conditional.class_coverage_max_deviation,
                    "class_coverages": json.dumps(conditional.class_coverages),
                    "tuning_nll": tuning_nll,
                    "confts_loss": confts_loss,
                    "rejected_temperatures": json.dumps(rejected),
                    "calibration_zero_count": calibration_diagnostics.zero_count,
                    "calibration_exactly_one_count": calibration_diagnostics.exactly_one_count,
                    "test_zero_count": test_diagnostics.zero_count,
                    "test_exactly_one_count": test_diagnostics.exactly_one_count,
                    "test_mean_max_probability": test_diagnostics.mean_max_probability,
                }
            )
    return rows
