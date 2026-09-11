"""Phase 8 multi-seed robustness evaluation and checkpointed grid execution."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from chf.conformal import evaluate_prediction_sets
from chf.models import fit_classifier
from chf.scaling import probabilities_from_logits

from .checkpoints import (
    CheckpointStore,
    atomic_write_csv,
    atomic_write_json,
    experiment_config_sha256,
)
from .protocol import code_version, dataset_name, evaluate_logits


FULL_GRID = "full_grid"
SENSITIVITY_GRID = "alpha_sensitivity"
PRIMARY_ONLY = "primary_only"
_SCOPE_ORDER = {PRIMARY_ONLY: 0, SENSITIVITY_GRID: 1, FULL_GRID: 2}


def validate_phase8_spec(spec: Mapping[str, Any]) -> None:
    """Reject incomplete or scientifically inconsistent Phase 8 grids."""
    if int(spec.get("phase", 0)) != 8:
        raise ValueError("Phase 8 specification must declare phase: 8")
    seeds = [int(value) for value in spec.get("seeds", ())]
    if len(seeds) != 10 or len(set(seeds)) != 10:
        raise ValueError("the initial Phase 8 design requires ten unique seeds")
    alphas = tuple(float(value) for value in spec["conformal_grid"]["alphas"])
    if alphas != (0.1, 0.05):
        raise ValueError("Phase 8 alphas must be ordered as [0.10, 0.05]")
    lambdas = tuple(float(value) for value in spec["conformal_grid"]["raps_lambda"])
    k_values = tuple(int(value) for value in spec["conformal_grid"]["raps_k_reg"])
    if any(value <= 0 for value in lambdas) or any(value < 0 for value in k_values):
        raise ValueError("RAPS lambda values must be positive and k_reg non-negative")
    scalings = tuple(spec["conformal_grid"]["scalings"])
    if scalings != ("base", "ts", "confts"):
        raise ValueError("Phase 8 must retain Base, TS, and ConfTS")
    removals = tuple(int(value) for value in spec["selection"]["removed_features"])
    if removals != (1, 2, 3, 4, 5):
        raise ValueError("Phase 8 subset sensitivity must contain removals 1 through 5")
    losses = tuple(float(value) for value in spec["selection"]["accuracy_loss"])
    if losses != (0.0, 0.005, 0.01, 0.02):
        raise ValueError("unexpected Phase 8 accuracy-loss sensitivity grid")
    datasets = spec.get("datasets", {})
    if set(datasets) != {"dry_bean", "covertype", "human_activity_recognition"}:
        raise ValueError("Phase 8 requires the three frozen Phase 7 datasets")


def phase8_grid_rows(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the exact 60 full-grid cells in deterministic order."""
    validate_phase8_spec(spec)
    grid = spec["conformal_grid"]
    rows: list[dict[str, Any]] = []
    for alpha in grid["alphas"]:
        for scaling in grid["scalings"]:
            rows.append(
                {
                    "alpha": float(alpha),
                    "scaling": str(scaling),
                    "score": "aps",
                    "raps_lambda": None,
                    "raps_k_reg": None,
                }
            )
            for lambda_value in grid["raps_lambda"]:
                for k_reg in grid["raps_k_reg"]:
                    rows.append(
                        {
                            "alpha": float(alpha),
                            "scaling": str(scaling),
                            "score": "raps",
                            "raps_lambda": float(lambda_value),
                            "raps_k_reg": int(k_reg),
                        }
                    )
    return rows


def derive_seed_config(
    base_config: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    dataset_key: str,
    seed: int,
    resume: bool,
) -> dict[str, Any]:
    """Apply Phase 8 controls without modifying the frozen Phase 7 config."""
    validate_phase8_spec(spec)
    config = copy.deepcopy(dict(base_config))
    config["phase"] = 8
    config["seed"] = int(seed)
    config["experiment_name"] = f"phase8_{dataset_key}_seed_{seed}"
    primary_alpha = float(spec["selection"]["primary_alpha"])
    primary_accuracy_loss = float(spec["selection"]["primary_accuracy_loss"])
    config["conformal"]["alpha"] = primary_alpha
    config["selection"]["max_accuracy_loss"] = primary_accuracy_loss
    config["harm"]["constraints"]["max_accuracy_loss"] = primary_accuracy_loss
    max_removals = max(int(value) for value in spec["selection"]["removed_features"])
    config["progressive_selection"]["max_removals"] = max_removals
    config["progressive_selection"]["continue_path_after_stop"] = True
    config["progressive_selection"]["selection_only"] = True
    config["baselines"]["target_removals"] = list(
        spec["selection"]["removed_features"]
    )
    config["baselines"]["include_progressive_steps"] = True
    config["baselines"]["selection_only"] = True
    config["checkpointing"] = {"enabled": True, "resume": bool(resume)}
    config["phase8"] = {
        "protocol_version": int(spec["protocol_version"]),
        "primary_alpha": primary_alpha,
        "primary_accuracy_loss": primary_accuracy_loss,
        "accuracy_loss_grid": list(spec["selection"]["accuracy_loss"]),
        "removed_features_grid": list(spec["selection"]["removed_features"]),
        "conformal_grid": copy.deepcopy(spec["conformal_grid"]),
    }
    return config


def _selection_scope(selection: Mapping[str, Any]) -> str:
    method = str(selection["method"])
    if method == "random":
        return PRIMARY_ONLY
    if method == "all_features" or bool(selection.get("phase8_primary_selected", False)):
        return FULL_GRID
    return SENSITIVITY_GRID


def phase8_accuracy_loss_choices(
    selections: pd.DataFrame,
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Choose a tuning-only progressive subset under every loss allowance."""
    proposed = selections.loc[
        selections["method"].isin(
            {"conformal_harm_one_shot", "conformal_harm_recursive"}
        )
    ].copy()
    required = {
        "model",
        "method",
        "selected_indices",
        "selected_features",
        "n_features",
        "tuning_n_removed",
        "tuning_max_accuracy_loss",
        "tuning_max_coverage_shortfall",
        "tuning_efficiency_gain",
        "tuning_conditional_violation",
    }
    if missing := required.difference(proposed.columns):
        raise ValueError(
            f"proposed sensitivity paths are missing columns: {sorted(missing)}"
        )
    coverage_limit = float(config["harm"]["constraints"]["max_coverage_shortfall"])
    rows: list[dict[str, Any]] = []
    for (model, method), path in proposed.groupby(["model", "method"], sort=False):
        path = path.sort_values("tuning_n_removed")
        if not path["tuning_n_removed"].astype(int).eq(0).any():
            raise ValueError("every progressive sensitivity path requires step zero")
        for loss_limit in spec["selection"]["accuracy_loss"]:
            eligible = path.loc[
                path["tuning_max_accuracy_loss"].le(float(loss_limit) + 1e-12)
                & path["tuning_max_coverage_shortfall"].le(
                    coverage_limit + 1e-12
                )
                & path["tuning_efficiency_gain"].ge(-1e-12)
            ].copy()
            if eligible.empty:
                eligible = path.loc[path["tuning_n_removed"].astype(int).eq(0)].copy()
            chosen = eligible.sort_values(
                [
                    "tuning_efficiency_gain",
                    "tuning_conditional_violation",
                    "tuning_n_removed",
                ],
                ascending=[False, True, True],
                kind="stable",
            ).iloc[0]
            rows.append(
                {
                    "model": model,
                    "method": method,
                    "accuracy_loss_limit": float(loss_limit),
                    "coverage_shortfall_limit": coverage_limit,
                    "selected_indices": chosen["selected_indices"],
                    "selected_features": chosen["selected_features"],
                    "n_features": int(chosen["n_features"]),
                    "n_removed": int(chosen["tuning_n_removed"]),
                    "tuning_max_accuracy_loss": float(
                        chosen["tuning_max_accuracy_loss"]
                    ),
                    "tuning_max_coverage_shortfall": float(
                        chosen["tuning_max_coverage_shortfall"]
                    ),
                    "tuning_efficiency_gain": float(
                        chosen["tuning_efficiency_gain"]
                    ),
                    "choice_partition": "outer_tune_crossfit_only",
                }
            )
    return pd.DataFrame(rows)


def _cells_for_scope(spec: Mapping[str, Any], scope: str) -> list[dict[str, Any]]:
    rows = phase8_grid_rows(spec)
    if scope == FULL_GRID:
        return rows
    filtered = [
        row
        for row in rows
        if row["score"] == "aps" and row["scaling"] == "base"
    ]
    if scope == SENSITIVITY_GRID:
        return filtered
    if scope == PRIMARY_ONLY:
        primary_alpha = float(spec["selection"]["primary_alpha"])
        return [row for row in filtered if row["alpha"] == primary_alpha]
    raise ValueError(f"unknown Phase 8 grid scope: {scope}")


def _evaluate_cells(
    *,
    base_config: Mapping[str, Any],
    spec: Mapping[str, Any],
    scope: str,
    logits_tune: np.ndarray,
    logits_calibration: np.ndarray,
    logits_test: np.ndarray,
    labels_tune: np.ndarray,
    labels_calibration: np.ndarray,
    labels_test: np.ndarray,
    calibration_uniforms: np.ndarray,
    test_uniforms: np.ndarray,
    test_groups: np.ndarray | None,
    seed: int,
) -> pd.DataFrame:
    requested = _cells_for_scope(spec, scope)
    rows: list[dict[str, Any]] = []
    batches: dict[tuple[float, str, float | None, int | None], set[str]] = {}
    for cell in requested:
        key = (
            float(cell["alpha"]),
            str(cell["score"]),
            cell["raps_lambda"],
            cell["raps_k_reg"],
        )
        batches.setdefault(key, set()).add(str(cell["scaling"]))
    for key, scalings in batches.items():
        alpha, score, lambda_value, k_reg = key
        config = copy.deepcopy(dict(base_config))
        config["conformal"]["alpha"] = alpha
        if score == "raps":
            config["conformal"]["raps_lambda"] = float(lambda_value)
            config["conformal"]["raps_k_reg"] = int(k_reg)
        evaluated = evaluate_logits(
            logits_tune=logits_tune,
            logits_calibration=logits_calibration,
            logits_test=logits_test,
            labels_tune=labels_tune,
            labels_calibration=labels_calibration,
            labels_test=labels_test,
            config=config,
            calibration_uniforms=calibration_uniforms,
            test_uniforms=test_uniforms,
            # Keep the inner ConfTS split identical across every alpha and
            # RAPS setting; only the scientific grid parameters may change.
            seed=seed,
            included_scores=(score,),
            included_scalings=tuple(
                value for value in ("base", "ts", "confts") if value in scalings
            ),
            test_groups=test_groups,
        )
        for row in evaluated:
            row["grid_scope"] = scope
            if score == "aps":
                row["raps_lambda"] = np.nan
                row["raps_k_reg"] = np.nan
            rows.append(row)
    result = pd.DataFrame(rows)
    expected = len(requested)
    if len(result) != expected:
        raise RuntimeError(
            f"Phase 8 grid produced {len(result)} rows; expected {expected}"
        )
    return result


def _subject_metrics(
    *,
    evaluated: pd.DataFrame,
    logits_test: np.ndarray,
    labels_test: np.ndarray,
    test_groups: np.ndarray,
    test_uniforms: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in evaluated.to_dict("records"):
        probabilities = probabilities_from_logits(
            logits_test, float(result["temperature"])
        )
        prediction_sets = evaluate_prediction_sets(
            probabilities,
            labels_test,
            float(result["threshold"]),
            str(result["score"]),
            u_values=test_uniforms,
            k_reg=(1 if pd.isna(result["raps_k_reg"]) else int(result["raps_k_reg"])),
            lambda_reg=(
                0.001
                if pd.isna(result["raps_lambda"])
                else float(result["raps_lambda"])
            ),
        )
        included = prediction_sets.included
        covered = included[np.arange(len(labels_test)), labels_test]
        sizes = included.sum(axis=1)
        predictions = probabilities.argmax(axis=1)
        for subject in np.unique(test_groups):
            mask = test_groups == subject
            rows.append(
                {
                    "alpha": result["alpha"],
                    "scaling": result["scaling"],
                    "score": result["score"],
                    "raps_lambda": result["raps_lambda"],
                    "raps_k_reg": result["raps_k_reg"],
                    "subject_id": subject.item() if isinstance(subject, np.generic) else subject,
                    "window_count": int(mask.sum()),
                    "accuracy": float(np.mean(predictions[mask] == labels_test[mask])),
                    "coverage": float(covered[mask].mean()),
                    "mean_size": float(sizes[mask].mean()),
                    "size_p90": float(np.quantile(sizes[mask], 0.9)),
                }
            )
    return pd.DataFrame(rows)


def _selection_digest(selections: pd.DataFrame) -> str:
    encoded = selections.sort_values(
        ["model", "method", "target_size", "repetition"]
    ).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_phase8_grid(
    *,
    selections: pd.DataFrame,
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    dataset: Any,
    split: Any,
    split_identifier: str,
    selection_identifier: str,
    output_dir: Path,
    repository_root: Path,
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every unique frozen subset once, checkpoint it, and evaluate its grid."""
    validate_phase8_spec(spec)
    required = {
        "model", "method", "target_size", "n_features", "selected_indices",
        "selected_features", "ranking_source", "selection_seed", "repetition",
        "subset_frozen_before_final_calibration", "selection_data_id",
        "phase8_primary_selected",
    }
    if missing := required.difference(selections.columns):
        raise ValueError(f"Phase 8 selections are missing columns: {sorted(missing)}")
    if not selections["selection_data_id"].eq(selection_identifier).all():
        raise ValueError("Phase 8 selections use different selection data")
    seed = int(config["seed"])
    version = code_version(repository_root)
    manifest = {
        "schema_version": 1,
        "stage": "phase8_final_grid",
        "experiment_name": config["experiment_name"],
        "seed": seed,
        "split_id": split_identifier,
        "selection_data_id": selection_identifier,
        "selection_sha256": _selection_digest(selections),
        "config_sha256": experiment_config_sha256(config),
        "phase8_grid": phase8_grid_rows(spec),
        "code_version": version,
    }
    checkpoint_store = CheckpointStore(
        output_dir / "checkpoints" / "final_grid", manifest
    )
    checkpoint_store.initialize(resume=resume)
    calibration_uniforms = np.random.default_rng(seed + 800_001).random(
        len(split.calibration)
    )
    test_uniforms = np.random.default_rng(seed + 800_002).random(len(split.test))
    dataset_groups = getattr(dataset, "groups", None)
    test_groups = (
        None if dataset_groups is None else np.asarray(dataset_groups)[split.test]
    )
    metadata_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    selections = selections.copy()
    selections["grid_scope"] = selections.apply(_selection_scope, axis=1)
    fit_groups = selections.groupby(["model", "selected_indices"], sort=False)
    for (model_name, selected_indices), fit_selections in fit_groups:
        scope = max(
            fit_selections["grid_scope"], key=lambda value: _SCOPE_ORDER[value]
        )
        checkpoint_key = {
            "artifact": "grid",
            "model": model_name,
            "selected_indices": selected_indices,
            "scope": scope,
        }
        evaluated = checkpoint_store.load_frame(checkpoint_key)
        subject_checkpoint_key = {**checkpoint_key, "artifact": "subjects"}
        subjects = checkpoint_store.load_frame(subject_checkpoint_key)
        if evaluated is None:
            selected = tuple(json.loads(selected_indices))
            fitted = fit_classifier(
                dataset.features[split.train][:, selected],
                dataset.labels[split.train],
                config["models"][model_name],
                seed=seed,
            )
            logits_tune = fitted.logits(dataset.features[split.tune][:, selected])
            logits_calibration = fitted.logits(
                dataset.features[split.calibration][:, selected]
            )
            logits_test = fitted.logits(dataset.features[split.test][:, selected])
            evaluated = _evaluate_cells(
                base_config=config,
                spec=spec,
                scope=scope,
                logits_tune=logits_tune,
                logits_calibration=logits_calibration,
                logits_test=logits_test,
                labels_tune=dataset.labels[split.tune],
                labels_calibration=dataset.labels[split.calibration],
                labels_test=dataset.labels[split.test],
                calibration_uniforms=calibration_uniforms,
                test_uniforms=test_uniforms,
                test_groups=test_groups,
                seed=seed + 800_000,
            )
            checkpoint_store.save_frame(checkpoint_key, evaluated)
            if test_groups is not None:
                subjects = _subject_metrics(
                    evaluated=evaluated,
                    logits_test=logits_test,
                    labels_test=dataset.labels[split.test],
                    test_groups=test_groups,
                    test_uniforms=test_uniforms,
                )
                checkpoint_store.save_frame(subject_checkpoint_key, subjects)
        elif test_groups is not None and subjects is None:
            raise ValueError("HAR grid checkpoint lacks its subject-level shard")
        print(
            f"Phase 8 grid: {model_name} | {len(json.loads(selected_indices))} "
            f"features | {scope}",
            flush=True,
        )
        for selection in fit_selections.to_dict("records"):
            selection_scope = str(selection.pop("grid_scope"))
            allowed = _cells_for_scope(spec, selection_scope)
            allowed_keys = {
                (
                    row["alpha"], row["scaling"], row["score"],
                    row["raps_lambda"], row["raps_k_reg"],
                )
                for row in allowed
            }
            for result in evaluated.to_dict("records"):
                result_key = (
                    float(result["alpha"]), str(result["scaling"]),
                    str(result["score"]),
                    None if pd.isna(result["raps_lambda"]) else float(result["raps_lambda"]),
                    None if pd.isna(result["raps_k_reg"]) else int(result["raps_k_reg"]),
                )
                if result_key not in allowed_keys:
                    continue
                metadata_rows.append(
                    {
                        "experiment": config["experiment_name"],
                        "dataset": dataset_name(dataset),
                        "seed": seed,
                        "split_id": split_identifier,
                        "selection_data_id": selection_identifier,
                        "code_version": version,
                        "classifier_refit": True,
                        "final_calibration_used_for_selection": False,
                        "final_test_used_for_selection": False,
                        **selection,
                        **result,
                    }
                )
            if subjects is not None:
                allowed_subjects = subjects.merge(
                    pd.DataFrame(list(allowed_keys), columns=[
                        "alpha", "scaling", "score", "raps_lambda", "raps_k_reg"
                    ]),
                    on=["alpha", "scaling", "score", "raps_lambda", "raps_k_reg"],
                    how="inner",
                )
                for subject in allowed_subjects.to_dict("records"):
                    subject_rows.append(
                        {
                            "experiment": config["experiment_name"],
                            "dataset": dataset_name(dataset),
                            "seed": seed,
                            "split_id": split_identifier,
                            "selection_data_id": selection_identifier,
                            "code_version": version,
                            "classifier_refit": True,
                            "final_calibration_used_for_selection": False,
                            "final_test_used_for_selection": False,
                            **selection,
                            **subject,
                        }
                    )
    results = _attach_reference_results(pd.DataFrame(metadata_rows))
    subject_results = _attach_subject_reference_results(pd.DataFrame(subject_rows))
    _validate_and_write_grid(
        results=results,
        subject_results=subject_results,
        selections=selections,
        spec=spec,
        output_dir=output_dir,
        manifest=manifest,
    )
    return results, subject_results


def _reference_condition_columns() -> list[str]:
    return ["model", "alpha", "scaling", "score", "raps_lambda", "raps_k_reg"]


def _attach_reference_results(results: pd.DataFrame) -> pd.DataFrame:
    condition = _reference_condition_columns()
    metrics = ["accuracy", "coverage", "mean_size", "sscv", "class_coverage_max_deviation"]
    reference = results.loc[results["method"] == "all_features", condition + metrics]
    if reference.duplicated(condition).any():
        raise ValueError("Phase 8 all-feature result is not unique by condition")
    reference = reference.rename(columns={value: f"all_features_{value}" for value in metrics})
    merged = results.merge(reference, on=condition, how="left", validate="many_to_one")
    merged["accuracy_loss_vs_all"] = merged["all_features_accuracy"] - merged["accuracy"]
    merged["coverage_delta_vs_all"] = merged["coverage"] - merged["all_features_coverage"]
    merged["mean_size_reduction_vs_all"] = merged["all_features_mean_size"] - merged["mean_size"]
    merged["sscv_delta_vs_all"] = merged["sscv"] - merged["all_features_sscv"]
    return merged


def _attach_subject_reference_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    condition = [*_reference_condition_columns(), "subject_id"]
    metrics = ["accuracy", "coverage", "mean_size", "size_p90"]
    reference = results.loc[results["method"] == "all_features", condition + metrics]
    if reference.duplicated(condition).any():
        raise ValueError("Phase 8 subject reference is not unique by condition")
    reference = reference.rename(columns={value: f"all_features_{value}" for value in metrics})
    merged = results.merge(reference, on=condition, how="left", validate="many_to_one")
    for metric in metrics:
        merged[f"{metric}_difference_vs_all"] = merged[metric] - merged[f"all_features_{metric}"]
    merged["mean_size_reduction_vs_all"] = -merged["mean_size_difference_vs_all"]
    return merged


def _validate_and_write_grid(
    *,
    results: pd.DataFrame,
    subject_results: pd.DataFrame,
    selections: pd.DataFrame,
    spec: Mapping[str, Any],
    output_dir: Path,
    manifest: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    key = [
        "model", "method", "target_size", "selected_indices", "repetition",
        "alpha", "scaling", "score", "raps_lambda", "raps_k_reg",
    ]
    numeric = [
        "temperature", "threshold", "accuracy", "coverage", "mean_size", "sscv"
    ]
    expected_counts = {FULL_GRID: 60, SENSITIVITY_GRID: 2, PRIMARY_ONLY: 1}
    observed_counts = results.groupby(
        ["model", "method", "target_size", "selected_indices", "repetition"],
        dropna=False,
    ).size()
    expected = selections.set_index(
        ["model", "method", "target_size", "selected_indices", "repetition"]
    )["grid_scope"].map(expected_counts)
    expected = expected.groupby(level=list(range(5))).max()
    aligned_counts = pd.concat(
        [
            observed_counts.rename("observed"),
            expected.rename("expected"),
        ],
        axis=1,
    )
    checks = {
        "result_keys_unique": not results.duplicated(key).any(),
        "selection_grid_counts_exact": bool(
            aligned_counts.notna().all().all()
            and aligned_counts["observed"].astype(int).eq(
                aligned_counts["expected"].astype(int)
            ).all()
        ),
        "metrics_finite": bool(np.isfinite(results[numeric].to_numpy()).all()),
        "subsets_frozen_before_calibration": bool(
            results["subset_frozen_before_final_calibration"].astype(bool).all()
        ),
        "calibration_and_test_excluded_from_selection": bool(
            (~results["final_calibration_used_for_selection"].astype(bool)).all()
            and (~results["final_test_used_for_selection"].astype(bool)).all()
        ),
        "no_zero_or_saturated_probabilities": bool(
            (results[[
                "calibration_zero_count", "calibration_exactly_one_count",
                "test_zero_count", "test_exactly_one_count",
            ]] == 0).all().all()
        ),
        "subject_rows_present_for_grouped_data": bool(
            not subject_results.empty
            if "group_coverage_count" in results.columns
            else subject_results.empty
        ),
    }
    record = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "manifest": dict(manifest),
        "grid": {
            "full_cells_per_primary_subset": len(phase8_grid_rows(spec)),
            "alphas": list(spec["conformal_grid"]["alphas"]),
            "raps_lambda": list(spec["conformal_grid"]["raps_lambda"]),
            "raps_k_reg": list(spec["conformal_grid"]["raps_k_reg"]),
            "scalings": list(spec["conformal_grid"]["scalings"]),
        },
        "observed_result_rows": len(results),
        "observed_subject_rows": len(subject_results),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Phase 8 grid validation failed: {failed}")
    atomic_write_csv(output_dir / "phase8_results.csv", results)
    if not subject_results.empty:
        atomic_write_csv(output_dir / "phase8_subject_results.csv", subject_results)
    atomic_write_json(output_dir / "phase8_grid_protocol.json", record)
