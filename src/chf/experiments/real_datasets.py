import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from chf.data import make_four_way_split, save_split_artifact

from .baselines import run_required_baselines
from .protocol import (
    dataset_from_config,
    dataset_name,
    selection_data_id,
    selection_data_indices,
    split_id,
)
from .scaling_interaction import SCALINGS, SCORES, run_scaling_interaction


def run_real_dataset_benchmark(
    config: Mapping[str, Any], output_dir: Path, repository_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the frozen Phase 4-6 protocol on one configured real dataset.

    The function intentionally delegates selection and evaluation to the same
    implementations validated on synthetic data. It adds real-data provenance,
    split audits, and cross-stage checks around those unchanged protocols.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = dataset_from_config(config, repository_root)
    _validate_dataset_declaration(config, dataset)
    split = _make_split(config, dataset.labels)
    identifier = split_id(split)
    seed = int(config["seed"])
    selection_train, selection_tune = selection_data_indices(
        config, split, dataset.labels, seed=seed
    )
    selection_identifier = selection_data_id(selection_train, selection_tune)
    provenance = _dataset_provenance(
        config,
        dataset,
        split,
        identifier,
        selection_train,
        selection_tune,
        selection_identifier,
    )
    save_split_artifact(
        output_dir / "phase7_split_indices.npz",
        split,
        seed=int(config["seed"]),
        metadata=provenance,
    )
    np.savez_compressed(
        output_dir / "phase7_selection_indices.npz",
        train=selection_train,
        tune=selection_tune,
        selection_data_id=np.asarray(selection_identifier),
    )
    split_distribution = _split_distribution(dataset, split)
    split_distribution.to_csv(output_dir / "phase7_split_distribution.csv", index=False)
    (output_dir / "dataset_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )

    selections, baseline_results, _ = run_required_baselines(
        config, output_dir, repository_root
    )
    factorial_results, interactions, rank_stability, _ = run_scaling_interaction(
        config,
        output_dir,
        repository_root,
        selections_path=output_dir / "baseline_selections.csv",
    )
    main_table = _main_table(factorial_results)
    main_table.to_csv(output_dir / "phase7_main_table.csv", index=False)
    _write_protocol_record(
        config=config,
        dataset=dataset,
        split=split,
        identifier=identifier,
        output_dir=output_dir,
        selections=selections,
        baseline_results=baseline_results,
        factorial_results=factorial_results,
        interactions=interactions,
        rank_stability=rank_stability,
        selection_train=selection_train,
        selection_tune=selection_tune,
        selection_identifier=selection_identifier,
    )
    return selections, factorial_results, interactions, main_table


def _validate_dataset_declaration(config: Mapping[str, Any], dataset: Any) -> None:
    declaration = config["dataset"]
    expected_samples = int(declaration["n_samples"])
    expected_features = int(declaration["n_features"])
    expected_classes = int(declaration["n_classes"])
    observed_classes = np.unique(dataset.labels)
    if dataset.features.shape != (expected_samples, expected_features):
        raise ValueError(
            "dataset shape differs from config: "
            f"expected {(expected_samples, expected_features)}, "
            f"observed {dataset.features.shape}"
        )
    if len(observed_classes) != expected_classes:
        raise ValueError(
            "dataset class count differs from config: "
            f"expected {expected_classes}, observed {len(observed_classes)}"
        )
    if expected_classes < 3:
        raise ValueError("Phase 7 requires a multiclass dataset with at least 3 classes")
    if not np.isfinite(dataset.features).all():
        raise ValueError("Phase 7 features must all be finite before preprocessing")


def _make_split(config: Mapping[str, Any], labels: np.ndarray) -> Any:
    split_values = config["split"]
    sizes = tuple(
        int(split_values[name])
        for name in ("train", "tune", "calibration", "test")
    )
    return make_four_way_split(labels, sizes, int(config["seed"]))


def _dataset_provenance(
    config: Mapping[str, Any],
    dataset: Any,
    split: Any,
    identifier: str,
    selection_train: np.ndarray,
    selection_tune: np.ndarray,
    selection_identifier: str,
) -> dict[str, Any]:
    return {
        "experiment_name": config["experiment_name"],
        "phase": 7,
        "dataset": dataset_name(dataset),
        "source_url": str(dataset.source_url),
        "archive_sha256": str(dataset.archive_sha256),
        "n_samples": int(dataset.features.shape[0]),
        "n_features": int(dataset.features.shape[1]),
        "n_classes": int(len(np.unique(dataset.labels))),
        "feature_names": list(dataset.feature_names),
        "feature_manifest": [
            {
                "index": int(feature.index),
                "name": feature.name,
                "role": feature.role,
                "source_feature": feature.source_feature,
            }
            for feature in dataset.feature_manifest
        ],
        "class_names": list(dataset.class_names),
        "split_id": identifier,
        "split_sizes": {
            name: int(len(getattr(split, name)))
            for name in ("train", "tune", "calibration", "test")
        },
        "selection_data_id": selection_identifier,
        "selection_sizes": {
            "train": int(len(selection_train)),
            "tune": int(len(selection_tune)),
        },
        "selection_budget": dict(config.get("selection_budget", {})),
        "preprocessing": (
            "selection models fit preprocessing on selection-train only; "
            "frozen final models refit preprocessing on full outer train"
        ),
    }


def _split_distribution(dataset: Any, split: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for partition in ("train", "tune", "calibration", "test"):
        indices = getattr(split, partition)
        labels, counts = np.unique(dataset.labels[indices], return_counts=True)
        for label, count in zip(labels, counts, strict=True):
            rows.append(
                {
                    "partition": partition,
                    "class_index": int(label),
                    "class_name": dataset.class_names[int(label)],
                    "count": int(count),
                    "proportion": float(count / len(indices)),
                }
            )
    return pd.DataFrame(rows)


def _main_table(results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "model",
        "method",
        "selection_type",
        "target_size",
        "n_features",
        "scaling",
        "score",
        "temperature",
        "threshold",
        "accuracy",
        "macro_f1",
        "ece",
        "coverage",
        "mean_size",
        "size_p90",
        "sscv",
        "class_coverage_max_deviation",
        "accuracy_loss_vs_all",
        "mean_size_reduction_vs_all",
    ]
    deterministic = results.loc[
        results["repetition"].astype(int) == -1, columns
    ].rename(
        columns={
            "accuracy_loss_vs_all": "accuracy_loss",
            "mean_size_reduction_vs_all": "mean_size_reduction",
        }
    )
    return deterministic.sort_values(
        ["model", "target_size", "method", "score", "scaling"]
    ).reset_index(drop=True)


def _write_protocol_record(
    *,
    config: Mapping[str, Any],
    dataset: Any,
    split: Any,
    identifier: str,
    output_dir: Path,
    selections: pd.DataFrame,
    baseline_results: pd.DataFrame,
    factorial_results: pd.DataFrame,
    interactions: pd.DataFrame,
    rank_stability: pd.DataFrame,
    selection_train: np.ndarray,
    selection_tune: np.ndarray,
    selection_identifier: str,
) -> None:
    deterministic = selections.loc[selections["repetition"].astype(int) == -1]
    expected_factorial_rows = len(deterministic) * len(SCALINGS) * len(SCORES)
    target = 1.0 - float(config["conformal"]["alpha"])
    coverage_tolerance = float(
        config.get("stability", {}).get("max_coverage_deviation", 0.03)
    )
    class_counts = _split_distribution(dataset, split).groupby("partition").size()
    required_methods = {
        "all_features",
        "random",
        "mutual_information",
        "permutation_importance",
        "rfe",
        "shap",
        "crfe",
        "conformal_harm_one_shot",
        "conformal_harm_recursive",
    }
    checks = {
        "official_source_checksum_verified": str(dataset.archive_sha256)
        == str(config["dataset"]["archive_sha256"]),
        "declared_shape_verified": dataset.features.shape
        == (
            int(config["dataset"]["n_samples"]),
            int(config["dataset"]["n_features"]),
        ),
        "multiclass_task": len(np.unique(dataset.labels)) >= 3,
        "four_way_split_disjoint": _split_is_disjoint(split),
        "every_class_present_in_every_partition": bool(
            (class_counts == len(dataset.class_names)).all()
        ),
        "identical_split_across_subprotocols": bool(
            baseline_results["split_id"].eq(identifier).all()
            and factorial_results["split_id"].eq(identifier).all()
        ),
        "selection_rows_within_outer_partitions": bool(
            np.isin(selection_train, split.train).all()
            and np.isin(selection_tune, split.tune).all()
            and np.intersect1d(selection_train, selection_tune).size == 0
        ),
        "identical_selection_data_across_subprotocols": bool(
            baseline_results["selection_data_id"].eq(
                selection_identifier
            ).all()
            and factorial_results["selection_data_id"].eq(
                selection_identifier
            ).all()
        ),
        "all_required_methods_present": required_methods.issubset(
            set(selections["method"])
        ),
        "full_factorial_present": len(factorial_results) == expected_factorial_rows
        and set(factorial_results["scaling"]) == set(SCALINGS)
        and set(factorial_results["score"]) == set(SCORES),
        "fresh_thresholds_finite": bool(
            np.isfinite(factorial_results["threshold"]).all()
        ),
        "final_metrics_finite": bool(
            np.isfinite(
                factorial_results[
                    ["accuracy", "coverage", "mean_size", "sscv"]
                ]
            ).all().all()
        ),
        "coverage_within_sampling_tolerance": bool(
            factorial_results["coverage"].sub(target).abs().le(coverage_tolerance).all()
            and baseline_results["coverage"]
            .sub(target)
            .abs()
            .le(coverage_tolerance)
            .all()
        ),
        "no_extreme_probability_artifacts": bool(
            factorial_results[
                [
                    "calibration_zero_count",
                    "calibration_exactly_one_count",
                    "test_zero_count",
                    "test_exactly_one_count",
                ]
            ].eq(0).all().all()
        ),
        "subsets_frozen_before_final_calibration": bool(
            factorial_results["subset_frozen_before_final_calibration"].all()
        ),
        "final_partitions_used_once_per_pipeline": bool(
            factorial_results["final_calibration_used_once_per_pipeline"].all()
            and factorial_results["final_test_used_once_per_pipeline"].all()
        ),
        "interaction_rows_finite": bool(
            len(interactions) > 0
            and np.isfinite(interactions["interaction_size_gain"]).all()
        ),
        "rank_comparisons_present": len(rank_stability) > 0,
    }
    record = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "protocol": {
            "preprocessing_fit_partition": (
                "selection train during search; full outer train after freeze"
            ),
            "classifier_fit_partition": (
                "selection train during search; full outer train after freeze"
            ),
            "ranking_and_subset_size_partition": (
                "fixed selection train/tune rows only"
            ),
            "selection_data_id": selection_identifier,
            "selection_train_rows": int(len(selection_train)),
            "selection_tune_rows": int(len(selection_tune)),
            "final_calibration_access": "once after every choice was frozen",
            "final_test_access": "once after every choice was frozen",
            "paired_randomization": "shared uniforms within split and stage",
            "models": list(config["models"]),
            "pipelines": [
                f"{scaling}+{score}" for scaling in SCALINGS for score in SCORES
            ],
        },
        "split_id": identifier,
        "observed_selection_rows": int(len(selections)),
        "observed_factorial_rows": int(len(factorial_results)),
        "expected_factorial_rows": int(expected_factorial_rows),
    }
    (output_dir / "phase7_protocol.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
    )
    if record["status"] != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Phase 7 validation failed: {failed}")


def _split_is_disjoint(split: Any) -> bool:
    parts = [split.train, split.tune, split.calibration, split.test]
    return all(
        np.intersect1d(left, right).size == 0
        for left_index, left in enumerate(parts)
        for right in parts[left_index + 1 :]
    )
