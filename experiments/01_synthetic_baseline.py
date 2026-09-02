import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from chf.data import (
    class_separation_ratios,
    make_four_way_split,
    save_split_artifact,
)
from chf.experiments import code_version, dataset_from_config, evaluate_logits, split_id
from chf.models import fit_classifier


def _save_feature_manifest(dataset, train_indices: np.ndarray, output_dir: Path) -> None:
    train_features = dataset.features[train_indices]
    train_labels = dataset.labels[train_indices]
    ratios = class_separation_ratios(train_features, train_labels)
    rows = []
    for feature, ratio in zip(dataset.feature_manifest, ratios, strict=True):
        row = asdict(feature)
        row["train_class_separation_ratio"] = float(ratio)
        if feature.source_feature is None:
            row["train_source_correlation"] = np.nan
        else:
            source_index = dataset.feature_names.index(feature.source_feature)
            row["train_source_correlation"] = float(
                np.corrcoef(
                    train_features[:, feature.index], train_features[:, source_index]
                )[0, 1]
            )
        rows.append(row)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "feature_manifest.csv", index=False)

    role_medians = manifest.groupby("role")["train_class_separation_ratio"].median()
    required_roles = {"strong", "weak", "redundant", "noise"}
    if not required_roles.issubset(role_medians.index):
        raise ValueError("the controlled dataset must contain all four core feature roles")
    redundant_correlations = manifest.loc[
        manifest["role"] == "redundant", "train_source_correlation"
    ]
    recoverable = (
        role_medians["strong"] > role_medians["weak"] > role_medians["noise"]
        and role_medians["redundant"] > role_medians["weak"]
        and redundant_correlations.abs().min() > 0.9
    )
    if not recoverable:
        raise RuntimeError("synthetic feature roles failed the training-only recovery check")
    (output_dir / "feature_role_check.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "criterion": "median class separation: strong > weak > noise; "
                "redundant > weak; all redundant-source |correlations| > 0.9",
                "median_class_separation_by_role": {
                    role: float(value) for role, value in role_medians.items()
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def run(config: dict[str, Any], output_dir: Path, repository_root: Path) -> pd.DataFrame:
    seed = int(config["seed"])
    dataset = dataset_from_config(config)
    split_config = config["split"]
    sizes = tuple(
        int(split_config[name])
        for name in ("train", "tune", "calibration", "test")
    )
    split = make_four_way_split(dataset.labels, sizes, seed)
    split_identifier = split_id(split)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_split_artifact(
        output_dir / "split_indices.npz",
        split,
        seed=seed,
        metadata={
            "experiment_name": config["experiment_name"],
            "n_samples": len(dataset.labels),
            "split_id": split_identifier,
        },
    )
    _save_feature_manifest(dataset, split.train, output_dir)

    conformal_config = config["conformal"]
    alpha = float(conformal_config["alpha"])
    raps_lambda = float(conformal_config["raps_lambda"])
    raps_k_reg = int(conformal_config["raps_k_reg"])
    calibration_uniforms = np.random.default_rng(seed + 10_001).random(
        len(split.calibration)
    )
    test_uniforms = np.random.default_rng(seed + 20_001).random(len(split.test))
    selected_features = json.dumps(dataset.feature_names)
    version = code_version(repository_root)
    result_rows: list[dict[str, Any]] = []

    for model_name, model_config in config["models"].items():
        started = time.perf_counter()
        fitted = fit_classifier(
            dataset.features[split.train],
            dataset.labels[split.train],
            model_config,
            seed=seed,
        )
        fit_seconds = time.perf_counter() - started
        evaluated = evaluate_logits(
            logits_tune=fitted.logits(dataset.features[split.tune]),
            logits_calibration=fitted.logits(dataset.features[split.calibration]),
            logits_test=fitted.logits(dataset.features[split.test]),
            labels_tune=dataset.labels[split.tune],
            labels_calibration=dataset.labels[split.calibration],
            labels_test=dataset.labels[split.test],
            config=config,
            calibration_uniforms=calibration_uniforms,
            test_uniforms=test_uniforms,
            seed=seed,
        )
        metadata = {
            "experiment": config["experiment_name"],
            "dataset": "controlled_multiclass",
            "model": model_name,
            "seed": seed,
            "split_id": split_identifier,
            "selected_features": selected_features,
            "n_features": len(dataset.feature_names),
        }
        result_rows.extend(
            {
                **metadata,
                **row,
                "fit_seconds": fit_seconds,
                "code_version": version,
            }
            for row in evaluated
        )

    results = pd.DataFrame(result_rows)
    results.to_csv(output_dir / "baseline_results.csv", index=False)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    expected_rows = len(config["models"]) * len(conformal_config["scores"]) * 3
    stability_config = config.get("stability", {})
    max_coverage_deviation = float(
        stability_config.get("max_coverage_deviation", 0.03)
    )
    minimum_accuracy = float(stability_config.get("minimum_accuracy", 0.5))
    stability_checks = {
        "expected_result_rows": len(results) == expected_rows,
        "finite_core_metrics": bool(
            np.isfinite(
                results[
                    [
                        "temperature",
                        "threshold",
                        "accuracy",
                        "macro_f1",
                        "nll",
                        "ece",
                        "coverage",
                        "mean_size",
                        "sscv",
                        "class_coverage_gap",
                    ]
                ].to_numpy()
            ).all()
        ),
        "coverage_within_tolerance": bool(
            ((results["coverage"] - (1 - alpha)).abs() <= max_coverage_deviation).all()
        ),
        "classifier_accuracy_above_floor": bool(
            (results["accuracy"] >= minimum_accuracy).all()
        ),
        "no_zero_or_saturated_probabilities": bool(
            (
                results[
                    [
                        "calibration_zero_count",
                        "calibration_exactly_one_count",
                        "test_zero_count",
                        "test_exactly_one_count",
                    ]
                ]
                == 0
            )
            .all()
            .all()
        ),
    }
    stability_record = {
        "status": "PASS" if all(stability_checks.values()) else "FAIL",
        "checks": stability_checks,
        "coverage_target": 1 - alpha,
        "max_coverage_deviation": max_coverage_deviation,
        "minimum_accuracy": minimum_accuracy,
        "sscv_size_strata": [[0, 1], [2, 3], [4, 10]],
    }
    (output_dir / "baseline_stability.json").write_text(
        json.dumps(stability_record, indent=2, sort_keys=True), encoding="utf-8"
    )
    if stability_record["status"] != "PASS":
        failed = [name for name, passed in stability_checks.items() if not passed]
        raise RuntimeError(f"Phase 1 baseline stability checks failed: {failed}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    repository_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or repository_root / "outputs" / config["experiment_name"]
    results = run(config, output_dir, repository_root)
    display_columns = [
        "model",
        "scaling",
        "score",
        "temperature",
        "accuracy",
        "ece",
        "coverage",
        "mean_size",
        "sscv",
        "class_coverage_gap",
    ]
    print(results[display_columns].round(4).to_string(index=False))
    print("Synthetic feature-role check: PASS")
    print(f"Saved Phase 1 outputs: {output_dir}")


if __name__ == "__main__":
    main()
