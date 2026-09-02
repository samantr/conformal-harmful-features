import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from chf.data import make_four_way_split, save_split_artifact
from chf.interventions import intervene_feature
from chf.models import FittedClassifier, fit_classifier

from .protocol import (
    REFERENCE_METRICS,
    attach_reference_deltas,
    code_version,
    dataset_from_config,
    evaluate_logits,
    split_id,
)


def run_retrain_ablation(
    config: Mapping[str, Any], output_dir: Path, repository_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove each feature, refit the model, retune scaling and recalibrate."""
    context = _ExperimentContext(config, output_dir, repository_root)
    rows: list[dict[str, Any]] = []
    for model_name, model_config in config["models"].items():
        rows.extend(
            context.fit_and_evaluate(
                model_name=model_name,
                model_config=model_config,
                selected_indices=np.arange(context.n_features),
                feature_index=None,
                intervention="reference",
            )
        )
        for feature in context.dataset.feature_manifest:
            selected_indices = np.delete(np.arange(context.n_features), feature.index)
            rows.extend(
                context.fit_and_evaluate(
                    model_name=model_name,
                    model_config=model_config,
                    selected_indices=selected_indices,
                    feature_index=feature.index,
                    intervention="retrain_ablation",
                )
            )
            print(
                f"Retrain ablation: {model_name} removed {feature.name} "
                f"({feature.index + 1}/{context.n_features})",
                flush=True,
            )

    results = attach_reference_deltas(pd.DataFrame(rows))
    summary = _summarize(results, config)
    _write_outputs(
        results,
        summary,
        output_dir,
        result_name="single_feature_ablation.csv",
        summary_name="ablation_summary.csv",
        plot_name="ablation_accuracy_vs_aps_size.png",
        protocol_name="ablation_protocol.json",
        expected_rows=(1 + context.n_features)
        * len(config["models"])
        * len(config["conformal"]["scores"])
        * 3,
        config=config,
        protocol={
            "intervention": "retrain_ablation",
            "classifier_refit": True,
            "scaling_retuned_on": "tune",
            "threshold_recalibrated_on": "calibration",
            "evaluation_partition": "test",
        },
    )
    return results, summary


def run_masking_sensitivity(
    config: Mapping[str, Any], output_dir: Path, repository_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate mean masking and permutation while keeping the model fixed."""
    context = _ExperimentContext(config, output_dir, repository_root)
    rows: list[dict[str, Any]] = []
    for model_name, model_config in config["models"].items():
        started = time.perf_counter()
        fitted = fit_classifier(
            context.train_features,
            context.labels_train,
            model_config,
            seed=context.seed,
        )
        fit_seconds = time.perf_counter() - started
        rows.extend(
            context.evaluate_fitted(
                fitted=fitted,
                model_name=model_name,
                tune_features=context.tune_features,
                calibration_features=context.calibration_features,
                test_features=context.test_features,
                selected_indices=np.arange(context.n_features),
                feature_index=None,
                intervention="reference",
                fit_seconds=fit_seconds,
            )
        )

        for feature in context.dataset.feature_manifest:
            training_mean = float(context.train_features[:, feature.index].mean())
            for method in ("mean_mask", "permutation"):
                if method == "mean_mask":
                    kwargs_by_partition = {
                        name: {"training_mean": training_mean}
                        for name in ("tune", "calibration", "test")
                    }
                else:
                    base_seed = context.seed + 100_000 + feature.index * 10
                    kwargs_by_partition = {
                        "tune": {"rng": np.random.default_rng(base_seed + 1)},
                        "calibration": {
                            "rng": np.random.default_rng(base_seed + 2)
                        },
                        "test": {"rng": np.random.default_rng(base_seed + 3)},
                    }
                tune_features = intervene_feature(
                    context.tune_features,
                    feature.index,
                    method,
                    **kwargs_by_partition["tune"],
                )
                calibration_features = intervene_feature(
                    context.calibration_features,
                    feature.index,
                    method,
                    **kwargs_by_partition["calibration"],
                )
                test_features = intervene_feature(
                    context.test_features,
                    feature.index,
                    method,
                    **kwargs_by_partition["test"],
                )
                rows.extend(
                    context.evaluate_fitted(
                        fitted=fitted,
                        model_name=model_name,
                        tune_features=tune_features,
                        calibration_features=calibration_features,
                        test_features=test_features,
                        selected_indices=np.arange(context.n_features),
                        feature_index=feature.index,
                        intervention=method,
                        fit_seconds=fit_seconds,
                    )
                )
            print(
                f"Fixed-model sensitivity: {model_name} intervened on {feature.name} "
                f"({feature.index + 1}/{context.n_features})",
                flush=True,
            )

    results = attach_reference_deltas(pd.DataFrame(rows))
    summary = _summarize(results, config)
    _write_outputs(
        results,
        summary,
        output_dir,
        result_name="masking_sensitivity.csv",
        summary_name="masking_summary.csv",
        plot_name="masking_accuracy_vs_aps_size.png",
        protocol_name="masking_protocol.json",
        expected_rows=(1 + 2 * context.n_features)
        * len(config["models"])
        * len(config["conformal"]["scores"])
        * 3,
        config=config,
        protocol={
            "interventions": ["mean_mask", "permutation"],
            "classifier_refit": False,
            "mean_mask_value": "training-partition feature mean",
            "permutation_scope": "independently within each partition",
            "scaling_retuned_on": "intervened tune",
            "threshold_recalibrated_on": "intervened calibration",
            "evaluation_partition": "intervened test",
        },
    )
    return results, summary


class _ExperimentContext:
    def __init__(
        self,
        config: Mapping[str, Any],
        output_dir: Path,
        repository_root: Path,
    ) -> None:
        self.config = config
        self.seed = int(config["seed"])
        self.dataset = dataset_from_config(config)
        self.n_features = self.dataset.features.shape[1]
        split_config = config["split"]
        sizes = tuple(
            int(split_config[name])
            for name in ("train", "tune", "calibration", "test")
        )
        self.split = make_four_way_split(self.dataset.labels, sizes, self.seed)
        self.split_identifier = split_id(self.split)
        self.version = code_version(repository_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_split_artifact(
            output_dir / "split_indices.npz",
            self.split,
            seed=self.seed,
            metadata={
                "experiment_name": config["experiment_name"],
                "n_samples": len(self.dataset.labels),
                "split_id": self.split_identifier,
            },
        )
        self.train_features = self.dataset.features[self.split.train]
        self.tune_features = self.dataset.features[self.split.tune]
        self.calibration_features = self.dataset.features[self.split.calibration]
        self.test_features = self.dataset.features[self.split.test]
        self.labels_train = self.dataset.labels[self.split.train]
        self.labels_tune = self.dataset.labels[self.split.tune]
        self.labels_calibration = self.dataset.labels[self.split.calibration]
        self.labels_test = self.dataset.labels[self.split.test]
        self.calibration_uniforms = np.random.default_rng(
            self.seed + 10_001
        ).random(len(self.split.calibration))
        self.test_uniforms = np.random.default_rng(self.seed + 20_001).random(
            len(self.split.test)
        )

    def fit_and_evaluate(
        self,
        *,
        model_name: str,
        model_config: Mapping[str, Any],
        selected_indices: np.ndarray,
        feature_index: int | None,
        intervention: str,
    ) -> list[dict[str, Any]]:
        started = time.perf_counter()
        fitted = fit_classifier(
            self.train_features[:, selected_indices],
            self.labels_train,
            model_config,
            seed=self.seed,
        )
        fit_seconds = time.perf_counter() - started
        return self.evaluate_fitted(
            fitted=fitted,
            model_name=model_name,
            tune_features=self.tune_features[:, selected_indices],
            calibration_features=self.calibration_features[:, selected_indices],
            test_features=self.test_features[:, selected_indices],
            selected_indices=selected_indices,
            feature_index=feature_index,
            intervention=intervention,
            fit_seconds=fit_seconds,
        )

    def evaluate_fitted(
        self,
        *,
        fitted: FittedClassifier,
        model_name: str,
        tune_features: np.ndarray,
        calibration_features: np.ndarray,
        test_features: np.ndarray,
        selected_indices: np.ndarray,
        feature_index: int | None,
        intervention: str,
        fit_seconds: float,
    ) -> list[dict[str, Any]]:
        selected_names = [
            self.dataset.feature_names[index] for index in selected_indices
        ]
        if feature_index is None:
            feature_name = None
            feature_role = None
            source_feature = None
        else:
            feature = self.dataset.feature_manifest[feature_index]
            feature_name = feature.name
            feature_role = feature.role
            source_feature = feature.source_feature
        evaluated = evaluate_logits(
            logits_tune=fitted.logits(tune_features),
            logits_calibration=fitted.logits(calibration_features),
            logits_test=fitted.logits(test_features),
            labels_tune=self.labels_tune,
            labels_calibration=self.labels_calibration,
            labels_test=self.labels_test,
            config=self.config,
            calibration_uniforms=self.calibration_uniforms,
            test_uniforms=self.test_uniforms,
            seed=self.seed,
        )
        metadata = {
            "experiment": self.config["experiment_name"],
            "dataset": "controlled_multiclass",
            "model": model_name,
            "seed": self.seed,
            "split_id": self.split_identifier,
            "intervention": intervention,
            "feature_index": feature_index,
            "feature_name": feature_name,
            "feature_role": feature_role,
            "source_feature": source_feature,
            "selected_features": json.dumps(selected_names),
            "n_features": len(selected_names),
            "classifier_refit": intervention == "retrain_ablation",
            "scaling_retuned": intervention != "reference",
            "threshold_recalibrated": intervention != "reference",
            "fit_seconds": fit_seconds,
            "code_version": self.version,
        }
        return [{**metadata, **row} for row in evaluated]


def _summarize(results: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    interventions = results.loc[results["intervention"] != "reference"].copy()
    model_rows: list[dict[str, Any]] = []
    group_columns = [
        "intervention",
        "feature_index",
        "feature_name",
        "feature_role",
        "source_feature",
        "model",
    ]
    for keys, group in interventions.groupby(group_columns, dropna=False, sort=False):
        base_aps = group.loc[(group["scaling"] == "base") & (group["score"] == "aps")]
        if len(base_aps) != 1:
            raise RuntimeError("each feature/model must have exactly one Base APS row")
        if (
            group["accuracy_loss"].nunique() != 1
            or group["macro_f1_loss"].nunique() != 1
        ):
            raise RuntimeError("positive temperature scaling changed a class prediction")
        model_rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "accuracy_loss": float(group["accuracy_loss"].iloc[0]),
                "macro_f1_loss": float(group["macro_f1_loss"].iloc[0]),
                "base_aps_size_reduction": float(base_aps["mean_size_reduction"].iloc[0]),
                "mean_size_reduction_all_methods": float(
                    group["mean_size_reduction"].mean()
                ),
                "max_abs_coverage_delta": float(group["coverage_delta"].abs().max()),
                "max_coverage_target_deviation": float(
                    group["coverage_target_deviation"].max()
                ),
                "mean_sscv_delta": float(group["sscv_delta"].mean()),
                "mean_class_coverage_gap_delta": float(
                    group["class_coverage_gap_delta"].mean()
                ),
            }
        )
    by_model = pd.DataFrame(model_rows)
    selection_config = config.get("selection", {})
    max_accuracy_loss = float(selection_config.get("max_accuracy_loss", 0.01))
    max_coverage_delta = float(selection_config.get("max_coverage_deviation", 0.01))
    summary_rows: list[dict[str, Any]] = []
    summary_groups = [
        "intervention",
        "feature_index",
        "feature_name",
        "feature_role",
        "source_feature",
    ]
    for keys, group in by_model.groupby(summary_groups, dropna=False, sort=False):
        descriptive_pattern = bool(
            group["accuracy_loss"].max() <= max_accuracy_loss
            and group["max_abs_coverage_delta"].max() <= max_coverage_delta
            and group["base_aps_size_reduction"].min() > 0
        )
        summary_rows.append(
            {
                **dict(zip(summary_groups, keys, strict=True)),
                "models_evaluated": int(len(group)),
                "mean_accuracy_loss": float(group["accuracy_loss"].mean()),
                "max_accuracy_loss": float(group["accuracy_loss"].max()),
                "mean_macro_f1_loss": float(group["macro_f1_loss"].mean()),
                "mean_base_aps_size_reduction": float(
                    group["base_aps_size_reduction"].mean()
                ),
                "min_base_aps_size_reduction": float(
                    group["base_aps_size_reduction"].min()
                ),
                "mean_size_reduction_all_methods": float(
                    group["mean_size_reduction_all_methods"].mean()
                ),
                "max_abs_coverage_delta": float(
                    group["max_abs_coverage_delta"].max()
                ),
                "max_coverage_target_deviation": float(
                    group["max_coverage_target_deviation"].max()
                ),
                "mean_sscv_delta": float(group["mean_sscv_delta"].mean()),
                "mean_class_coverage_gap_delta": float(
                    group["mean_class_coverage_gap_delta"].mean()
                ),
                "descriptive_pattern_pass": descriptive_pattern,
            }
        )
    return pd.DataFrame(summary_rows).sort_values(
        ["descriptive_pattern_pass", "mean_base_aps_size_reduction"],
        ascending=[False, False],
        ignore_index=True,
    )


def _write_outputs(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    *,
    result_name: str,
    summary_name: str,
    plot_name: str,
    protocol_name: str,
    expected_rows: int,
    config: Mapping[str, Any],
    protocol: dict[str, Any],
) -> None:
    results.to_csv(output_dir / result_name, index=False)
    summary.to_csv(output_dir / summary_name, index=False)
    _plot_base_aps(results, output_dir / plot_name)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
    )
    core_metrics = [
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
    numerical_columns = [
        "calibration_zero_count",
        "calibration_exactly_one_count",
        "test_zero_count",
        "test_exactly_one_count",
    ]
    coverage_tolerance = float(
        config.get("stability", {}).get("max_coverage_deviation", 0.03)
    )
    checks = {
        "expected_result_rows": len(results) == expected_rows,
        "unique_result_keys": not results.duplicated(
            ["model", "intervention", "feature_name", "scaling", "score"]
        ).any(),
        "finite_core_metrics": bool(
            np.isfinite(results[core_metrics].to_numpy()).all()
        ),
        "coverage_within_stability_tolerance": bool(
            (results["coverage_target_deviation"] <= coverage_tolerance + 1e-12).all()
        ),
        "no_zero_or_saturated_probabilities": bool(
            (results[numerical_columns] == 0).all().all()
        ),
        "all_interventions_have_reference": bool(
            results[[f"reference_{metric}" for metric in REFERENCE_METRICS]]
            .notna()
            .all()
            .all()
        ),
    }
    record = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "expected_rows": expected_rows,
        "observed_rows": len(results),
        "coverage_target": 1.0 - float(config["conformal"]["alpha"]),
        "coverage_stability_tolerance": coverage_tolerance,
        "descriptive_test_pattern": {
            "purpose": "Phase 2 phenomenon check; not a ranking or selection input",
            "selection_use_permitted": False,
            "max_accuracy_loss_in_every_model": float(
                config.get("selection", {}).get("max_accuracy_loss", 0.01)
            ),
            "max_absolute_coverage_delta": float(
                config.get("selection", {}).get("max_coverage_deviation", 0.01)
            ),
            "positive_Base_APS_size_reduction_in_every_model": True,
            "matching_interventions": [
                f"{row.intervention}:{row.feature_name}"
                for row in summary.loc[summary["descriptive_pattern_pass"]].itertuples()
            ],
        },
        "protocol": protocol,
    }
    (output_dir / protocol_name).write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    if record["status"] != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Phase 2 output validation failed: {failed}")


def _plot_base_aps(results: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_data = results.loc[
        (results["intervention"] != "reference")
        & (results["scaling"] == "base")
        & (results["score"] == "aps")
    ].copy()
    models = list(plot_data["model"].unique())
    interventions = list(plot_data["intervention"].unique())
    figure, axes = plt.subplots(
        len(interventions),
        len(models),
        figsize=(6 * len(models), 4.5 * len(interventions)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    colors = {
        "strong": "#d62728",
        "weak": "#ff7f0e",
        "redundant": "#1f77b4",
        "noise": "#7f7f7f",
    }
    for row_index, intervention in enumerate(interventions):
        for column_index, model in enumerate(models):
            axis = axes[row_index, column_index]
            subset = plot_data.loc[
                (plot_data["intervention"] == intervention)
                & (plot_data["model"] == model)
            ]
            for role, role_data in subset.groupby("feature_role"):
                axis.scatter(
                    role_data["accuracy_loss"],
                    role_data["mean_size_reduction"],
                    label=role,
                    color=colors.get(role, "black"),
                    alpha=0.8,
                )
            axis.axhline(0, color="black", linewidth=0.8)
            axis.axvline(0.01, color="black", linewidth=0.8, linestyle="--")
            axis.set_title(f"{intervention} | {model}")
            axis.set_xlabel("Accuracy loss")
            axis.set_ylabel("Base APS mean-size reduction")
            axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
