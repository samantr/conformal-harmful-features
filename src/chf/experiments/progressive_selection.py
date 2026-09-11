import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from chf.data import save_split_artifact
from chf.models import fit_classifier
from chf.selection import (
    HarmConstraints,
    ProgressiveCandidate,
    choose_progressive_step,
    choose_subset_size,
    make_tuning_evidence_folds,
    non_dominated_steps,
    rank_progressive_candidates,
)

from .protocol import (
    code_version,
    dataset_from_config,
    dataset_name,
    evaluate_logits,
    experiment_split,
    selection_data_id,
    selection_data_indices,
    split_id,
)
from .checkpoints import CheckpointStore, experiment_config_sha256


PIPELINE_COLUMNS = ["scaling", "score"]


@dataclass
class _SubsetEvidence:
    selected_indices: tuple[int, ...]
    raw: pd.DataFrame
    pipeline: pd.DataFrame
    consensus: dict[str, float | bool]


class _TuningSubsetEvaluator:
    """Fit subsets on outer train and assess them only on outer tune."""

    def __init__(
        self,
        *,
        features_train: np.ndarray,
        labels_train: np.ndarray,
        features_tune: np.ndarray,
        labels_tune: np.ndarray,
        model_config: Mapping[str, Any],
        config: Mapping[str, Any],
        evidence_folds: list[tuple[Any, ...]],
        resample_seeds: list[int],
        seed: int,
        constraints: HarmConstraints,
        selection_score: str,
        selection_scaling: str,
        checkpoint_store: CheckpointStore | None = None,
    ) -> None:
        self.features_train = features_train
        self.labels_train = labels_train
        self.features_tune = features_tune
        self.labels_tune = labels_tune
        self.model_config = model_config
        self.config = config
        self.evidence_folds = evidence_folds
        self.resample_seeds = resample_seeds
        self.seed = seed
        self.constraints = constraints
        self.selection_score = selection_score
        self.selection_scaling = selection_scaling
        self.checkpoint_store = checkpoint_store
        self.cache: dict[tuple[int, ...], _SubsetEvidence] = {}
        self.reference_resamples: pd.DataFrame | None = None

    def evaluate(self, selected_indices: tuple[int, ...]) -> _SubsetEvidence:
        key = tuple(sorted(selected_indices))
        if key in self.cache:
            return self.cache[key]
        if not key:
            raise ValueError("a classifier subset must contain at least one feature")

        checkpoint_key = {"selected_indices": list(key)}
        raw = (
            None
            if self.checkpoint_store is None
            else self.checkpoint_store.load_frame(checkpoint_key)
        )
        if raw is None:
            started = time.perf_counter()
            fitted = fit_classifier(
                self.features_train[:, key], self.labels_train, self.model_config,
                seed=self.seed,
            )
            fit_seconds = time.perf_counter() - started
            logits = fitted.logits(self.features_tune[:, key])
            rows: list[dict[str, Any]] = []
            for resample_index, (resample_seed, folds) in enumerate(
                zip(self.resample_seeds, self.evidence_folds, strict=True)
            ):
                for fold_index, fold in enumerate(folds):
                    fold_seed = resample_seed + 100 * fold_index
                    evaluated = evaluate_logits(
                        logits_tune=logits[fold.scale_tuning],
                        logits_calibration=logits[fold.calibration],
                        logits_test=logits[fold.evaluation],
                        labels_tune=self.labels_tune[fold.scale_tuning],
                        labels_calibration=self.labels_tune[fold.calibration],
                        labels_test=self.labels_tune[fold.evaluation],
                        config=self.config,
                        calibration_uniforms=np.random.default_rng(
                            fold_seed + 10_001
                        ).random(len(fold.calibration)),
                        test_uniforms=np.random.default_rng(
                            fold_seed + 20_001
                        ).random(len(fold.evaluation)),
                        seed=fold_seed,
                        included_scores=(self.selection_score,),
                        included_scalings=(self.selection_scaling,),
                    )
                    rows.extend(
                        {
                            "selection_resample": resample_index,
                            "selection_seed": resample_seed,
                            "selection_fold": fold_index,
                            "selected_indices": json.dumps(key),
                            "n_features": len(key),
                            "fit_seconds": fit_seconds,
                            **row,
                        }
                        for row in evaluated
                    )
            raw = pd.DataFrame(rows)
            if self.checkpoint_store is not None:
                self.checkpoint_store.save_frame(checkpoint_key, raw)
        resamples = (
            raw.groupby(["selection_resample", *PIPELINE_COLUMNS], sort=False)
            .agg(
                accuracy=("accuracy", "mean"),
                coverage=("coverage", "mean"),
                mean_size=("mean_size", "mean"),
                sscv=("sscv", "mean"),
                class_coverage_max_deviation=(
                    "class_coverage_max_deviation", "mean"
                ),
            )
            .reset_index()
        )
        resamples["conditional_violation"] = resamples[
            ["sscv", "class_coverage_max_deviation"]
        ].max(axis=1)

        if self.reference_resamples is None:
            self.reference_resamples = resamples.rename(
                columns={
                    "accuracy": "reference_accuracy",
                    "mean_size": "reference_mean_size",
                }
            )
        reference = self.reference_resamples[
            ["selection_resample", *PIPELINE_COLUMNS,
             "reference_accuracy", "reference_mean_size"]
        ]
        paired = resamples.merge(
            reference,
            on=["selection_resample", *PIPELINE_COLUMNS],
            how="left",
            validate="one_to_one",
        )
        paired["accuracy_loss"] = (
            paired["reference_accuracy"] - paired["accuracy"]
        )
        paired["efficiency_gain"] = (
            paired["reference_mean_size"] - paired["mean_size"]
        )
        target = 1.0 - float(self.config["conformal"]["alpha"])
        paired["coverage_shortfall"] = (target - paired["coverage"]).clip(lower=0)
        pipeline = (
            paired.groupby(PIPELINE_COLUMNS, sort=False)
            .agg(
                mean_accuracy=("accuracy", "mean"),
                mean_accuracy_loss=("accuracy_loss", "mean"),
                max_accuracy_loss=("accuracy_loss", "max"),
                mean_coverage=("coverage", "mean"),
                max_coverage_shortfall=("coverage_shortfall", "max"),
                mean_size=("mean_size", "mean"),
                mean_efficiency_gain=("efficiency_gain", "mean"),
                mean_conditional_violation=("conditional_violation", "mean"),
            )
            .reset_index()
        )
        pipeline["eligible"] = (
            (pipeline["max_accuracy_loss"] <= self.constraints.max_accuracy_loss)
            & (
                pipeline["max_coverage_shortfall"]
                <= self.constraints.max_coverage_shortfall
            )
        )
        consensus: dict[str, float | bool] = {
            "mean_accuracy": float(pipeline["mean_accuracy"].mean()),
            "mean_accuracy_loss": float(pipeline["mean_accuracy_loss"].mean()),
            "max_accuracy_loss": float(pipeline["max_accuracy_loss"].max()),
            "mean_coverage": float(pipeline["mean_coverage"].mean()),
            "max_coverage_shortfall": float(
                pipeline["max_coverage_shortfall"].max()
            ),
            "mean_size": float(pipeline["mean_size"].mean()),
            "mean_efficiency_gain": float(
                pipeline["mean_efficiency_gain"].mean()
            ),
            "mean_conditional_violation": float(
                pipeline["mean_conditional_violation"].mean()
            ),
            "eligible": bool(pipeline["eligible"].all()),
        }
        result = _SubsetEvidence(key, raw, pipeline, consensus)
        self.cache[key] = result
        return result


def run_progressive_selection(
    config: Mapping[str, Any], output_dir: Path, repository_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare one-shot and recursive removal, then perform one final test."""
    seed = int(config["seed"])
    dataset = dataset_from_config(config, repository_root)
    split = experiment_split(config, dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    outer_split_id = split_id(split)
    selection_train, selection_tune = selection_data_indices(
        config, split, dataset.labels, seed=seed
    )
    selection_identifier = selection_data_id(selection_train, selection_tune)
    save_split_artifact(
        output_dir / "split_indices.npz", split, seed=seed,
        metadata={"experiment_name": config["experiment_name"],
                  "phase": int(config.get("phase", 4)),
                  "split_id": outer_split_id,
                  "selection_data_id": selection_identifier},
    )
    selection_config = config.get("progressive_selection", {})
    harm_config = config.get("harm", {})
    constraints_values = harm_config.get("constraints", {})
    constraints = HarmConstraints(
        max_accuracy_loss=float(constraints_values.get("max_accuracy_loss", 0.01)),
        max_coverage_shortfall=float(
            constraints_values.get("max_coverage_shortfall", 0.03)
        ),
    )
    resample_count = int(selection_config.get(
        "selection_resamples", harm_config.get("selection_resamples", 5)
    ))
    crossfit_folds = int(selection_config.get(
        "crossfit_folds", harm_config.get("crossfit_folds", 4)
    ))
    scale_share = float(harm_config.get("scale_share_of_remainder", 0.6))
    resample_seeds = [seed + 300_000 + 997 * index for index in range(resample_count)]
    evidence_folds = [
        make_tuning_evidence_folds(
            dataset.labels[selection_tune], n_folds=crossfit_folds,
            scale_share_of_remainder=scale_share, seed=value,
        )
        for value in resample_seeds
    ]
    max_removals = min(
        int(selection_config.get("max_removals", len(dataset.feature_names) - 1)),
        len(dataset.feature_names) - 1,
    )
    require_positive = bool(selection_config.get("require_positive_incremental_gain", True))
    continue_after_stop = bool(
        selection_config.get("continue_path_after_stop", False)
    )
    selection_score = str(selection_config.get("score", "aps"))
    selection_scaling = str(selection_config.get("scaling", "base"))
    all_indices = tuple(range(len(dataset.feature_names)))
    path_frames: list[pd.DataFrame] = []
    consensus_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    evaluators: dict[str, _TuningSubsetEvaluator] = {}

    for model_name, model_config in config["models"].items():
        checkpoint_store = _progressive_checkpoint_store(
            config=config,
            output_dir=output_dir,
            model_name=model_name,
            model_config=model_config,
            seed=seed,
            split_identifier=outer_split_id,
            selection_identifier=selection_identifier,
            repository_root=repository_root,
        )
        evaluator = _TuningSubsetEvaluator(
            features_train=dataset.features[selection_train],
            labels_train=dataset.labels[selection_train],
            features_tune=dataset.features[selection_tune],
            labels_tune=dataset.labels[selection_tune],
            model_config=model_config,
            config=config,
            evidence_folds=evidence_folds,
            resample_seeds=resample_seeds,
            seed=seed,
            constraints=constraints,
            selection_score=selection_score,
            selection_scaling=selection_scaling,
            checkpoint_store=checkpoint_store,
        )
        evaluators[model_name] = evaluator
        reference = evaluator.evaluate(all_indices)
        initial_candidates = [
            _candidate_record(
                evaluator, all_indices, index, dataset.feature_names[index],
                current_gain=0.0,
            )
            for index in all_indices
        ]
        one_shot_order = rank_progressive_candidates(initial_candidates)
        paths = {
            "one_shot": _run_one_shot_path(
                evaluator, all_indices, one_shot_order, dataset.feature_names,
                max_removals=max_removals,
                continue_after_stop=continue_after_stop,
            ),
            "recursive": _run_recursive_path(
                evaluator, all_indices, dataset.feature_names,
                max_removals=max_removals, require_positive=require_positive,
                continue_after_stop=continue_after_stop,
            ),
        }
        for method, path in paths.items():
            selected_step = choose_subset_size(path)
            for row in path:
                evidence = evaluator.evaluate(tuple(row["selected_indices"]))
                pipeline = evidence.pipeline.copy()
                pipeline.insert(0, "model", model_name)
                pipeline.insert(1, "method", method)
                pipeline.insert(2, "step", row["step"])
                pipeline["selection_data_id"] = selection_identifier
                pipeline["n_removed"] = row["n_removed"]
                pipeline["removed_feature"] = row["removed_feature"]
                pipeline["selected_indices"] = json.dumps(row["selected_indices"])
                pipeline["selected_for_final"] = row["step"] == selected_step
                path_frames.append(pipeline)
                consensus_rows.append({
                    "model": model_name,
                    "method": method,
                    "selection_data_id": selection_identifier,
                    **row,
                    "selected_indices": json.dumps(row["selected_indices"]),
                    "selected_features": json.dumps(
                        [dataset.feature_names[i] for i in row["selected_indices"]]
                    ),
                    "selected_for_final": row["step"] == selected_step,
                })
            chosen = next(row for row in path if row["step"] == selected_step)
            selected_rows.append({"model": model_name, "method": method, **chosen})
        print(
            f"Progressive tuning complete: {model_name} | "
            f"{len(evaluator.cache)} unique subsets", flush=True,
        )

    paths = pd.concat(path_frames, ignore_index=True)
    consensus_paths = pd.DataFrame(consensus_rows)
    _mark_frontiers(paths)
    _mark_frontiers(consensus_paths)
    selection_only = bool(
        config.get("progressive_selection", {}).get("selection_only", False)
    )
    if selection_only:
        final = pd.DataFrame()
    else:
        final = _final_evaluation(
            config=config, dataset=dataset, split=split, selected_rows=selected_rows,
            model_configs=config["models"], seed=seed, split_identifier=outer_split_id,
            version=code_version(repository_root),
            selection_identifier=selection_identifier,
        )
    _write_outputs(
        paths, consensus_paths, final, output_dir, config, constraints,
        resample_count, crossfit_folds, max_removals,
        selection_score, selection_scaling,
        selection_identifier, len(selection_train), len(selection_tune),
    )
    return paths, consensus_paths, final


def _candidate_record(
    evaluator: _TuningSubsetEvaluator,
    selected: tuple[int, ...],
    removed_index: int,
    removed_name: str,
    *,
    current_gain: float,
) -> ProgressiveCandidate:
    candidate_subset = tuple(index for index in selected if index != removed_index)
    evidence = evaluator.evaluate(candidate_subset)
    summary = evidence.consensus
    cumulative_gain = float(summary["mean_efficiency_gain"])
    return ProgressiveCandidate(
        feature_index=removed_index,
        feature_name=removed_name,
        cumulative_efficiency_gain=cumulative_gain,
        incremental_efficiency_gain=cumulative_gain - current_gain,
        max_accuracy_loss=float(summary["max_accuracy_loss"]),
        max_coverage_shortfall=float(summary["max_coverage_shortfall"]),
        mean_conditional_violation=float(summary["mean_conditional_violation"]),
        eligible=bool(summary["eligible"]),
    )


def _path_row(
    *, step: int, selected: tuple[int, ...], removed_feature: str | None,
    evidence: _SubsetEvidence,
) -> dict[str, Any]:
    summary = evidence.consensus
    return {
        "step": step,
        "n_removed": step,
        "n_features": len(selected),
        "removed_feature": removed_feature,
        "selected_indices": list(selected),
        "eligible": bool(summary["eligible"]),
        "mean_accuracy": float(summary["mean_accuracy"]),
        "mean_accuracy_loss": float(summary["mean_accuracy_loss"]),
        "max_accuracy_loss": float(summary["max_accuracy_loss"]),
        "mean_coverage": float(summary["mean_coverage"]),
        "max_coverage_shortfall": float(summary["max_coverage_shortfall"]),
        "mean_size": float(summary["mean_size"]),
        "cumulative_efficiency_gain": float(summary["mean_efficiency_gain"]),
        "mean_conditional_violation": float(summary["mean_conditional_violation"]),
    }


def _run_one_shot_path(
    evaluator: _TuningSubsetEvaluator,
    all_indices: tuple[int, ...],
    order: list[ProgressiveCandidate],
    feature_names: tuple[str, ...],
    *, max_removals: int, continue_after_stop: bool = False,
) -> list[dict[str, Any]]:
    selected = all_indices
    path = [_path_row(step=0, selected=selected, removed_feature=None,
                      evidence=evaluator.evaluate(selected))]
    for candidate in order[:max_removals]:
        proposed = tuple(i for i in selected if i != candidate.feature_index)
        evidence = evaluator.evaluate(proposed)
        if not bool(evidence.consensus["eligible"]) and not continue_after_stop:
            break
        selected = proposed
        path.append(_path_row(
            step=len(path), selected=selected,
            removed_feature=feature_names[candidate.feature_index], evidence=evidence,
        ))
    return path


def _run_recursive_path(
    evaluator: _TuningSubsetEvaluator,
    all_indices: tuple[int, ...],
    feature_names: tuple[str, ...],
    *, max_removals: int, require_positive: bool,
    continue_after_stop: bool = False,
) -> list[dict[str, Any]]:
    selected = all_indices
    reference = evaluator.evaluate(selected)
    path = [_path_row(step=0, selected=selected, removed_feature=None,
                      evidence=reference)]
    for _ in range(max_removals):
        current_gain = float(evaluator.evaluate(selected).consensus["mean_efficiency_gain"])
        candidates = [
            _candidate_record(
                evaluator, selected, index, feature_names[index],
                current_gain=current_gain,
            )
            for index in selected
        ]
        chosen = choose_progressive_step(
            candidates, require_positive_gain=require_positive
        )
        if chosen is None:
            if not continue_after_stop:
                break
            ordered = rank_progressive_candidates(candidates)
            if not ordered:
                break
            chosen = ordered[0]
        selected = tuple(i for i in selected if i != chosen.feature_index)
        path.append(_path_row(
            step=len(path), selected=selected, removed_feature=chosen.feature_name,
            evidence=evaluator.evaluate(selected),
        ))
    return path


def _progressive_checkpoint_store(
    *,
    config: Mapping[str, Any],
    output_dir: Path,
    model_name: str,
    model_config: Mapping[str, Any],
    seed: int,
    split_identifier: str,
    selection_identifier: str,
    repository_root: Path,
) -> CheckpointStore | None:
    checkpoint_config = config.get("checkpointing", {})
    if not bool(checkpoint_config.get("enabled", False)):
        return None
    manifest = {
        "schema_version": 1,
        "stage": "progressive_selection",
        "experiment_name": config["experiment_name"],
        "seed": seed,
        "model": model_name,
        "model_config": dict(model_config),
        "split_id": split_identifier,
        "selection_data_id": selection_identifier,
        "config_sha256": experiment_config_sha256(config),
        "code_version": code_version(repository_root),
    }
    store = CheckpointStore(
        output_dir / "checkpoints" / "progressive" / model_name,
        manifest,
    )
    store.initialize(resume=bool(checkpoint_config.get("resume", False)))
    return store


def _mark_frontiers(frame: pd.DataFrame) -> None:
    frame["pareto_frontier"] = False
    group_columns = ["model", "method"]
    if "scaling" in frame.columns:
        group_columns += PIPELINE_COLUMNS
    for _, group in frame.groupby(group_columns, sort=False):
        frame.loc[group.index, "pareto_frontier"] = non_dominated_steps(
            group["mean_accuracy_loss"].to_numpy(),
            group["mean_size"].to_numpy(),
            group["mean_conditional_violation"].to_numpy(),
        )


def _final_evaluation(
    *, config: Mapping[str, Any], dataset: Any, split: Any,
    selected_rows: list[dict[str, Any]], model_configs: Mapping[str, Any],
    seed: int, split_identifier: str, version: str,
    selection_identifier: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_indices = tuple(range(len(dataset.feature_names)))
    selections: list[tuple[str, str, tuple[int, ...]]] = [
        (model_name, "reference", all_indices) for model_name in model_configs
    ]
    selections += [
        (str(row["model"]), str(row["method"]), tuple(row["selected_indices"]))
        for row in selected_rows
    ]
    calibration_uniforms = np.random.default_rng(seed + 400_001).random(
        len(split.calibration)
    )
    test_uniforms = np.random.default_rng(seed + 400_002).random(len(split.test))
    for model_name, method, selected in selections:
        fitted = fit_classifier(
            dataset.features[split.train][:, selected], dataset.labels[split.train],
            model_configs[model_name], seed=seed,
        )
        evaluated = evaluate_logits(
            logits_tune=fitted.logits(dataset.features[split.tune][:, selected]),
            logits_calibration=fitted.logits(
                dataset.features[split.calibration][:, selected]
            ),
            logits_test=fitted.logits(dataset.features[split.test][:, selected]),
            labels_tune=dataset.labels[split.tune],
            labels_calibration=dataset.labels[split.calibration],
            labels_test=dataset.labels[split.test],
            config=config,
            calibration_uniforms=calibration_uniforms,
            test_uniforms=test_uniforms,
            seed=seed + 400_000,
        )
        metadata = {
            "experiment": config["experiment_name"], "dataset": dataset_name(dataset),
            "model": model_name, "method": method, "seed": seed,
            "split_id": split_identifier, "selected_indices": json.dumps(selected),
            "selection_data_id": selection_identifier,
            "selected_features": json.dumps([dataset.feature_names[i] for i in selected]),
            "n_features": len(selected), "classifier_refit": True,
            "subset_frozen_before_final_calibration": True,
            "code_version": version,
        }
        rows.extend({**metadata, **row} for row in evaluated)
    result = pd.DataFrame(rows)
    reference = result.loc[result["method"] == "reference", [
        "model", *PIPELINE_COLUMNS, "accuracy", "coverage", "mean_size",
        "sscv", "class_coverage_max_deviation",
    ]].rename(columns={
        "accuracy": "reference_accuracy", "coverage": "reference_coverage",
        "mean_size": "reference_mean_size", "sscv": "reference_sscv",
        "class_coverage_max_deviation": "reference_class_coverage_max_deviation",
    })
    result = result.merge(
        reference, on=["model", *PIPELINE_COLUMNS], how="left", validate="many_to_one"
    )
    result["accuracy_loss"] = result["reference_accuracy"] - result["accuracy"]
    result["mean_size_reduction"] = result["reference_mean_size"] - result["mean_size"]
    result["coverage_deviation"] = (
        result["coverage"] - (1 - result["alpha"])
    ).abs()
    result["conditional_violation"] = result[
        ["sscv", "class_coverage_max_deviation"]
    ].max(axis=1)
    return result


def _write_outputs(
    paths: pd.DataFrame, consensus_paths: pd.DataFrame, final: pd.DataFrame,
    output_dir: Path, config: Mapping[str, Any], constraints: HarmConstraints,
    resample_count: int, crossfit_folds: int, max_removals: int,
    selection_score: str, selection_scaling: str,
    selection_identifier: str, selection_train_size: int,
    selection_tune_size: int,
) -> None:
    paths.to_csv(output_dir / "progressive_pipeline_paths.csv", index=False)
    consensus_paths.to_csv(output_dir / "progressive_consensus_paths.csv", index=False)
    final.to_csv(output_dir / "progressive_final_results.csv", index=False)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
    )
    _plot_paths(paths, output_dir / "progressive_pareto_curves.png")
    selection_only = bool(
        config.get("progressive_selection", {}).get("selection_only", False)
    )
    expected_final = (
        0
        if selection_only
        else len(config["models"]) * 3 * len(config["conformal"]["scores"]) * 3
    )
    checks = {
        "both_selection_methods_present": set(consensus_paths["method"]) == {
            "one_shot", "recursive"
        },
        "one_frozen_subset_per_method_and_model": bool(
            (consensus_paths.groupby(["model", "method"])["selected_for_final"].sum() == 1).all()
        ),
        "selected_steps_are_tuning_eligible": bool(
            consensus_paths.loc[consensus_paths["selected_for_final"], "eligible"].all()
        ),
        "final_row_count": len(final) == expected_final,
        "fresh_final_thresholds_are_finite": bool(
            selection_only or np.isfinite(final["threshold"]).all()
        ),
        "final_values_are_finite": bool(
            selection_only
            or np.isfinite(final[
                ["accuracy", "coverage", "mean_size", "sscv", "conditional_violation"]
            ]).all().all()
        ),
        "subset_frozen_before_final_calibration": bool(
            selection_only
            or final["subset_frozen_before_final_calibration"].all()
        ),
    }
    record = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "protocol": {
            "classifier_fit_partition": "outer train",
            "ranking_and_subset_size_partition": "outer tune only",
            "selection_data_id": selection_identifier,
            "selection_train_rows": selection_train_size,
            "selection_tune_rows": selection_tune_size,
            "selection_evidence": f"{resample_count}x{crossfit_folds} cross-fitting",
            "selection_pipeline": f"{selection_scaling}+{selection_score}",
            "final_calibration_access": (
                "none in selection-only mode"
                if selection_only
                else "once after subset freeze"
            ),
            "final_test_access": (
                "none in selection-only mode"
                if selection_only
                else "once after subset freeze"
            ),
            "one_shot": "fixed initial constrained-efficiency order",
            "recursive": "retrain and rerank remaining features after every removal",
        },
        "constraints": {
            "max_accuracy_loss": constraints.max_accuracy_loss,
            "max_coverage_shortfall": constraints.max_coverage_shortfall,
            "max_removals": max_removals,
        },
        "observed_final_rows": len(final), "expected_final_rows": expected_final,
    }
    (output_dir / "progressive_protocol.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    if record["status"] != "PASS":
        raise RuntimeError(
            f"Phase 4 output validation failed: {[k for k, v in checks.items() if not v]}"
        )


def _plot_paths(paths: pd.DataFrame, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subset = paths.loc[(paths["scaling"] == "base") & (paths["score"] == "aps")]
    models = list(subset["model"].unique())
    figure, axes = plt.subplots(2, len(models), figsize=(6 * len(models), 9), squeeze=False)
    for column, model in enumerate(models):
        model_rows = subset.loc[subset["model"] == model]
        for method, group in model_rows.groupby("method", sort=False):
            ordered = group.sort_values("step")
            axes[0, column].plot(ordered["mean_accuracy"], ordered["mean_size"], "o-", label=method)
            axes[1, column].plot(ordered["mean_size"], ordered["mean_conditional_violation"], "o-", label=method)
        axes[0, column].set(title=model, xlabel="Tuning accuracy", ylabel="Mean APS size")
        axes[1, column].set(xlabel="Mean APS size", ylabel="Conditional violation")
        axes[0, column].legend()
        axes[1, column].legend()
    figure.suptitle("Progressive removal paths (tuning-only Base APS evidence)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
