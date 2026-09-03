import json
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from chf.data import make_four_way_split, save_split_artifact
from chf.models import fit_classifier
from chf.selection import (
    HarmConstraints,
    HarmWeights,
    make_tuning_evidence_folds,
    pareto_fronts,
)

from .protocol import (
    REFERENCE_METRICS,
    attach_reference_deltas,
    code_version,
    dataset_from_config,
    dataset_name,
    evaluate_logits,
    split_id,
)


RANK_COLUMNS = {
    "constrained": "constrained_rank",
    "weighted": "weighted_rank",
    "pareto": "pareto_rank",
}


def run_harm_ranking(
    config: Mapping[str, Any], output_dir: Path, repository_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare conformal-harm definitions using outer tuning data only."""
    seed = int(config["seed"])
    dataset = dataset_from_config(config, repository_root)
    split_config = config["split"]
    split = make_four_way_split(
        dataset.labels,
        tuple(
            int(split_config[name])
            for name in ("train", "tune", "calibration", "test")
        ),
        seed,
    )
    outer_split_id = split_id(split)
    version = code_version(repository_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_split_artifact(
        output_dir / "split_indices.npz",
        split,
        seed=seed,
        metadata={
            "experiment_name": config["experiment_name"],
            "phase": 3,
            "split_id": outer_split_id,
        },
    )

    features_train = dataset.features[split.train]
    labels_train = dataset.labels[split.train]
    features_tune = dataset.features[split.tune]
    labels_tune = dataset.labels[split.tune]
    harm_config = config.get("harm", {})
    resample_count = int(harm_config.get("selection_resamples", 5))
    if resample_count < 2:
        raise ValueError("harm.selection_resamples must be at least two")
    crossfit_folds = int(harm_config.get("crossfit_folds", 4))
    scale_share = float(harm_config.get("scale_share_of_remainder", 0.6))
    resample_seeds = [seed + 200_000 + 997 * index for index in range(resample_count)]
    evidence_splits = [
        make_tuning_evidence_folds(
            labels_tune,
            n_folds=crossfit_folds,
            scale_share_of_remainder=scale_share,
            seed=value,
        )
        for value in resample_seeds
    ]
    constraints = _constraints_from_config(harm_config)
    weights = _weights_from_config(harm_config)

    rows: list[dict[str, Any]] = []
    all_indices = np.arange(features_train.shape[1])
    for model_name, model_config in config["models"].items():
        candidate_indices = [(None, all_indices)] + [
            (feature.index, np.delete(all_indices, feature.index))
            for feature in dataset.feature_manifest
        ]
        for feature_index, selected_indices in candidate_indices:
            started = time.perf_counter()
            fitted = fit_classifier(
                features_train[:, selected_indices],
                labels_train,
                model_config,
                seed=seed,
            )
            fit_seconds = time.perf_counter() - started
            tune_logits = fitted.logits(features_tune[:, selected_indices])
            if feature_index is None:
                feature_name = None
                feature_role = None
                source_feature = None
                intervention = "reference"
            else:
                feature = dataset.feature_manifest[feature_index]
                feature_name = feature.name
                feature_role = feature.role
                source_feature = feature.source_feature
                intervention = "retrain_ablation"

            for resample_index, (resample_seed, resample_folds) in enumerate(
                zip(resample_seeds, evidence_splits, strict=True)
            ):
                for fold_index, evidence_split in enumerate(resample_folds):
                    fold_seed = resample_seed + 100 * fold_index
                    calibration_uniforms = np.random.default_rng(
                        fold_seed + 10_001
                    ).random(len(evidence_split.calibration))
                    evaluation_uniforms = np.random.default_rng(
                        fold_seed + 20_001
                    ).random(len(evidence_split.evaluation))
                    evaluated = evaluate_logits(
                        logits_tune=tune_logits[evidence_split.scale_tuning],
                        logits_calibration=tune_logits[evidence_split.calibration],
                        logits_test=tune_logits[evidence_split.evaluation],
                        labels_tune=labels_tune[evidence_split.scale_tuning],
                        labels_calibration=labels_tune[evidence_split.calibration],
                        labels_test=labels_tune[evidence_split.evaluation],
                        config=config,
                        calibration_uniforms=calibration_uniforms,
                        test_uniforms=evaluation_uniforms,
                        seed=fold_seed,
                    )
                    selected_names = [
                        dataset.feature_names[index] for index in selected_indices
                    ]
                    metadata = {
                        "experiment": config["experiment_name"],
                        "dataset": dataset_name(dataset),
                        "model": model_name,
                        "seed": seed,
                        "split_id": outer_split_id,
                        "evidence_source": "outer_tune_only",
                        "selection_resample": resample_index,
                        "selection_seed": resample_seed,
                        "selection_fold": fold_index,
                        "scale_tuning_size": len(evidence_split.scale_tuning),
                        "selection_calibration_size": len(
                            evidence_split.calibration
                        ),
                        "selection_evaluation_size": len(evidence_split.evaluation),
                        "intervention": intervention,
                        "feature_index": feature_index,
                        "feature_name": feature_name,
                        "feature_role": feature_role,
                        "source_feature": source_feature,
                        "selected_features": json.dumps(selected_names),
                        "n_features": len(selected_names),
                        "classifier_refit": intervention == "retrain_ablation",
                        "final_calibration_used": False,
                        "final_test_used": False,
                        "fit_seconds": fit_seconds,
                        "code_version": version,
                    }
                    rows.extend({**metadata, **row} for row in evaluated)
            label = "reference" if feature_index is None else feature_name
            print(f"Tuning-only harm evidence: {model_name} | {label}", flush=True)

    evidence = attach_reference_deltas(
        pd.DataFrame(rows),
        key_columns=(
            "model",
            "scaling",
            "score",
            "selection_resample",
            "selection_fold",
        ),
    )
    evidence = _add_harm_components(evidence, constraints, weights)
    candidates = _aggregate_crossfit_evidence(
        evidence.loc[evidence["intervention"] != "reference"].copy(),
        constraints,
        weights,
    )
    rankings = _aggregate_and_rank(candidates, constraints)
    resample_rankings = _rank_resamples(candidates, constraints)
    stability = _rank_stability(resample_rankings)
    agreement = _formulation_agreement(rankings)
    consensus = _consensus_rankings(rankings)
    _write_outputs(
        evidence=evidence,
        resample_evidence=candidates,
        rankings=rankings,
        resample_rankings=resample_rankings,
        stability=stability,
        agreement=agreement,
        consensus=consensus,
        output_dir=output_dir,
        config=config,
        constraints=constraints,
        weights=weights,
        n_features=len(dataset.feature_manifest),
        resample_count=resample_count,
        crossfit_folds=crossfit_folds,
    )
    return rankings, stability, consensus


def _constraints_from_config(config: Mapping[str, Any]) -> HarmConstraints:
    values = config.get("constraints", {})
    return HarmConstraints(
        max_accuracy_loss=float(values.get("max_accuracy_loss", 0.01)),
        max_coverage_shortfall=float(values.get("max_coverage_shortfall", 0.03)),
    )


def _weights_from_config(config: Mapping[str, Any]) -> HarmWeights:
    values = config.get("weights", {})
    return HarmWeights(
        beta=float(values.get("beta", 4.0)),
        gamma=float(values.get("gamma", 10.0)),
        eta=float(values.get("eta", 1.0)),
    )


def _add_harm_components(
    evidence: pd.DataFrame,
    constraints: HarmConstraints,
    weights: HarmWeights,
) -> pd.DataFrame:
    result = evidence.copy()
    target = 1.0 - result["alpha"]
    result["efficiency_gain"] = result["mean_size_reduction"]
    result["coverage_shortfall"] = (target - result["coverage"]).clip(lower=0.0)
    result["coverage_deviation"] = (result["coverage"] - target).abs()
    result["conditional_violation"] = result[
        ["sscv", "class_coverage_max_deviation"]
    ].max(axis=1)
    result["accuracy_constraint_pass"] = (
        result["accuracy_loss"] <= constraints.max_accuracy_loss
    )
    result["coverage_constraint_pass"] = (
        result["coverage_shortfall"] <= constraints.max_coverage_shortfall
    )
    result["constrained_eligible"] = (
        result["accuracy_constraint_pass"] & result["coverage_constraint_pass"]
    )
    result["weighted_harm_score"] = (
        result["efficiency_gain"]
        - weights.beta * result["accuracy_loss"]
        - weights.gamma * result["coverage_shortfall"]
        - weights.eta * result["conditional_violation"]
    )
    return result


def _aggregate_crossfit_evidence(
    fold_evidence: pd.DataFrame,
    constraints: HarmConstraints,
    weights: HarmWeights,
) -> pd.DataFrame:
    group_columns = [
        "selection_resample",
        "selection_seed",
        "feature_index",
        "feature_name",
        "feature_role",
        "source_feature",
        "model",
        "scaling",
        "score",
    ]
    result = (
        fold_evidence.groupby(group_columns, dropna=False, sort=False)
        .agg(
            crossfit_folds=("selection_fold", "nunique"),
            alpha=("alpha", "first"),
            accuracy_loss=("accuracy_loss", "mean"),
            efficiency_gain=("efficiency_gain", "mean"),
            coverage=("coverage", "mean"),
            sscv=("sscv", "mean"),
            class_coverage_max_deviation=(
                "class_coverage_max_deviation",
                "mean",
            ),
        )
        .reset_index()
    )
    target = 1.0 - result["alpha"]
    result["coverage_shortfall"] = (target - result["coverage"]).clip(lower=0.0)
    result["coverage_deviation"] = (result["coverage"] - target).abs()
    result["conditional_violation"] = result[
        ["sscv", "class_coverage_max_deviation"]
    ].max(axis=1)
    result["accuracy_constraint_pass"] = (
        result["accuracy_loss"] <= constraints.max_accuracy_loss
    )
    result["coverage_constraint_pass"] = (
        result["coverage_shortfall"] <= constraints.max_coverage_shortfall
    )
    result["constrained_eligible"] = (
        result["accuracy_constraint_pass"] & result["coverage_constraint_pass"]
    )
    result["weighted_harm_score"] = (
        result["efficiency_gain"]
        - weights.beta * result["accuracy_loss"]
        - weights.gamma * result["coverage_shortfall"]
        - weights.eta * result["conditional_violation"]
    )
    return result


def _aggregate_and_rank(
    candidates: pd.DataFrame, constraints: HarmConstraints
) -> pd.DataFrame:
    group_columns = [
        "feature_index",
        "feature_name",
        "feature_role",
        "source_feature",
        "model",
        "scaling",
        "score",
    ]
    aggregated = (
        candidates.groupby(group_columns, dropna=False, sort=False)
        .agg(
            resamples=("selection_resample", "nunique"),
            mean_accuracy_loss=("accuracy_loss", "mean"),
            max_accuracy_loss=("accuracy_loss", "max"),
            mean_efficiency_gain=("efficiency_gain", "mean"),
            min_efficiency_gain=("efficiency_gain", "min"),
            mean_coverage_shortfall=("coverage_shortfall", "mean"),
            max_coverage_shortfall=("coverage_shortfall", "max"),
            mean_coverage_deviation=("coverage_deviation", "mean"),
            mean_conditional_violation=("conditional_violation", "mean"),
            max_conditional_violation=("conditional_violation", "max"),
            mean_weighted_harm_score=("weighted_harm_score", "mean"),
        )
        .reset_index()
    )
    aggregated["constrained_eligible"] = (
        (aggregated["max_accuracy_loss"] <= constraints.max_accuracy_loss)
        & (
            aggregated["max_coverage_shortfall"]
            <= constraints.max_coverage_shortfall
        )
    )
    aggregated["constrained_harm_candidate"] = (
        aggregated["constrained_eligible"]
        & (aggregated["mean_efficiency_gain"] > 0)
    )
    aggregated["pareto_eligible"] = (
        aggregated["max_coverage_shortfall"]
        <= constraints.max_coverage_shortfall
    )
    return _rank_candidates(aggregated)


def _rank_resamples(
    candidates: pd.DataFrame, constraints: HarmConstraints
) -> pd.DataFrame:
    columns = [
        "selection_resample",
        "selection_seed",
        "feature_index",
        "feature_name",
        "feature_role",
        "source_feature",
        "model",
        "scaling",
        "score",
        "accuracy_loss",
        "efficiency_gain",
        "coverage_shortfall",
        "conditional_violation",
        "weighted_harm_score",
    ]
    result = candidates[columns].rename(
        columns={
            "accuracy_loss": "mean_accuracy_loss",
            "efficiency_gain": "mean_efficiency_gain",
            "coverage_shortfall": "max_coverage_shortfall",
            "conditional_violation": "mean_conditional_violation",
            "weighted_harm_score": "mean_weighted_harm_score",
        }
    )
    result["max_accuracy_loss"] = result["mean_accuracy_loss"]
    result["constrained_eligible"] = (
        (result["max_accuracy_loss"] <= constraints.max_accuracy_loss)
        & (result["max_coverage_shortfall"] <= constraints.max_coverage_shortfall)
    )
    result["constrained_harm_candidate"] = (
        result["constrained_eligible"] & (result["mean_efficiency_gain"] > 0)
    )
    result["pareto_eligible"] = (
        result["max_coverage_shortfall"] <= constraints.max_coverage_shortfall
    )
    return _rank_candidates(result, extra_group_columns=("selection_resample",))


def _rank_candidates(
    candidates: pd.DataFrame, *, extra_group_columns: tuple[str, ...] = ()
) -> pd.DataFrame:
    result = candidates.copy()
    for column in (*RANK_COLUMNS.values(), "pareto_front"):
        result[column] = np.nan
    group_columns = [*extra_group_columns, "model", "scaling", "score"]
    for _, group in result.groupby(group_columns, sort=False):
        constrained_order = group.sort_values(
            [
                "constrained_eligible",
                "mean_efficiency_gain",
                "max_accuracy_loss",
                "max_coverage_shortfall",
                "feature_name",
            ],
            ascending=[False, False, True, True, True],
        ).index
        result.loc[constrained_order, "constrained_rank"] = np.arange(
            1, len(group) + 1
        )

        weighted_order = group.sort_values(
            ["mean_weighted_harm_score", "feature_name"],
            ascending=[False, True],
        ).index
        result.loc[weighted_order, "weighted_rank"] = np.arange(1, len(group) + 1)

        eligible = group.loc[group["pareto_eligible"]]
        if len(eligible):
            objectives = np.column_stack(
                (
                    eligible["mean_accuracy_loss"],
                    -eligible["mean_efficiency_gain"],
                    eligible["mean_conditional_violation"],
                )
            )
            result.loc[eligible.index, "pareto_front"] = pareto_fronts(objectives)
        pareto_order = result.loc[group.index].sort_values(
            [
                "pareto_eligible",
                "pareto_front",
                "mean_efficiency_gain",
                "mean_accuracy_loss",
                "feature_name",
            ],
            ascending=[False, True, False, True, True],
            na_position="last",
        ).index
        result.loc[pareto_order, "pareto_rank"] = np.arange(1, len(group) + 1)
    for column in RANK_COLUMNS.values():
        result[column] = result[column].astype(int)
    return result.sort_values(
        [*group_columns, "constrained_rank"], ignore_index=True
    )


def _rank_stability(resample_rankings: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["model", "scaling", "score"]
    for keys, group in resample_rankings.groupby(group_columns, sort=False):
        for formulation, rank_column in RANK_COLUMNS.items():
            pivot = group.pivot(
                index="feature_name",
                columns="selection_resample",
                values=rank_column,
            ).sort_index()
            correlations = [
                float(np.corrcoef(pivot[left], pivot[right])[0, 1])
                for left, right in combinations(pivot.columns, 2)
            ]
            top_features = (
                group.loc[group[rank_column] == 1, "feature_name"]
                .value_counts()
                .sort_index()
            )
            maximum_count = int(top_features.max())
            modal_top = sorted(top_features[top_features == maximum_count].index)[0]
            front_jaccards: list[float] = []
            if formulation == "pareto":
                front_sets = {
                    resample: set(
                        resample_rows.loc[
                            (resample_rows["pareto_front"] == 1)
                            & resample_rows["pareto_eligible"],
                            "feature_name",
                        ]
                    )
                    for resample, resample_rows in group.groupby(
                        "selection_resample"
                    )
                }
                for left, right in combinations(front_sets, 2):
                    union = front_sets[left] | front_sets[right]
                    front_jaccards.append(
                        len(front_sets[left] & front_sets[right]) / len(union)
                        if union
                        else 1.0
                    )
            rows.append(
                {
                    **dict(zip(group_columns, keys, strict=True)),
                    "formulation": formulation,
                    "resamples": int(pivot.shape[1]),
                    "mean_rank_correlation": float(np.mean(correlations)),
                    "min_rank_correlation": float(np.min(correlations)),
                    "max_rank_correlation": float(np.max(correlations)),
                    "modal_top_feature": modal_top,
                    "top_feature_frequency": maximum_count / pivot.shape[1],
                    "mean_front_one_jaccard": (
                        float(np.mean(front_jaccards))
                        if front_jaccards
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(group_columns + ["formulation"])


def _formulation_agreement(rankings: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["model", "scaling", "score"]
    for keys, group in rankings.groupby(group_columns, sort=False):
        for left, right in combinations(RANK_COLUMNS, 2):
            correlation = float(
                np.corrcoef(group[RANK_COLUMNS[left]], group[RANK_COLUMNS[right]])[0, 1]
            )
            rows.append(
                {
                    **dict(zip(group_columns, keys, strict=True)),
                    "left_formulation": left,
                    "right_formulation": right,
                    "rank_correlation": correlation,
                }
            )
    return pd.DataFrame(rows)


def _consensus_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
    consensus = (
        rankings.groupby(
            ["feature_index", "feature_name", "feature_role", "source_feature"],
            dropna=False,
            sort=False,
        )
        .agg(
            pipelines=("model", "size"),
            constrained_eligible_fraction=("constrained_eligible", "mean"),
            constrained_harm_candidate_fraction=(
                "constrained_harm_candidate",
                "mean",
            ),
            pareto_eligible_fraction=("pareto_eligible", "mean"),
            pareto_front_one_fraction=("pareto_front", lambda values: float((values == 1).mean())),
            mean_efficiency_gain=("mean_efficiency_gain", "mean"),
            mean_accuracy_loss=("mean_accuracy_loss", "mean"),
            mean_weighted_harm_score=("mean_weighted_harm_score", "mean"),
            mean_constrained_rank=("constrained_rank", "mean"),
            mean_weighted_rank=("weighted_rank", "mean"),
            mean_pareto_rank=("pareto_rank", "mean"),
        )
        .reset_index()
    )
    for formulation in RANK_COLUMNS:
        mean_column = f"mean_{formulation}_rank"
        consensus[f"{formulation}_consensus_rank"] = (
            consensus[mean_column].rank(method="min", ascending=True).astype(int)
        )
    return consensus.sort_values("constrained_consensus_rank", ignore_index=True)


def _write_outputs(
    *,
    evidence: pd.DataFrame,
    resample_evidence: pd.DataFrame,
    rankings: pd.DataFrame,
    resample_rankings: pd.DataFrame,
    stability: pd.DataFrame,
    agreement: pd.DataFrame,
    consensus: pd.DataFrame,
    output_dir: Path,
    config: Mapping[str, Any],
    constraints: HarmConstraints,
    weights: HarmWeights,
    n_features: int,
    resample_count: int,
    crossfit_folds: int,
) -> None:
    evidence.to_csv(output_dir / "harm_tuning_evidence.csv", index=False)
    resample_evidence.to_csv(
        output_dir / "harm_resample_evidence.csv", index=False
    )
    rankings.to_csv(output_dir / "harm_rankings.csv", index=False)
    resample_rankings.to_csv(output_dir / "harm_resample_rankings.csv", index=False)
    stability.to_csv(output_dir / "harm_rank_stability.csv", index=False)
    agreement.to_csv(output_dir / "harm_formulation_agreement.csv", index=False)
    consensus.to_csv(output_dir / "harm_consensus.csv", index=False)
    _plot_formulations(rankings, output_dir / "harm_formulations.png")
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
    )

    expected_evidence_rows = (
        (1 + n_features)
        * len(config["models"])
        * len(config["conformal"]["scores"])
        * 3
        * resample_count
        * crossfit_folds
    )
    expected_resample_rows = (
        n_features
        * len(config["models"])
        * len(config["conformal"]["scores"])
        * 3
        * resample_count
    )
    expected_ranking_rows = (
        n_features
        * len(config["models"])
        * len(config["conformal"]["scores"])
        * 3
    )
    reference_columns = [f"reference_{metric}" for metric in REFERENCE_METRICS]
    scientific_columns = [
        "accuracy",
        "coverage",
        "mean_size",
        "sscv",
        "class_coverage_max_deviation",
        "efficiency_gain",
        "accuracy_loss",
        "coverage_shortfall",
        "conditional_violation",
        "weighted_harm_score",
    ]
    primary_formulation = str(
        config.get("harm", {}).get("primary_formulation", "constrained")
    )
    if primary_formulation not in RANK_COLUMNS:
        raise ValueError(f"unknown primary harm formulation: {primary_formulation}")
    minimum_stability = float(
        config.get("harm", {})
        .get("stability", {})
        .get("min_primary_mean_rank_correlation", 0.50)
    )
    mean_stability_by_formulation = (
        stability.groupby("formulation")["mean_rank_correlation"].mean().to_dict()
    )
    checks = {
        "expected_evidence_rows": len(evidence) == expected_evidence_rows,
        "expected_resample_rows": len(resample_evidence) == expected_resample_rows,
        "expected_ranking_rows": len(rankings) == expected_ranking_rows,
        "unique_evidence_keys": not evidence.duplicated(
            [
                "selection_resample",
                "selection_fold",
                "model",
                "intervention",
                "feature_name",
                "scaling",
                "score",
            ]
        ).any(),
        "outer_tune_is_only_ranking_source": bool(
            (evidence["evidence_source"] == "outer_tune_only").all()
            and not evidence["final_calibration_used"].any()
            and not evidence["final_test_used"].any()
        ),
        "finite_scientific_values": bool(
            np.isfinite(evidence[scientific_columns].to_numpy()).all()
        ),
        "all_candidates_have_references": bool(
            evidence[reference_columns].notna().all().all()
        ),
        "complete_integer_rankings": bool(
            rankings[list(RANK_COLUMNS.values())].notna().all().all()
        ),
        "primary_rank_stability_above_threshold": bool(
            mean_stability_by_formulation[primary_formulation] >= minimum_stability
        ),
    }
    record = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "expected_evidence_rows": expected_evidence_rows,
        "observed_evidence_rows": len(evidence),
        "expected_resample_rows": expected_resample_rows,
        "observed_resample_rows": len(resample_evidence),
        "expected_ranking_rows": expected_ranking_rows,
        "observed_ranking_rows": len(rankings),
        "definitions": {
            "efficiency_gain": "reference mean size - intervention mean size",
            "accuracy_loss": "reference accuracy - intervention accuracy",
            "coverage_shortfall": "max(0, target coverage - intervention coverage)",
            "conditional_violation": "max(SSCV, class-coverage maximum target deviation)",
            "constrained": "rank efficiency gain after worst-resample accuracy and coverage safeguards",
            "weighted": "efficiency_gain - beta*accuracy_loss - gamma*coverage_shortfall - eta*conditional_violation",
            "pareto": "coverage-gated non-dominated sorting of accuracy loss, negative efficiency gain, and conditional violation",
        },
        "constraints": {
            "max_accuracy_loss": constraints.max_accuracy_loss,
            "max_coverage_shortfall": constraints.max_coverage_shortfall,
        },
        "weights": {
            "beta": weights.beta,
            "gamma": weights.gamma,
            "eta": weights.eta,
        },
        "selection_resamples": resample_count,
        "crossfit_folds_per_resample": crossfit_folds,
        "primary_formulation": primary_formulation,
        "minimum_primary_mean_rank_correlation": minimum_stability,
        "mean_rank_correlation_by_formulation": mean_stability_by_formulation,
        "protocol": {
            "classifier_fit_partition": "outer train",
            "ranking_partition": "outer tune only",
            "outer_calibration_access": "prohibited",
            "outer_test_access": "prohibited",
            "inner_partitions": [
                "scale tuning",
                "selection calibration",
                "selection evaluation",
            ],
            "cross_fitting": (
                "every outer-tuning row is evaluated once per resample and is "
                "excluded from scaling and threshold calibration in that fold"
            ),
            "paired_randomization": True,
        },
    }
    (output_dir / "harm_protocol.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    if record["status"] != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Phase 3 output validation failed: {failed}")


def _plot_formulations(rankings: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subset = rankings.loc[
        (rankings["scaling"] == "base") & (rankings["score"] == "aps")
    ]
    models = list(subset["model"].unique())
    figure, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)
    colors = {
        "strong": "#d62728",
        "weak": "#ff7f0e",
        "redundant": "#1f77b4",
        "noise": "#7f7f7f",
    }
    for column, model in enumerate(models):
        axis = axes[0, column]
        model_rows = subset.loc[subset["model"] == model]
        for role, role_rows in model_rows.groupby("feature_role"):
            axis.scatter(
                role_rows["mean_accuracy_loss"],
                role_rows["mean_efficiency_gain"],
                color=colors.get(role, "black"),
                label=role,
                alpha=0.8,
            )
        eligible = model_rows.loc[model_rows["constrained_eligible"]]
        axis.scatter(
            eligible["mean_accuracy_loss"],
            eligible["mean_efficiency_gain"],
            facecolors="none",
            edgecolors="black",
            s=90,
            linewidths=1.2,
            label="constrained eligible",
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.axvline(0.01, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(f"Base APS | {model}")
        axis.set_xlabel("Mean tuning accuracy loss")
        axis.set_ylabel("Mean tuning set-size reduction")
        axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(labels))
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
