import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from chf.data import save_split_artifact
from chf.models import fit_classifier
from chf.scaling import probabilities_from_logits

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


SCALINGS = ("base", "ts", "confts")
SCORES = ("aps", "raps")
DEFAULT_METHODS = (
    "all_features",
    "mutual_information",
    "permutation_importance",
    "rfe",
    "shap",
    "crfe",
    "conformal_harm_one_shot",
    "conformal_harm_recursive",
)
SELECTION_KEY = ("model", "method", "target_size", "selected_indices")


def run_scaling_interaction(
    config: Mapping[str, Any],
    output_dir: Path,
    repository_root: Path,
    *,
    selections_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate frozen Phase 5 subsets under Base/TS/ConfTS x APS/RAPS.

    Phase 6 deliberately consumes only the Phase 5 *selection* artifact. It
    evaluates every preregistered deterministic selector, rather than choosing
    a comparator from Phase 5 test results.
    """
    seed = int(config["seed"])
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = dataset_from_config(config, repository_root)
    split = experiment_split(config, dataset)
    identifier = split_id(split)
    selection_train, selection_tune = selection_data_indices(
        config, split, dataset.labels, seed=seed
    )
    selection_identifier = selection_data_id(selection_train, selection_tune)
    source_path = selections_path or output_dir / "baseline_selections.csv"
    selections = _load_frozen_selections(
        source_path,
        config,
        n_total_features=len(dataset.feature_names),
        expected_selection_data_id=selection_identifier,
    )
    save_split_artifact(
        output_dir / "scaling_interaction_split_indices.npz",
        split,
        seed=seed,
        metadata={
            "experiment_name": config["experiment_name"],
            "phase": int(config.get("phase", 6)),
            "split_id": identifier,
            "selection_data_id": selection_identifier,
        },
    )

    results = _evaluate_frozen_selections(
        selections=selections,
        config=config,
        dataset=dataset,
        split=split,
        seed=seed,
        identifier=identifier,
        version=code_version(repository_root),
    )
    tolerance = float(
        config.get("scaling_interaction", {}).get(
            "descriptive_interaction_tolerance", 0.01
        )
    )
    interactions = interaction_decomposition(results, tolerance=tolerance)
    rank_stability = scaling_rank_stability(results)
    summary = _summarize_interactions(interactions)
    _write_outputs(
        selections=selections,
        results=results,
        interactions=interactions,
        rank_stability=rank_stability,
        summary=summary,
        output_dir=output_dir,
        config=config,
        selections_path=source_path,
        tolerance=tolerance,
        selection_identifier=selection_identifier,
    )
    return results, interactions, rank_stability, summary


def _load_frozen_selections(
    selections_path: Path,
    config: Mapping[str, Any],
    *,
    n_total_features: int,
    expected_selection_data_id: str,
) -> pd.DataFrame:
    if not selections_path.exists():
        raise FileNotFoundError(
            f"Phase 5 selection artifact not found: {selections_path}. "
            "Run experiments/06_required_baselines.py first or pass --selections."
        )
    selections = pd.read_csv(selections_path)
    required_columns = {
        "model",
        "method",
        "target_size",
        "n_features",
        "selected_indices",
        "selected_features",
        "ranking_source",
        "selection_seed",
        "repetition",
        "subset_frozen_before_final_calibration",
        "selection_data_id",
    }
    missing_columns = required_columns.difference(selections.columns)
    if missing_columns:
        raise ValueError(
            f"selection artifact is missing columns: {sorted(missing_columns)}"
        )

    requested = tuple(
        config.get("scaling_interaction", {}).get("methods", DEFAULT_METHODS)
    )
    unknown_requested = set(requested).difference(DEFAULT_METHODS)
    if unknown_requested:
        raise ValueError(
            f"unsupported Phase 6 methods: {sorted(unknown_requested)}"
        )
    selections = selections.loc[selections["method"].isin(requested)].copy()
    missing_methods = set(requested).difference(selections["method"])
    if missing_methods:
        raise ValueError(
            f"Phase 5 artifact lacks requested methods: {sorted(missing_methods)}"
        )
    if "all_features" not in requested:
        raise ValueError("Phase 6 methods must include all_features as the reference")
    if (selections["method"] == "random").any() or (
        selections["repetition"].astype(int) != -1
    ).any():
        raise ValueError("Phase 6 accepts deterministic frozen selections only")
    if set(selections["model"]) != set(config["models"]):
        raise ValueError("selection models do not match the configured model families")
    if (selections["selection_seed"].astype(int) != int(config["seed"])).any():
        raise ValueError("selection seeds do not match the configured paired seed")
    if not selections["selection_data_id"].eq(expected_selection_data_id).all():
        raise ValueError(
            "selection artifact does not match the configured selection data"
        )
    if not selections["subset_frozen_before_final_calibration"].astype(bool).all():
        raise ValueError("every Phase 6 subset must already be frozen")
    if selections.duplicated(list(SELECTION_KEY)).any():
        raise ValueError("selection artifact contains duplicate deterministic subsets")

    for row in selections.itertuples(index=False):
        selected = tuple(json.loads(row.selected_indices))
        if any(not isinstance(index, int) for index in selected):
            raise ValueError("selected_indices must contain integers")
        if len(selected) != len(set(selected)):
            raise ValueError("selected_indices must be unique")
        if not selected or min(selected) < 0 or max(selected) >= n_total_features:
            raise ValueError("selected_indices contain an out-of-range feature")
        if len(selected) != int(row.n_features) or len(selected) != int(row.target_size):
            raise ValueError("selected_indices disagree with the recorded subset size")
        if row.method == "all_features" and selected != tuple(range(n_total_features)):
            raise ValueError("all_features must contain every feature in canonical order")
    reference_counts = selections.loc[
        selections["method"] == "all_features"
    ].groupby("model").size()
    if not (reference_counts == 1).all() or len(reference_counts) != len(config["models"]):
        raise ValueError("exactly one all_features reference is required per model")
    return selections.reset_index(drop=True)


def _selection_type(method: str) -> str:
    if method == "all_features":
        return "all_features"
    if method.startswith("conformal_harm_"):
        return "proposed_selection"
    return "standard_selection"


def _mean_true_label_rank(logits: np.ndarray, labels: np.ndarray) -> float:
    true_logits = logits[np.arange(len(labels)), labels]
    ranks = 1 + np.count_nonzero(logits > true_logits[:, None], axis=1)
    return float(ranks.mean())


def _mean_entropy(probabilities: np.ndarray) -> float:
    positive = probabilities > 0
    terms = np.zeros_like(probabilities, dtype=float)
    terms[positive] = probabilities[positive] * np.log(probabilities[positive])
    return float(-terms.sum(axis=1).mean())


def _evaluate_frozen_selections(
    *,
    selections: pd.DataFrame,
    config: Mapping[str, Any],
    dataset: Any,
    split: Any,
    seed: int,
    identifier: str,
    version: str,
) -> pd.DataFrame:
    calibration_uniforms = np.random.default_rng(seed + 600_001).random(
        len(split.calibration)
    )
    test_uniforms = np.random.default_rng(seed + 600_002).random(len(split.test))
    logits_cache: dict[
        tuple[str, tuple[int, ...]], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    evaluated_cache: dict[
        tuple[str, tuple[int, ...]], list[dict[str, Any]]
    ] = {}

    for selection in selections.to_dict("records"):
        selected = tuple(json.loads(selection["selected_indices"]))
        cache_key = (str(selection["model"]), selected)
        if cache_key in logits_cache:
            continue
        model_name = str(selection["model"])
        fitted = fit_classifier(
            dataset.features[split.train][:, selected],
            dataset.labels[split.train],
            config["models"][model_name],
            seed=seed,
        )
        logits_cache[cache_key] = (
            fitted.logits(dataset.features[split.tune][:, selected]),
            fitted.logits(dataset.features[split.calibration][:, selected]),
            fitted.logits(dataset.features[split.test][:, selected]),
        )
        print(
            f"Phase 6 fit: {model_name} | {len(selected)} features",
            flush=True,
        )

    reference_logits: dict[str, np.ndarray] = {}
    for selection in selections.loc[
        selections["method"] == "all_features"
    ].to_dict("records"):
        selected = tuple(json.loads(selection["selected_indices"]))
        reference_logits[str(selection["model"])] = logits_cache[
            (str(selection["model"]), selected)
        ][2]

    rows: list[dict[str, Any]] = []
    for selection in selections.to_dict("records"):
        model_name = str(selection["model"])
        selected = tuple(json.loads(selection["selected_indices"]))
        cache_key = (model_name, selected)
        logits_tune, logits_calibration, logits_test = logits_cache[cache_key]
        if cache_key not in evaluated_cache:
            evaluated_cache[cache_key] = evaluate_logits(
                logits_tune=logits_tune,
                logits_calibration=logits_calibration,
                logits_test=logits_test,
                labels_tune=dataset.labels[split.tune],
                labels_calibration=dataset.labels[split.calibration],
                labels_test=dataset.labels[split.test],
                config=config,
                calibration_uniforms=calibration_uniforms,
                test_uniforms=test_uniforms,
                seed=seed + 600_000,
                included_scores=SCORES,
                included_scalings=SCALINGS,
            )
        reference_test_logits = reference_logits[model_name]
        top1_disagreement = float(
            np.mean(
                np.argmax(logits_test, axis=1)
                != np.argmax(reference_test_logits, axis=1)
            )
        )
        mean_true_rank = _mean_true_label_rank(
            logits_test, dataset.labels[split.test]
        )
        reference_true_rank = _mean_true_label_rank(
            reference_test_logits, dataset.labels[split.test]
        )
        metadata = {
            "experiment": config["experiment_name"],
            "dataset": dataset_name(dataset),
            "seed": seed,
            "split_id": identifier,
            "classifier_refit": True,
            "selection_type": _selection_type(selection["method"]),
            "final_calibration_used_once_per_pipeline": True,
            "final_test_used_once_per_pipeline": True,
            "code_version": version,
            **selection,
            "top1_disagreement_rate_vs_all": top1_disagreement,
            "mean_true_label_rank": mean_true_rank,
            "mean_true_label_rank_change_vs_all": mean_true_rank
            - reference_true_rank,
        }
        for evaluated in evaluated_cache[cache_key]:
            temperature = float(evaluated["temperature"])
            calibration_probabilities = probabilities_from_logits(
                logits_calibration, temperature
            )
            test_probabilities = probabilities_from_logits(logits_test, temperature)
            concentration = {
                "calibration_mean_max_probability": float(
                    calibration_probabilities.max(axis=1).mean()
                ),
                "calibration_mean_entropy": _mean_entropy(
                    calibration_probabilities
                ),
                "test_mean_entropy": _mean_entropy(test_probabilities),
            }
            rows.append({**metadata, **evaluated, **concentration})

    results = pd.DataFrame(rows)
    reference_columns = [
        "temperature",
        "threshold",
        "accuracy",
        "coverage",
        "mean_size",
        "sscv",
        "class_coverage_max_deviation",
        "test_mean_max_probability",
        "test_mean_entropy",
    ]
    reference = results.loc[
        results["method"] == "all_features",
        ["model", "scaling", "score", *reference_columns],
    ].rename(
        columns={column: f"all_features_{column}" for column in reference_columns}
    )
    results = results.merge(
        reference,
        on=["model", "scaling", "score"],
        how="left",
        validate="many_to_one",
    )
    results["accuracy_loss_vs_all"] = (
        results["all_features_accuracy"] - results["accuracy"]
    )
    results["coverage_delta_vs_all"] = (
        results["coverage"] - results["all_features_coverage"]
    )
    results["mean_size_reduction_vs_all"] = (
        results["all_features_mean_size"] - results["mean_size"]
    )
    results["sscv_delta_vs_all"] = results["sscv"] - results["all_features_sscv"]
    results["temperature_delta_vs_all"] = (
        results["temperature"] - results["all_features_temperature"]
    )
    results["threshold_delta_vs_all"] = (
        results["threshold"] - results["all_features_threshold"]
    )
    results["mean_max_probability_delta_vs_all"] = (
        results["test_mean_max_probability"]
        - results["all_features_test_mean_max_probability"]
    )
    results["mean_entropy_delta_vs_all"] = (
        results["test_mean_entropy"] - results["all_features_test_mean_entropy"]
    )
    selected_mask = results["method"] != "all_features"
    results["efficiency_rank_within_target_size"] = np.nan
    results.loc[selected_mask, "efficiency_rank_within_target_size"] = (
        results.loc[selected_mask]
        .groupby(["model", "target_size", "scaling", "score"])["mean_size"]
        .rank(method="average", ascending=True)
    )
    return results


def interaction_decomposition(
    results: pd.DataFrame, *, tolerance: float
) -> pd.DataFrame:
    """Decompose joint size gain into feature, scaling, and interaction terms.

    A positive interaction means the scaling gain is larger after feature
    selection (synergy). A negative interaction means the gains overlap or
    oppose one another. These labels are descriptive until multi-seed Phase 8.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("interaction tolerance must be finite and non-negative")
    required = {
        "model",
        "method",
        "target_size",
        "selection_type",
        "scaling",
        "score",
        "mean_size",
        "temperature",
        "threshold",
        "test_mean_max_probability",
        "test_mean_entropy",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"interaction results are missing columns: {sorted(missing)}")

    references = results.loc[results["method"] == "all_features"].set_index(
        ["model", "score", "scaling"]
    )
    selected_rows = results.loc[results["method"] != "all_features"]
    decomposition: list[dict[str, Any]] = []
    for keys, group in selected_rows.groupby(
        ["model", "method", "target_size", "selection_type", "score"],
        sort=False,
    ):
        model, method, target_size, selection_type, score = keys
        indexed = group.set_index("scaling")
        selected_base = indexed.loc["base"]
        all_base = references.loc[(model, score, "base")]
        for scaling in ("ts", "confts"):
            selected_scaled = indexed.loc[scaling]
            all_scaled = references.loc[(model, score, scaling)]
            feature_gain_at_base = float(
                all_base["mean_size"] - selected_base["mean_size"]
            )
            scaling_gain_on_all = float(
                all_base["mean_size"] - all_scaled["mean_size"]
            )
            expected_additive_gain = feature_gain_at_base + scaling_gain_on_all
            observed_joint_gain = float(
                all_base["mean_size"] - selected_scaled["mean_size"]
            )
            interaction_gain = observed_joint_gain - expected_additive_gain
            if interaction_gain > tolerance:
                interaction_label = "synergistic"
            elif interaction_gain < -tolerance:
                interaction_label = "redundant_or_antagonistic"
            else:
                interaction_label = "approximately_additive"
            decomposition.append(
                {
                    "model": model,
                    "method": method,
                    "target_size": int(target_size),
                    "selection_type": selection_type,
                    "score": score,
                    "scaling": scaling,
                    "feature_gain_at_base": feature_gain_at_base,
                    "scaling_gain_on_all_features": scaling_gain_on_all,
                    "expected_additive_gain": expected_additive_gain,
                    "observed_joint_gain": observed_joint_gain,
                    "interaction_size_gain": interaction_gain,
                    "feature_gain_under_scaling": float(
                        all_scaled["mean_size"] - selected_scaled["mean_size"]
                    ),
                    "scaling_gain_on_selected": float(
                        selected_base["mean_size"] - selected_scaled["mean_size"]
                    ),
                    "interaction_label": interaction_label,
                    "descriptive_tolerance": tolerance,
                    "selected_temperature": float(selected_scaled["temperature"]),
                    "all_features_temperature": float(all_scaled["temperature"]),
                    "temperature_delta_vs_all": float(
                        selected_scaled["temperature"]
                        - all_scaled["temperature"]
                    ),
                    "selected_threshold": float(selected_scaled["threshold"]),
                    "all_features_threshold": float(all_scaled["threshold"]),
                    "threshold_delta_vs_all": float(
                        selected_scaled["threshold"] - all_scaled["threshold"]
                    ),
                    "mean_max_probability_delta_vs_all": float(
                        selected_scaled["test_mean_max_probability"]
                        - all_scaled["test_mean_max_probability"]
                    ),
                    "mean_entropy_delta_vs_all": float(
                        selected_scaled["test_mean_entropy"]
                        - all_scaled["test_mean_entropy"]
                    ),
                }
            )
    return pd.DataFrame(decomposition)


def _rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2:
        return 1.0
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = float(
        np.sqrt(np.sum(first_centered**2) * np.sum(second_centered**2))
    )
    if denominator == 0:
        return 1.0 if np.array_equal(first, second) else 0.0
    return float(np.sum(first_centered * second_centered) / denominator)


def scaling_rank_stability(results: pd.DataFrame) -> pd.DataFrame:
    """Compare matched-size selector rankings between scaling methods."""
    selected = results.loc[results["method"] != "all_features"].copy()
    rows: list[dict[str, Any]] = []
    for keys, group in selected.groupby(["model", "target_size", "score"], sort=False):
        model, target_size, score = keys
        pivot = group.pivot(index="method", columns="scaling", values="mean_size")
        if set(SCALINGS).difference(pivot.columns):
            raise ValueError("every selector requires Base, TS, and ConfTS results")
        ranks = pivot.rank(method="average", ascending=True)
        for first, second in combinations(SCALINGS, 2):
            first_ranks = ranks[first].to_numpy(dtype=float)
            second_ranks = ranks[second].to_numpy(dtype=float)
            first_best = sorted(
                pivot.index[pivot[first] == pivot[first].min()].tolist()
            )
            second_best = sorted(
                pivot.index[pivot[second] == pivot[second].min()].tolist()
            )
            rows.append(
                {
                    "model": model,
                    "target_size": int(target_size),
                    "score": score,
                    "first_scaling": first,
                    "second_scaling": second,
                    "methods_compared": len(pivot),
                    "spearman_rank_correlation": _rank_correlation(
                        first_ranks, second_ranks
                    ),
                    "methods_with_changed_rank": int(
                        np.count_nonzero(first_ranks != second_ranks)
                    ),
                    "first_best_methods": json.dumps(first_best),
                    "second_best_methods": json.dumps(second_best),
                    "best_method_changed": first_best != second_best,
                }
            )
    return pd.DataFrame(rows)


def _summarize_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    summary = (
        interactions.groupby(["selection_type", "score", "scaling"], sort=False)
        .agg(
            comparisons=("interaction_size_gain", "size"),
            mean_feature_gain_at_base=("feature_gain_at_base", "mean"),
            mean_scaling_gain_on_all=("scaling_gain_on_all_features", "mean"),
            mean_observed_joint_gain=("observed_joint_gain", "mean"),
            mean_interaction_size_gain=("interaction_size_gain", "mean"),
            min_interaction_size_gain=("interaction_size_gain", "min"),
            max_interaction_size_gain=("interaction_size_gain", "max"),
            synergistic_count=(
                "interaction_label",
                lambda values: int((values == "synergistic").sum()),
            ),
            approximately_additive_count=(
                "interaction_label",
                lambda values: int((values == "approximately_additive").sum()),
            ),
            redundant_or_antagonistic_count=(
                "interaction_label",
                lambda values: int((values == "redundant_or_antagonistic").sum()),
            ),
        )
        .reset_index()
    )
    return summary


def _write_outputs(
    *,
    selections: pd.DataFrame,
    results: pd.DataFrame,
    interactions: pd.DataFrame,
    rank_stability: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    config: Mapping[str, Any],
    selections_path: Path,
    tolerance: float,
    selection_identifier: str,
) -> None:
    results.to_csv(output_dir / "scaling_interaction_results.csv", index=False)
    interactions.to_csv(
        output_dir / "scaling_interaction_decomposition.csv", index=False
    )
    rank_stability.to_csv(
        output_dir / "scaling_rank_stability.csv", index=False
    )
    summary.to_csv(output_dir / "scaling_interaction_summary.csv", index=False)
    (output_dir / "scaling_interaction_resolved_config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
    )
    _plot_proposed_interactions(
        results, output_dir / "scaling_interaction_proposed.png"
    )

    selection_groups = results.groupby(list(SELECTION_KEY), dropna=False, sort=False)
    expected_interactions = (
        int((selections["method"] != "all_features").sum())
        * len(SCORES)
        * (len(SCALINGS) - 1)
    )
    interaction_identity = (
        interactions["interaction_size_gain"]
        - (
            interactions["feature_gain_under_scaling"]
            - interactions["feature_gain_at_base"]
        )
    ).abs()
    rank_columns = [
        "top1_disagreement_rate_vs_all",
        "mean_true_label_rank",
        "mean_true_label_rank_change_vs_all",
    ]
    numerical_safety_columns = [
        "calibration_zero_count",
        "calibration_exactly_one_count",
        "test_zero_count",
        "test_exactly_one_count",
    ]
    core_metrics = [
        "temperature",
        "threshold",
        "accuracy",
        "coverage",
        "mean_size",
        "sscv",
        "test_mean_max_probability",
        "test_mean_entropy",
    ]
    coverage_tolerance = float(
        config.get("stability", {}).get("max_coverage_deviation", 0.03)
    )
    checks = {
        "all_feature_set_types_present": set(results["selection_type"])
        == {"all_features", "standard_selection", "proposed_selection"},
        "all_six_pipelines_per_selection": bool(
            (selection_groups.size() == len(SCALINGS) * len(SCORES)).all()
        ),
        "exact_scaling_score_factorial": set(
            zip(results["scaling"], results["score"], strict=False)
        )
        == set((scaling, score) for scaling in SCALINGS for score in SCORES),
        "no_random_selection_evaluated": not (results["method"] == "random").any(),
        "subsets_frozen_before_calibration": bool(
            results["subset_frozen_before_final_calibration"].all()
        ),
        "identical_selection_data": bool(
            results["selection_data_id"].eq(selection_identifier).all()
        ),
        "fresh_thresholds_and_metrics_finite": bool(
            np.isfinite(results[core_metrics].to_numpy()).all()
        ),
        "coverage_within_stability_tolerance": bool(
            ((results["coverage"] - (1 - results["alpha"])).abs()
             <= coverage_tolerance + 1e-12).all()
        ),
        "no_zero_or_saturated_probabilities": bool(
            (results[numerical_safety_columns] == 0).all().all()
        ),
        "positive_temperature_preserves_accuracy": bool(
            (selection_groups["accuracy"].nunique() == 1).all()
        ),
        "positive_temperature_preserves_class_rankings": bool(
            (selection_groups[rank_columns].nunique() == 1).all().all()
        ),
        "interaction_row_count": len(interactions) == expected_interactions,
        "interaction_identity_holds": bool((interaction_identity <= 1e-12).all()),
        "finite_rank_stability": bool(
            np.isfinite(rank_stability["spearman_rank_correlation"]).all()
        ),
    }
    selection_digest = hashlib.sha256(selections_path.read_bytes()).hexdigest()
    record = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "protocol": {
            "classifier_fit_partition": "outer train",
            "temperature_tuning_partition": "outer tune only",
            "confts_tuning": "disjoint threshold/loss halves inside outer tune",
            "feature_subsets": "all requested deterministic Phase 5 selections",
            "selection_data_id": selection_identifier,
            "phase_5_test_results_used_for_method_choice": False,
            "final_calibration_access": "once per frozen scaling-score pipeline",
            "final_test_access": "once per frozen scaling-score pipeline",
            "paired_randomization": "shared calibration/test uniforms across every subset",
            "interaction_definition": (
                "observed joint size gain - (Base feature gain + "
                "all-features scaling gain)"
            ),
            "interaction_labels": "descriptive only; inference deferred to Phase 8",
        },
        "descriptive_interaction_tolerance": tolerance,
        "coverage_target": 1.0 - float(config["conformal"]["alpha"]),
        "coverage_stability_tolerance": coverage_tolerance,
        "phase_5_selection_artifact": str(selections_path),
        "phase_5_selection_sha256": selection_digest,
        "observed_selection_rows": len(selections),
        "observed_result_rows": len(results),
        "observed_interaction_rows": len(interactions),
        "expected_interaction_rows": expected_interactions,
    }
    (output_dir / "scaling_interaction_protocol.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    if record["status"] != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Phase 6 validation failed: {failed}")


def _plot_proposed_interactions(results: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_data = results.loc[
        results["selection_type"].isin({"all_features", "proposed_selection"})
    ].copy()
    models = list(plot_data["model"].drop_duplicates())
    scores = list(SCORES)
    figure, axes = plt.subplots(
        len(models), len(scores), figsize=(12, 4.5 * len(models)), squeeze=False
    )
    display_names = {
        "all_features": "All features",
        "conformal_harm_one_shot": "Proposed one-shot",
        "conformal_harm_recursive": "Proposed recursive",
    }
    scaling_positions = {name: index for index, name in enumerate(SCALINGS)}
    for row_index, model in enumerate(models):
        for column_index, score in enumerate(scores):
            axis = axes[row_index, column_index]
            panel = plot_data.loc[
                (plot_data["model"] == model) & (plot_data["score"] == score)
            ]
            for _, group in panel.groupby("selected_indices", sort=False):
                methods = list(group["method"].drop_duplicates())
                curve = group.loc[group["method"] == methods[0]]
                ordered = curve.assign(
                    scaling_order=curve["scaling"].map(scaling_positions)
                ).sort_values("scaling_order")
                label = " / ".join(
                    display_names.get(method, method) for method in methods
                )
                axis.plot(
                    ordered["scaling_order"],
                    ordered["mean_size"],
                    marker="o",
                    linewidth=2,
                    label=label,
                )
            axis.set_xticks(range(len(SCALINGS)), [name.upper() for name in SCALINGS])
            axis.set_title(f"{model} - {score.upper()}")
            axis.set_xlabel("Scaling method")
            axis.set_ylabel("Mean prediction-set size")
            axis.grid(axis="y", alpha=0.25)
            axis.legend(fontsize=8)
    figure.suptitle(
        "Feature-selection interaction with temperature scaling", y=0.99
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
