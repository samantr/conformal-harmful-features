import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from chf.data import save_split_artifact
from chf.models import fit_classifier
from chf.selection import (
    crfe_order,
    mutual_information_order,
    permutation_order,
    rfe_order,
    shap_order,
)

from .progressive_selection import run_progressive_selection
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


def run_required_baselines(
    config: Mapping[str, Any], output_dir: Path, repository_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare required selectors at Phase 4's tuning-selected subset sizes."""
    seed = int(config["seed"])
    dataset = dataset_from_config(config, repository_root)
    split = experiment_split(config, dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    identifier = split_id(split)
    selection_train, selection_tune = selection_data_indices(
        config, split, dataset.labels, seed=seed
    )
    selection_identifier = selection_data_id(selection_train, selection_tune)
    save_split_artifact(
        output_dir / "split_indices.npz", split, seed=seed,
        metadata={"experiment_name": config["experiment_name"],
                  "phase": int(config.get("phase", 5)),
                  "split_id": identifier,
                  "selection_data_id": selection_identifier},
    )

    # Phase 4 is rerun in a private subdirectory so the proposed choices are
    # obtained from tune only and remain frozen before this phase sees cal/test.
    proposed_dir = output_dir / "proposed_selection"
    proposed_path = proposed_dir / "progressive_consensus_paths.csv"
    if proposed_path.exists():
        progressive_paths = pd.read_csv(proposed_path)
        valid_cached_selection = (
            "selection_data_id" in progressive_paths
            and progressive_paths["selection_data_id"].eq(
                selection_identifier
            ).all()
        )
    else:
        valid_cached_selection = False
    if bool(config.get("checkpointing", {}).get("enabled", False)):
        # Reconstruct the path from manifest-validated candidate shards. A CSV
        # alone does not prove that code, configuration, and split still match.
        valid_cached_selection = False
    if not valid_cached_selection:
        _, progressive_paths, _ = run_progressive_selection(
            config, proposed_dir, repository_root
        )
    primary_proposed = progressive_paths.loc[
        progressive_paths["selected_for_final"]
    ].copy()
    train_x = dataset.features[selection_train]
    train_y = dataset.labels[selection_train]
    tune_x = dataset.features[selection_tune]
    tune_y = dataset.labels[selection_tune]
    baseline_config = config.get("baselines", {})
    random_repetitions = int(baseline_config.get("random_repetitions", 10))
    permutation_repeats = int(baseline_config.get("permutation_repeats", 10))
    crfe_lambda = float(baseline_config.get("crfe_lambda", 0.5))
    shap_background_size = int(baseline_config.get("shap_background_size", 30))
    shap_evaluation_size = int(baseline_config.get("shap_evaluation_size", 20))
    shap_max_evaluations = int(baseline_config.get(
        "shap_max_evaluations", 2 * train_x.shape[1] + 1
    ))
    target_removals = tuple(
        sorted({int(value) for value in baseline_config.get("target_removals", ())})
    )
    if any(value <= 0 or value >= train_x.shape[1] for value in target_removals):
        raise ValueError(
            "baselines.target_removals must lie between one and n_features - 1"
        )
    include_progressive_steps = bool(
        baseline_config.get("include_progressive_steps", False)
    )
    if include_progressive_steps:
        proposed = progressive_paths.loc[
            progressive_paths["n_removed"].astype(int).eq(0)
            | progressive_paths["n_removed"].astype(int).isin(target_removals)
            | progressive_paths["selected_for_final"].astype(bool)
        ].copy()
    else:
        proposed = primary_proposed.copy()
    primary_keys = set(
        primary_proposed[["model", "method", "selected_indices"]].itertuples(
            index=False, name=None
        )
    )

    shared_orders = {
        "mutual_information": mutual_information_order(train_x, train_y, seed=seed),
        "rfe": rfe_order(train_x, train_y, seed=seed),
        "crfe": crfe_order(
            train_x, train_y, tune_x, tune_y, seed=seed, lambda_value=crfe_lambda
        ),
    }
    selection_rows: list[dict[str, Any]] = []
    all_indices = tuple(range(train_x.shape[1]))
    for model_name, model_config in config["models"].items():
        fitted_all = fit_classifier(train_x, train_y, model_config, seed=seed)
        model_orders = dict(shared_orders)
        model_orders["permutation_importance"] = permutation_order(
            fitted_all, tune_x, tune_y, seed=seed + 501, repeats=permutation_repeats
        )
        rng = np.random.default_rng(seed + 601)
        background = train_x[rng.choice(len(train_x), size=min(shap_background_size, len(train_x)), replace=False)]
        evaluation = tune_x[rng.choice(len(tune_x), size=min(shap_evaluation_size, len(tune_x)), replace=False)]
        model_orders["shap"] = shap_order(
            fitted_all.predict_proba, background, evaluation, seed=seed + 602,
            max_evaluations=shap_max_evaluations,
        )
        model_primary = primary_proposed.loc[
            primary_proposed["model"] == model_name
        ]
        primary_targets = set(model_primary["n_features"].astype(int))
        targets = set(
            proposed.loc[proposed["model"] == model_name, "n_features"].astype(int)
        )
        targets.update(train_x.shape[1] - value for value in target_removals)
        targets = sorted(targets)
        selection_rows.append(_selection_row(
            model_name, "all_features", None, all_indices, dataset.feature_names,
            ranking_source="none", selection_seed=seed, primary_selected=True,
            ranking=all_indices,
        ))
        for target_size in targets:
            for method, (order, scores) in model_orders.items():
                selected = tuple(sorted(order[:target_size].astype(int).tolist()))
                selection_rows.append(_selection_row(
                    model_name, method, target_size, selected, dataset.feature_names,
                    ranking_source=(
                        "selection_train"
                        if method in {"mutual_information", "rfe"}
                        else "selection_train+tune"
                    ),
                    selection_seed=seed, scores=scores,
                    primary_selected=target_size in primary_targets,
                    ranking=tuple(int(value) for value in order),
                ))
            repetitions = (
                random_repetitions
                if not target_removals or target_size in primary_targets
                else 0
            )
            for repetition in range(repetitions):
                random_seed = seed + 10_000 + 997 * repetition + 31 * target_size
                selected = tuple(sorted(np.random.default_rng(random_seed).choice(
                    len(all_indices), size=target_size, replace=False
                ).tolist()))
                selection_rows.append(_selection_row(
                    model_name, "random", target_size, selected, dataset.feature_names,
                    ranking_source="random", selection_seed=random_seed,
                    repetition=repetition, primary_selected=True,
                    ranking=selected,
                ))
        for row in proposed.loc[proposed["model"] == model_name].to_dict("records"):
            selected = tuple(json.loads(row["selected_indices"]))
            selection_rows.append(_selection_row(
                model_name, f"conformal_harm_{row['method']}", len(selected), selected,
                dataset.feature_names,
                ranking_source="selection_train+tune_crossfit",
                selection_seed=seed,
                primary_selected=(
                    model_name,
                    str(row["method"]),
                    row["selected_indices"],
                ) in primary_keys,
                metadata={
                    "tuning_n_removed": int(row["n_removed"]),
                    "tuning_max_accuracy_loss": float(row["max_accuracy_loss"]),
                    "tuning_max_coverage_shortfall": float(
                        row["max_coverage_shortfall"]
                    ),
                    "tuning_efficiency_gain": float(
                        row["cumulative_efficiency_gain"]
                    ),
                    "tuning_conditional_violation": float(
                        row["mean_conditional_violation"]
                    ),
                },
            ))

    selections = pd.DataFrame(selection_rows).drop_duplicates(
        ["model", "method", "target_size", "selected_indices", "repetition"]
    ).reset_index(drop=True)
    selections["selection_data_id"] = selection_identifier
    if bool(baseline_config.get("selection_only", False)):
        selections.to_csv(output_dir / "baseline_selections.csv", index=False)
        (output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
        )
        record = {
            "status": "PASS",
            "checks": {
                "all_required_methods_present": {
                    "all_features", "random", "mutual_information",
                    "permutation_importance", "rfe", "shap", "crfe",
                    "conformal_harm_one_shot",
                    "conformal_harm_recursive",
                }.issubset(set(selections["method"])),
                "subsets_frozen_before_calibration": bool(
                    selections["subset_frozen_before_final_calibration"].all()
                ),
                "identical_selection_data": bool(
                    selections["selection_data_id"].eq(selection_identifier).all()
                ),
            },
            "protocol": {
                "mode": "selection_only_for_phase8",
                "ranking_partitions": "selection train or selection train+tune",
                "final_calibration_access": "none in this stage",
                "final_test_access": "none in this stage",
                "selection_data_id": selection_identifier,
            },
            "observed_selection_rows": len(selections),
            "observed_final_rows": 0,
        }
        if not all(record["checks"].values()):
            raise RuntimeError("Phase 8 baseline-selection validation failed")
        (output_dir / "baseline_protocol.json").write_text(
            json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
        )
        return selections, pd.DataFrame(), pd.DataFrame()
    checkpoint_store = _baseline_checkpoint_store(
        config=config,
        output_dir=output_dir,
        seed=seed,
        split_identifier=identifier,
        selection_identifier=selection_identifier,
        repository_root=repository_root,
    )
    results = _evaluate_frozen_selections(
        selections,
        config,
        dataset,
        split,
        seed,
        identifier,
        code_version(repository_root),
        checkpoint_store=checkpoint_store,
    )
    summary = _summarize(results)
    _write_outputs(
        selections, results, summary, output_dir, config,
        random_repetitions=random_repetitions,
        selection_identifier=selection_identifier,
        selection_train_size=len(selection_train),
        selection_tune_size=len(selection_tune),
    )
    return selections, results, summary


def _selection_row(
    model: str, method: str, target_size: int | None, selected: tuple[int, ...],
    feature_names: tuple[str, ...], *, ranking_source: str, selection_seed: int,
    scores: np.ndarray | None = None, repetition: int | None = None,
    primary_selected: bool = False,
    ranking: tuple[int, ...] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "model": model, "method": method,
        "target_size": len(selected) if target_size is None else target_size,
        "n_features": len(selected), "selected_indices": json.dumps(selected),
        "selected_features": json.dumps([feature_names[index] for index in selected]),
        "ranking_source": ranking_source, "selection_seed": selection_seed,
        "repetition": -1 if repetition is None else repetition,
        "feature_scores": None if scores is None else json.dumps(np.asarray(scores).tolist()),
        "feature_ranking": None if ranking is None else json.dumps(ranking),
        "subset_frozen_before_final_calibration": True,
        "phase8_primary_selected": bool(primary_selected),
        **dict(metadata or {}),
    }


def _evaluate_frozen_selections(
    selections: pd.DataFrame, config: Mapping[str, Any], dataset: Any, split: Any,
    seed: int, identifier: str, version: str,
    *, checkpoint_store: CheckpointStore | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    calibration_uniforms = np.random.default_rng(seed + 500_001).random(len(split.calibration))
    test_uniforms = np.random.default_rng(seed + 500_002).random(len(split.test))
    for selection in selections.to_dict("records"):
        checkpoint_key = {
            "model": selection["model"],
            "method": selection["method"],
            "target_size": int(selection["target_size"]),
            "selected_indices": selection["selected_indices"],
            "repetition": int(selection["repetition"]),
        }
        restored = (
            None
            if checkpoint_store is None
            else checkpoint_store.load_frame(checkpoint_key)
        )
        if restored is not None:
            if len(restored) != 1:
                raise ValueError("a baseline checkpoint must contain exactly one row")
            rows.append(restored.iloc[0].to_dict())
            print(
                f"Baseline resume: {selection['model']} | "
                f"{selection['method']} | {selection['n_features']}",
                flush=True,
            )
            continue
        selected = tuple(json.loads(selection["selected_indices"]))
        fitted = fit_classifier(
            dataset.features[split.train][:, selected], dataset.labels[split.train],
            config["models"][selection["model"]], seed=seed,
        )
        evaluated = evaluate_logits(
            logits_tune=fitted.logits(dataset.features[split.tune][:, selected]),
            logits_calibration=fitted.logits(dataset.features[split.calibration][:, selected]),
            logits_test=fitted.logits(dataset.features[split.test][:, selected]),
            labels_tune=dataset.labels[split.tune],
            labels_calibration=dataset.labels[split.calibration],
            labels_test=dataset.labels[split.test], config=config,
            calibration_uniforms=calibration_uniforms, test_uniforms=test_uniforms,
            seed=seed + 500_000, included_scores=("aps",), included_scalings=("base",),
        )[0]
        metadata = {
            "experiment": config["experiment_name"], "dataset": dataset_name(dataset),
            "seed": seed, "split_id": identifier, "classifier_refit": True,
            "final_calibration_used_once": True, "final_test_used_once": True,
            "code_version": version, **selection,
        }
        completed = {**metadata, **evaluated}
        rows.append(completed)
        if checkpoint_store is not None:
            checkpoint_store.save_frame(
                checkpoint_key, pd.DataFrame([completed])
            )
        print(f"Baseline final: {selection['model']} | {selection['method']} | {len(selected)}", flush=True)
    results = pd.DataFrame(rows)
    reference = results.loc[results["method"] == "all_features", [
        "model", "accuracy", "coverage", "mean_size", "sscv",
        "class_coverage_max_deviation",
    ]].rename(columns={column: f"reference_{column}" for column in [
        "accuracy", "coverage", "mean_size", "sscv", "class_coverage_max_deviation"
    ]})
    results = results.merge(reference, on="model", how="left", validate="many_to_one")
    results["accuracy_loss"] = results["reference_accuracy"] - results["accuracy"]
    results["mean_size_reduction"] = results["reference_mean_size"] - results["mean_size"]
    results["coverage_deviation"] = (results["coverage"] - (1 - results["alpha"])).abs()
    results["conditional_violation"] = results[["sscv", "class_coverage_max_deviation"]].max(axis=1)
    return results


def _baseline_checkpoint_store(
    *,
    config: Mapping[str, Any],
    output_dir: Path,
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
        "stage": "baseline_final_evaluation",
        "experiment_name": config["experiment_name"],
        "seed": seed,
        "split_id": split_identifier,
        "selection_data_id": selection_identifier,
        "config_sha256": experiment_config_sha256(config),
        "code_version": code_version(repository_root),
    }
    store = CheckpointStore(output_dir / "checkpoints" / "baselines", manifest)
    store.initialize(resume=bool(checkpoint_config.get("resume", False)))
    return store


def _summarize(results: pd.DataFrame) -> pd.DataFrame:
    return results.groupby(["model", "method", "target_size"], sort=False).agg(
        repetitions=("repetition", "size"), accuracy=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"), coverage=("coverage", "mean"),
        mean_size=("mean_size", "mean"), mean_size_std=("mean_size", "std"),
        mean_size_reduction=("mean_size_reduction", "mean"), sscv=("sscv", "mean"),
        class_coverage_max_deviation=("class_coverage_max_deviation", "mean"),
    ).reset_index()


def _write_outputs(
    selections: pd.DataFrame, results: pd.DataFrame, summary: pd.DataFrame,
    output_dir: Path, config: Mapping[str, Any], *, random_repetitions: int,
    selection_identifier: str, selection_train_size: int,
    selection_tune_size: int,
) -> None:
    selections.to_csv(output_dir / "baseline_selections.csv", index=False)
    results.to_csv(output_dir / "baseline_final_results.csv", index=False)
    summary.to_csv(output_dir / "baseline_summary.csv", index=False)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
    )
    methods = {"all_features", "random", "mutual_information", "permutation_importance",
               "rfe", "shap", "crfe", "conformal_harm_one_shot", "conformal_harm_recursive"}
    expected_random = (
        selections.loc[
            selections["phase8_primary_selected"].astype(bool)
            & ~selections["method"].isin({"all_features", "random"}),
            ["model", "target_size"],
        ]
        .drop_duplicates().shape[0] * random_repetitions
    )
    checks = {
        "all_required_methods_present": methods.issubset(set(selections["method"])),
        "only_base_aps_evaluated": set(zip(results["scaling"], results["score"])) == {("base", "aps")},
        "matched_non_reference_sizes": bool((selections.loc[selections["method"] != "all_features", "n_features"] == selections.loc[selections["method"] != "all_features", "target_size"]).all()),
        "random_repetition_count": int((selections["method"] == "random").sum()) == expected_random,
        "fresh_thresholds_finite": bool(np.isfinite(results["threshold"]).all()),
        "final_metrics_finite": bool(np.isfinite(results[["accuracy", "coverage", "mean_size", "sscv"]]).all().all()),
        "subsets_frozen_before_calibration": bool(results["subset_frozen_before_final_calibration"].all()),
        "identical_selection_data": bool(
            selections["selection_data_id"].eq(selection_identifier).all()
            and results["selection_data_id"].eq(selection_identifier).all()
        ),
    }
    record = {
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "protocol": {
            "classifier_fit_partition": "train", "ranking_partitions": "train or train+tune",
            "selection_data_id": selection_identifier,
            "selection_train_rows": selection_train_size,
            "selection_tune_rows": selection_tune_size,
            "subset_sizes": (
                "Phase 8 requested removal grid plus primary proposed sizes"
                if config.get("baselines", {}).get("target_removals")
                else "matched to Phase 4 frozen proposed sizes"
            ),
            "final_calibration_access": "once after every subset was frozen",
            "final_test_access": "once after every subset was frozen",
            "evaluation_pipeline": "Base APS only; scaling interaction deferred to Phase 6",
            "crfe": "official multiclass beta rule, fixed Lambda, matched-size stopping",
            "rfe": "standard coefficient RFE with shared logistic selector",
        },
        "observed_selection_rows": len(selections), "observed_final_rows": len(results),
    }
    (output_dir / "baseline_protocol.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    if record["status"] != "PASS":
        raise RuntimeError(f"Phase 5 validation failed: {[key for key, value in checks.items() if not value]}")
