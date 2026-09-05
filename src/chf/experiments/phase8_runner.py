"""Orchestration and aggregation for the approved Phase 8 protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

from chf.data import save_split_artifact

from .baselines import run_required_baselines
from .checkpoints import atomic_write_csv, atomic_write_json, experiment_config_sha256
from .protocol import (
    code_version,
    dataset_from_config,
    experiment_split,
    selection_data_id,
    selection_data_indices,
    split_id,
)
from .real_datasets import _validate_dataset_declaration
from .robustness import (
    derive_seed_config,
    phase8_accuracy_loss_choices,
    phase8_grid_rows,
    run_phase8_grid,
    validate_phase8_spec,
)
from .statistics import (
    holm_adjust,
    paired_effect,
    paired_effect_table,
    rank_stability,
    removal_path_stability,
    two_way_cluster_bootstrap,
)


def load_phase8_spec(path: Path) -> dict[str, Any]:
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_phase8_spec(spec)
    return spec


def phase8_run_plan(
    spec: Mapping[str, Any],
    repository_root: Path,
    *,
    datasets: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Materialize the exact dataset/seed units without loading any dataset."""
    validate_phase8_spec(spec)
    selected_datasets = list(datasets or spec["datasets"])
    unknown = set(selected_datasets).difference(spec["datasets"])
    if unknown:
        raise ValueError(f"unknown Phase 8 datasets: {sorted(unknown)}")
    selected_seeds = [int(value) for value in (seeds or spec["seeds"])]
    if not selected_seeds or not set(selected_seeds).issubset(
        {int(value) for value in spec["seeds"]}
    ):
        raise ValueError("requested seeds must be a non-empty subset of the frozen seeds")
    rows: list[dict[str, Any]] = []
    for dataset_key in selected_datasets:
        config_path = repository_root / str(spec["datasets"][dataset_key]["config"])
        if not config_path.exists():
            raise FileNotFoundError(f"dataset config not found: {config_path}")
        base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for seed in selected_seeds:
            derived = derive_seed_config(
                base_config,
                spec,
                dataset_key=dataset_key,
                seed=seed,
                resume=False,
            )
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "seed": seed,
                    "base_config": str(config_path.relative_to(repository_root)),
                    "experiment_name": derived["experiment_name"],
                    "config_sha256": experiment_config_sha256(derived),
                    "split_unit": str(derived["split"].get("unit", "rows")),
                    "models": json.dumps(sorted(derived["models"])),
                    "output_subdirectory": f"{dataset_key}/seed_{seed}",
                }
            )
    return pd.DataFrame(rows)


def write_phase8_plan(
    spec: Mapping[str, Any],
    repository_root: Path,
    output_dir: Path,
    *,
    datasets: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Write the zero-fit execution plan and provenance manifest."""
    full_plan = phase8_run_plan(spec, repository_root)
    selected_plan = phase8_run_plan(
        spec, repository_root, datasets=datasets, seeds=seeds
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_dir / "phase8_run_plan.csv", full_plan)
    atomic_write_json(
        output_dir / "phase8_plan_manifest.json",
        {
            "status": "PLANNED_NOT_RUN",
            "phase": 8,
            "protocol_version": int(spec["protocol_version"]),
            "code_version": code_version(repository_root),
            "spec_sha256": experiment_config_sha256(spec),
            "dataset_seed_units": len(full_plan),
            "full_grid_cells_per_primary_subset": len(phase8_grid_rows(spec)),
            "benchmark_started": False,
        },
    )
    return selected_plan


def run_phase8_unit(
    *,
    spec: Mapping[str, Any],
    repository_root: Path,
    output_root: Path,
    dataset_key: str,
    seed: int,
    resume: bool,
) -> None:
    """Run one independently resumable dataset/seed unit."""
    config_path = repository_root / str(spec["datasets"][dataset_key]["config"])
    base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = derive_seed_config(
        base_config,
        spec,
        dataset_key=dataset_key,
        seed=seed,
        resume=resume,
    )
    output_dir = output_root / dataset_key / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = dataset_from_config(config, repository_root)
    _validate_dataset_declaration(config, dataset)
    split = experiment_split(config, dataset)
    split_identifier = split_id(split)
    selection_train, selection_tune = selection_data_indices(
        config, split, dataset.labels, seed=seed
    )
    selection_identifier = selection_data_id(selection_train, selection_tune)
    save_split_artifact(
        output_dir / "phase8_split_indices.npz",
        split,
        seed=seed,
        metadata={
            "experiment_name": config["experiment_name"],
            "phase": 8,
            "split_id": split_identifier,
            "selection_data_id": selection_identifier,
        },
    )
    np.savez_compressed(
        output_dir / "phase8_selection_indices.npz",
        train=selection_train,
        tune=selection_tune,
        selection_data_id=np.asarray(selection_identifier),
    )
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    selections, _, _ = run_required_baselines(config, output_dir, repository_root)
    loss_choices = phase8_accuracy_loss_choices(selections, spec, config)
    loss_choices.insert(0, "dataset", dataset_key)
    loss_choices.insert(1, "seed", seed)
    loss_choices.insert(2, "split_id", split_identifier)
    loss_choices.insert(3, "selection_data_id", selection_identifier)
    atomic_write_csv(output_dir / "phase8_accuracy_loss_choices.csv", loss_choices)
    results, subject_results = run_phase8_grid(
        selections=selections,
        config=config,
        spec=spec,
        dataset=dataset,
        split=split,
        split_identifier=split_identifier,
        selection_identifier=selection_identifier,
        output_dir=output_dir,
        repository_root=repository_root,
        resume=resume,
    )
    atomic_write_json(
        output_dir / "phase8_unit_complete.json",
        {
            "status": "PASS",
            "dataset": dataset_key,
            "seed": seed,
            "split_id": split_identifier,
            "selection_data_id": selection_identifier,
            "result_rows": len(results),
            "subject_rows": len(subject_results),
            "config_sha256": experiment_config_sha256(config),
            "code_version": code_version(repository_root),
        },
    )


def run_phase8(
    *,
    spec: Mapping[str, Any],
    repository_root: Path,
    output_dir: Path,
    datasets: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
    resume: bool = False,
) -> None:
    """Run the selected units sequentially; shell-level workers may shard seeds."""
    plan = write_phase8_plan(
        spec,
        repository_root,
        output_dir,
        datasets=datasets,
        seeds=seeds,
    )
    for row in plan.to_dict("records"):
        run_phase8_unit(
            spec=spec,
            repository_root=repository_root,
            output_root=output_dir,
            dataset_key=str(row["dataset_key"]),
            seed=int(row["seed"]),
            resume=resume,
        )
    complete_seed_design = bool(
        plan.groupby("dataset_key")["seed"].nunique().eq(len(spec["seeds"])).all()
    )
    if complete_seed_design:
        aggregate_phase8(spec=spec, output_dir=output_dir, expected_plan=plan)
    else:
        atomic_write_json(
            output_dir / "phase8_partial_run.json",
            {
                "status": "PARTIAL_UNAGGREGATED",
                "completed_requested_units": len(plan),
                "required_paired_seeds": len(spec["seeds"]),
                "message": "Run remaining paired seeds before statistical inference.",
            },
        )


def _read_completed_results(
    output_dir: Path, expected_plan: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results: list[pd.DataFrame] = []
    subjects: list[pd.DataFrame] = []
    missing: list[str] = []
    for row in expected_plan.to_dict("records"):
        unit = output_dir / str(row["output_subdirectory"])
        result_path = unit / "phase8_results.csv"
        completion_path = unit / "phase8_unit_complete.json"
        if not result_path.exists() or not completion_path.exists():
            missing.append(str(row["output_subdirectory"]))
            continue
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("status") != "PASS":
            raise ValueError(f"unit is not complete: {unit}")
        results.append(pd.read_csv(result_path))
        subject_path = unit / "phase8_subject_results.csv"
        if subject_path.exists():
            subjects.append(pd.read_csv(subject_path))
    if missing:
        raise FileNotFoundError(
            "Phase 8 aggregation requires every planned unit; missing: "
            + ", ".join(missing)
        )
    return pd.concat(results, ignore_index=True), (
        pd.concat(subjects, ignore_index=True) if subjects else pd.DataFrame()
    )


def _read_accuracy_loss_choices(
    output_dir: Path, expected_plan: pd.DataFrame
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for row in expected_plan.to_dict("records"):
        path = (
            output_dir
            / str(row["output_subdirectory"])
            / "phase8_accuracy_loss_choices.csv"
        )
        if not path.exists():
            raise FileNotFoundError(f"missing Phase 8 sensitivity choices: {path}")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def _read_rankings(
    output_dir: Path, expected_plan: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_rankings: list[dict[str, Any]] = []
    removal_paths: list[dict[str, Any]] = []
    for plan_row in expected_plan.to_dict("records"):
        dataset_key = str(plan_row["dataset_key"])
        seed = int(plan_row["seed"])
        unit = output_dir / str(plan_row["output_subdirectory"])
        selections = pd.read_csv(unit / "baseline_selections.csv")
        standard = selections.loc[
            ~selections["method"].isin(
                {
                    "all_features",
                    "random",
                    "conformal_harm_one_shot",
                    "conformal_harm_recursive",
                }
            )
            & selections["feature_ranking"].notna()
        ].drop_duplicates(["model", "method"])
        for selection in standard.to_dict("records"):
            ranking = list(reversed(json.loads(selection["feature_ranking"])))
            for rank, feature in enumerate(ranking, start=1):
                full_rankings.append(
                    {
                        "dataset": dataset_key,
                        "model": selection["model"],
                        "method": selection["method"],
                        "seed": seed,
                        "feature": int(feature),
                        "rank": rank,
                    }
                )
        paths = pd.read_csv(
            unit / "proposed_selection" / "progressive_consensus_paths.csv"
        )
        for (model, method), path in paths.groupby(
            ["model", "method"], sort=False
        ):
            universe_size = int(path["n_features"].max())
            removed = path.loc[path["n_removed"].astype(int).gt(0)].sort_values(
                "n_removed"
            )
            for rank, feature in zip(
                removed["n_removed"].astype(int),
                removed["removed_feature"],
                strict=True,
            ):
                removal_paths.append(
                    {
                        "dataset": dataset_key,
                        "model": model,
                        "method": f"conformal_harm_{method}",
                        "seed": seed,
                        "feature": str(feature),
                        "rank": int(rank),
                        "universe_size": universe_size,
                    }
                )
    return pd.DataFrame(full_rankings), pd.DataFrame(removal_paths)


def _holm_within(
    frame: pd.DataFrame, family_columns: list[str]
) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    for source, target in (
        ("sign_flip_pvalue", "holm_sign_flip_pvalue"),
        ("wilcoxon_pvalue", "holm_wilcoxon_pvalue"),
    ):
        frame[target] = frame.groupby(
            family_columns, sort=False, dropna=False
        )[source].transform(lambda values: holm_adjust(values.to_numpy()))
    return frame


def _matched_standard_effects(
    results: pd.DataFrame, spec: Mapping[str, Any]
) -> pd.DataFrame:
    primary_alpha = float(spec["selection"]["primary_alpha"])
    primary = results.loc[
        results["alpha"].eq(primary_alpha)
        & results["scaling"].eq("base")
        & results["score"].eq("aps")
        & results["phase8_primary_selected"].astype(bool)
    ].copy()
    proposed_methods = {
        "conformal_harm_one_shot",
        "conformal_harm_recursive",
    }
    excluded = {"all_features", "random", *proposed_methods}
    proposed = primary.loc[primary["method"].isin(proposed_methods)]
    standards = primary.loc[~primary["method"].isin(excluded)]
    paired = proposed.merge(
        standards,
        on=["dataset", "model", "seed", "target_size"],
        suffixes=("_proposed", "_standard"),
        validate="many_to_many",
    )
    rows: list[dict[str, Any]] = []
    for keys, group in paired.groupby(
        ["dataset", "model", "method_proposed", "method_standard"],
        sort=False,
    ):
        if group["seed"].duplicated().any():
            raise ValueError("matched standard comparison is not unique by seed")
        for metric, differences, direction in (
            (
                "mean_size",
                group["mean_size_standard"] - group["mean_size_proposed"],
                "proposed_smaller",
            ),
            (
                "accuracy",
                group["accuracy_proposed"] - group["accuracy_standard"],
                "proposed_higher",
            ),
        ):
            rows.append(
                {
                    "dataset": keys[0],
                    "model": keys[1],
                    "method": keys[2],
                    "reference_method": keys[3],
                    "metric": metric,
                    "positive_direction": direction,
                    "target_size_rule": "matched_within_seed",
                    "target_size_min": int(group["target_size"].min()),
                    "target_size_max": int(group["target_size"].max()),
                    **paired_effect(
                        differences,
                        float(spec["statistics"]["confidence_level"]),
                    ),
                }
            )
    return _holm_within(
        pd.DataFrame(rows), ["dataset", "model", "metric"]
    )


def _sensitivity_effects(
    results: pd.DataFrame,
    choices: pd.DataFrame,
    spec: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha = float(spec["selection"]["primary_alpha"])
    base = results.loc[
        results["alpha"].eq(alpha)
        & results["scaling"].eq("base")
        & results["score"].eq("aps")
    ].copy()
    chosen = choices.merge(
        base,
        on=["dataset", "seed", "model", "method", "selected_indices"],
        how="left",
        validate="many_to_one",
        suffixes=("_choice", ""),
    )
    if chosen["mean_size_reduction_vs_all"].isna().any():
        raise ValueError("an accuracy-loss sensitivity choice lacks a final result")

    def summarize(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for keys, group in frame.groupby(groups, sort=False, dropna=False):
            key_values = keys if isinstance(keys, tuple) else (keys,)
            for metric in (
                "mean_size_reduction_vs_all",
                "accuracy_loss_vs_all",
                "coverage_delta_vs_all",
            ):
                values = group[metric].to_numpy(dtype=float)
                if metric == "accuracy_loss_vs_all":
                    values = -values
                rows.append(
                    {
                        **dict(zip(groups, key_values, strict=True)),
                        "metric": metric,
                        "positive_direction": (
                            "smaller_accuracy_loss"
                            if metric == "accuracy_loss_vs_all"
                            else "as_named"
                        ),
                        **paired_effect(
                            values,
                            float(spec["statistics"]["confidence_level"]),
                        ),
                    }
                )
        return _holm_within(pd.DataFrame(rows), ["dataset", "model", "metric"])

    loss_effects = summarize(
        chosen, ["dataset", "model", "method", "accuracy_loss_limit"]
    )
    proposed_path = base.loc[
        base["method"].isin(
            {"conformal_harm_one_shot", "conformal_harm_recursive"}
        )
        & base["tuning_n_removed"].notna()
    ].copy()
    proposed_path["n_removed"] = proposed_path["tuning_n_removed"].astype(int)
    subset_effects = summarize(
        proposed_path, ["dataset", "model", "method", "n_removed"]
    )
    return loss_effects, subset_effects


def _rank_stability_tables(
    output_dir: Path, expected_plan: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full, paths = _read_rankings(output_dir, expected_plan)
    frames: list[pd.DataFrame] = []
    if not full.empty:
        standard = rank_stability(full)
        standard.insert(3, "stability_type", "complete_harmfulness_ranking")
        frames.append(standard)
    if not paths.empty:
        proposed = removal_path_stability(paths)
        proposed.insert(3, "stability_type", "progressive_removal_path")
        frames.append(proposed)
    pairs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if pairs.empty:
        return pairs, pd.DataFrame()
    summary = pairs.groupby(
        ["dataset", "model", "method", "stability_type", "top_k"],
        dropna=False,
        sort=False,
    ).agg(
        seed_pairs=("left_seed", "size"),
        spearman_mean=("spearman", "mean"),
        spearman_min=("spearman", "min"),
        jaccard_mean=("jaccard", "mean"),
        jaccard_min=("jaccard", "min"),
        kuncheva_mean=("kuncheva", "mean"),
        kuncheva_min=("kuncheva", "min"),
    ).reset_index()
    return pairs, summary


def _primary_proposed(results: pd.DataFrame, spec: Mapping[str, Any]) -> pd.DataFrame:
    primary_alpha = float(spec["selection"]["primary_alpha"])
    return results.loc[
        results["method"].isin(
            {"all_features", "conformal_harm_one_shot", "conformal_harm_recursive"}
        )
        & (
            results["method"].eq("all_features")
            | results["phase8_primary_selected"].astype(bool)
        )
        & results["alpha"].eq(primary_alpha)
        & results["scaling"].eq("base")
        & results["score"].eq("aps")
    ].copy()


def _paired_subject_bootstrap_table(
    subject_results: pd.DataFrame, spec: Mapping[str, Any]
) -> pd.DataFrame:
    if subject_results.empty:
        return pd.DataFrame()
    primary = _primary_proposed(subject_results, spec)
    primary = primary.loc[primary["method"] != "all_features"]
    rows: list[dict[str, Any]] = []
    for keys, group in primary.groupby(["dataset", "model", "method"], sort=False):
        for metric in ("mean_size_reduction_vs_all", "coverage_difference_vs_all"):
            result = two_way_cluster_bootstrap(
                group,
                value_column=metric,
                repetitions=int(spec["statistics"]["subject_bootstrap_repetitions"]),
                confidence_level=float(spec["statistics"]["confidence_level"]),
                random_seed=int(spec["statistics"]["bootstrap_seed"]),
            )
            rows.append(
                {
                    "dataset": keys[0],
                    "model": keys[1],
                    "method": keys[2],
                    "metric": metric,
                    **result,
                }
            )
    return pd.DataFrame(rows)


def _precision_escalation(
    effects: pd.DataFrame, primary: pd.DataFrame
) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    all_sizes = primary.loc[primary["method"] == "all_features"].groupby(
        "dataset"
    )["mean_size"].mean()
    for dataset, rows in effects.groupby("dataset", sort=False):
        epsilon = max(0.01, 0.01 * float(all_sizes[dataset]))
        half_width = (rows["ci_upper"] - rows["ci_lower"]) / 2.0
        material = half_width > epsilon
        decisions[str(dataset)] = {
            "smallest_practical_effect": epsilon,
            "material_variance_contrasts": int(material.sum()),
            "evaluated_contrasts": int(len(rows)),
            "extend_to_20_seeds": bool(material.sum() >= 2),
            "rule": "extend when at least two co-primary contrasts exceed the precision target",
        }
    return decisions


def aggregate_phase8(
    *,
    spec: Mapping[str, Any],
    output_dir: Path,
    expected_plan: pd.DataFrame,
) -> None:
    """Aggregate only after every planned paired seed has completed."""
    seed_counts = expected_plan.groupby("dataset_key")["seed"].nunique()
    if not seed_counts.eq(len(spec["seeds"])).all():
        raise ValueError(
            "Phase 8 inference requires all ten paired seeds for every "
            "requested dataset"
        )
    results, subjects = _read_completed_results(output_dir, expected_plan)
    choices = _read_accuracy_loss_choices(output_dir, expected_plan)
    primary = _primary_proposed(results, spec)
    effects = paired_effect_table(
        primary,
        group_columns=("dataset", "model"),
        pair_column="seed",
        method_column="method",
        reference_method="all_features",
        metric="mean_size",
        higher_is_better=False,
        confidence_level=float(spec["statistics"]["confidence_level"]),
    )
    accuracy = paired_effect_table(
        primary,
        group_columns=("dataset", "model"),
        pair_column="seed",
        method_column="method",
        reference_method="all_features",
        metric="accuracy",
        higher_is_better=True,
        confidence_level=float(spec["statistics"]["confidence_level"]),
    )
    standard_effects = _matched_standard_effects(results, spec)
    loss_effects, subset_effects = _sensitivity_effects(results, choices, spec)
    rank_pairs, rank_summary = _rank_stability_tables(output_dir, expected_plan)
    subject_effects = _paired_subject_bootstrap_table(subjects, spec)
    atomic_write_csv(output_dir / "phase8_all_results.csv", results)
    if not subjects.empty:
        atomic_write_csv(output_dir / "phase8_all_subject_results.csv", subjects)
    atomic_write_csv(output_dir / "phase8_paired_size_effects.csv", effects)
    atomic_write_csv(output_dir / "phase8_paired_accuracy_effects.csv", accuracy)
    atomic_write_csv(
        output_dir / "phase8_matched_standard_effects.csv", standard_effects
    )
    atomic_write_csv(
        output_dir / "phase8_accuracy_loss_sensitivity_effects.csv", loss_effects
    )
    atomic_write_csv(
        output_dir / "phase8_subset_size_sensitivity_effects.csv", subset_effects
    )
    atomic_write_csv(output_dir / "phase8_rank_stability_pairs.csv", rank_pairs)
    atomic_write_csv(output_dir / "phase8_rank_stability_summary.csv", rank_summary)
    if not subject_effects.empty:
        atomic_write_csv(
            output_dir / "phase8_har_subject_effects.csv", subject_effects
        )
    atomic_write_json(
        output_dir / "phase8_precision_decision.json",
        _precision_escalation(effects, primary),
    )
    atomic_write_json(
        output_dir / "phase8_protocol.json",
        {
            "status": "PASS",
            "phase": 8,
            "protocol_version": int(spec["protocol_version"]),
            "completed_units": len(expected_plan),
            "paired_seeds_per_dataset": int(
                expected_plan.groupby("dataset_key")["seed"].nunique().min()
            ),
            "full_grid_cells": len(phase8_grid_rows(spec)),
            "final_result_rows": len(results),
            "har_subject_rows": len(subjects),
            "inference": {
                "paired_interval": "Student t interval across paired seeds",
                "paired_test": "exact two-sided sign-flip",
                "sensitivity_test": "Wilcoxon signed-rank",
                "multiplicity": "Holm within each reported family",
                "har_uncertainty": "two-way seed/subject cluster bootstrap",
                "rank_stability": (
                    "pairwise Spearman/Jaccard/Kuncheva for complete standard "
                    "rankings; Jaccard/Kuncheva for progressive removal paths"
                ),
                "matched_standard_rule": (
                    "compare against every preregistered deterministic standard "
                    "at the exact within-seed subset size; Holm correction avoids "
                    "test-set selection of a single oracle comparator"
                ),
            },
        },
    )
