"""Aggregate final Phase 8 inference after the preregistered 10->20 extension.

The scientific rows are produced by the original ``e4b3645`` Phase 8 code.
This control layer only combines the initial 10-seed batch with the second
10-seed batch for the two datasets whose pre-registered CI-width rule fired.
Covertype remains at ten seeds.  Statistical definitions are unchanged.

For 20 paired seeds the exact sign-flip test has 2^20 assignments.  The
original implementation materializes all sign vectors at once; here we replace
that *implementation* with an exactly equivalent meet-in-the-middle counter so
that the pre-registered exact test remains practical without changing the test.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import chf.experiments.statistics as statistics
from chf.experiments.checkpoints import atomic_write_csv, atomic_write_json
from chf.experiments.phase8_runner import (
    _matched_standard_effects,
    _paired_subject_bootstrap_table,
    _primary_proposed,
    _rank_stability_tables,
    _read_accuracy_loss_choices,
    _read_completed_results,
    _sensitivity_effects,
    load_phase8_spec,
    phase8_run_plan,
)
from chf.experiments.statistics import paired_effect_table
from chf.experiments.protocol import code_version


EXPECTED_SCIENTIFIC_VERSION = "e4b3645"
INITIAL_SEEDS = tuple(range(43, 53))
EXTENSION_SEEDS = tuple(range(53, 63))
EXTENDED_DATASETS = {"dry_bean", "human_activity_recognition"}
ALL_DATASETS = {"dry_bean", "covertype", "human_activity_recognition"}


def exact_sign_flip_pvalue_mitm(differences: Iterable[float]) -> float:
    """Exact two-sided sign-flip p-value using meet-in-the-middle enumeration.

    This counts exactly the same 2^n sign assignments as the frozen Phase 8
    implementation but never materializes the 2^20 x 20 sign matrix.
    """
    values = np.asarray(list(differences), dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        raise ValueError("at least one finite paired difference is required")
    if n > 20:
        raise ValueError("exact sign-flip enumeration is capped at 20 pairs")

    observed = abs(float(values.mean()))
    threshold = n * (observed - 1e-15)
    if threshold <= 0:
        return 1.0

    split = n // 2
    left_values = values[:split]
    right_values = values[split:]

    def signed_sums(part: np.ndarray) -> np.ndarray:
        m = len(part)
        if m == 0:
            return np.asarray([0.0])
        ids = np.arange(1 << m, dtype=np.uint32)[:, None]
        bits = ((ids >> np.arange(m, dtype=np.uint32)) & 1).astype(np.int8)
        signs = bits * 2 - 1
        return signs @ part

    left = signed_sums(left_values)
    right = np.sort(signed_sums(right_values))
    total_extreme = 0
    for value in left:
        # |value + right| >= threshold, i.e. either tail.  The tails are
        # disjoint because threshold > 0.
        left_cut = -threshold - value
        right_cut = threshold - value
        total_extreme += int(np.searchsorted(right, left_cut, side="right"))
        total_extreme += int(len(right) - np.searchsorted(right, right_cut, side="left"))
    return float(total_extreme / float(1 << n))


def _install_exact_test_implementation() -> None:
    statistics.exact_sign_flip_pvalue = exact_sign_flip_pvalue_mitm


def _combined_plan(
    *,
    initial_spec: dict[str, Any],
    extension_spec: dict[str, Any],
    repository_root: Path,
) -> pd.DataFrame:
    initial = phase8_run_plan(
        initial_spec,
        repository_root,
        datasets=sorted(ALL_DATASETS),
        seeds=INITIAL_SEEDS,
    )
    extension = phase8_run_plan(
        extension_spec,
        repository_root,
        datasets=sorted(EXTENDED_DATASETS),
        seeds=EXTENSION_SEEDS,
    )
    plan = pd.concat([initial, extension], ignore_index=True)
    if plan.duplicated(["dataset_key", "seed"]).any():
        raise ValueError("combined precision-extension plan contains duplicate units")
    counts = plan.groupby("dataset_key")["seed"].nunique().to_dict()
    expected = {
        "covertype": 10,
        "dry_bean": 20,
        "human_activity_recognition": 20,
    }
    if counts != expected:
        raise ValueError(f"unexpected final seed counts: {counts}")
    return plan


def _audit_units(output_dir: Path, plan: pd.DataFrame) -> dict[str, Any]:
    extension_manifests = 0
    recovery_manifests = 0
    for row in plan.to_dict("records"):
        unit = output_dir / str(row["output_subdirectory"])
        completion_path = unit / "phase8_unit_complete.json"
        if not completion_path.exists():
            raise FileNotFoundError(f"missing completion record: {unit}")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("status") != "PASS":
            raise ValueError(f"unit is not PASS: {unit}")
        if str(completion.get("dataset")) != str(row["dataset_key"]):
            raise ValueError(f"dataset provenance mismatch: {unit}")
        if int(completion.get("seed")) != int(row["seed"]):
            raise ValueError(f"seed provenance mismatch: {unit}")
        if str(completion.get("config_sha256")) != str(row["config_sha256"]):
            raise ValueError(f"config provenance mismatch: {unit}")
        if str(completion.get("code_version")) != EXPECTED_SCIENTIFIC_VERSION:
            raise ValueError(f"scientific code provenance mismatch: {unit}")

        subject_path = unit / "phase8_subject_results.csv"
        if row["dataset_key"] == "human_activity_recognition" and not subject_path.exists():
            raise FileNotFoundError(f"HAR unit lacks subject results: {unit}")

        extension_path = unit / "phase8_precision_extension_manifest.json"
        if int(row["seed"]) in EXTENSION_SEEDS:
            if row["dataset_key"] not in EXTENDED_DATASETS:
                raise ValueError("unexpected extension unit for a stopped dataset")
            if not extension_path.exists():
                raise FileNotFoundError(f"extension unit lacks extension manifest: {unit}")
            record = json.loads(extension_path.read_text(encoding="utf-8"))
            if record.get("status") != "PASS":
                raise ValueError(f"extension manifest is not PASS: {unit}")
            extension_manifests += 1

        recovery_path = unit / "phase8_recovery_manifest.json"
        if recovery_path.exists():
            recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
            if recovery.get("status") != "PASS":
                raise ValueError(f"recovery manifest is not PASS: {unit}")
            recovery_manifests += 1

    if extension_manifests != 20:
        raise ValueError(f"expected 20 extension manifests, found {extension_manifests}")
    if recovery_manifests != 12:
        raise ValueError(f"expected 12 recovered initial units, found {recovery_manifests}")
    return {
        "audited_units": int(len(plan)),
        "extension_units": extension_manifests,
        "recovered_initial_units": recovery_manifests,
    }


def _final_precision_status(
    *,
    effects: pd.DataFrame,
    primary: pd.DataFrame,
    plan: pd.DataFrame,
    initial_decision: dict[str, Any],
) -> dict[str, Any]:
    all_sizes = primary.loc[primary["method"].eq("all_features")].groupby("dataset")[
        "mean_size"
    ].mean()
    seed_counts = plan.groupby("dataset_key")["seed"].nunique().to_dict()
    datasets: dict[str, Any] = {}
    for dataset, rows in effects.groupby("dataset", sort=False):
        epsilon = max(0.01, 0.01 * float(all_sizes[dataset]))
        half_width = (rows["ci_upper"] - rows["ci_lower"]) / 2.0
        material = half_width > epsilon
        final_n = int(seed_counts[dataset])
        triggered_initially = bool(
            initial_decision.get(dataset, {}).get("extend_to_20_seeds", False)
        )
        datasets[str(dataset)] = {
            "initial_10_seed_extension_triggered": triggered_initially,
            "final_seed_count": final_n,
            "smallest_practical_effect": epsilon,
            "evaluated_contrasts": int(len(rows)),
            "material_variance_contrasts_at_final_n": int(material.sum()),
            "precision_target_met_at_final_n": bool(material.sum() < 2),
            "maximum_seeds_reached": bool(final_n == 20),
            "further_extension_permitted": False,
            "rule": (
                "initially extend when at least two co-primary contrasts exceed "
                "the precision target; stop no later than 20 seeds"
            ),
        }
    return {
        "status": "FINAL_STOP",
        "maximum_seeds": 20,
        "datasets": datasets,
        "extension_seeds": list(EXTENSION_SEEDS),
        "note": (
            "Dry Bean and HAR stop at the preregistered maximum of 20 seeds even "
            "if the final precision target remains unmet; Covertype stopped at 10."
        ),
    }


def aggregate_final(
    *,
    initial_spec_path: Path,
    extension_spec_path: Path,
    initial_precision_path: Path,
    repository_root: Path,
    output_dir: Path,
) -> None:
    if code_version(repository_root) != EXPECTED_SCIENTIFIC_VERSION:
        raise ValueError("final aggregation must use the original clean scientific checkout")

    initial_spec = load_phase8_spec(initial_spec_path)
    extension_spec = load_phase8_spec(extension_spec_path)
    if tuple(int(x) for x in initial_spec["seeds"]) != INITIAL_SEEDS:
        raise ValueError("unexpected initial Phase 8 seeds")
    if tuple(int(x) for x in extension_spec["seeds"]) != EXTENSION_SEEDS:
        raise ValueError("unexpected extension Phase 8 seeds")
    for key in ("datasets", "selection", "conformal_grid", "statistics"):
        if initial_spec[key] != extension_spec[key]:
            raise ValueError(f"extension changed frozen scientific field: {key}")

    plan = _combined_plan(
        initial_spec=initial_spec,
        extension_spec=extension_spec,
        repository_root=repository_root,
    )
    audit = _audit_units(output_dir, plan)
    _install_exact_test_implementation()

    results, subjects = _read_completed_results(output_dir, plan)
    choices = _read_accuracy_loss_choices(output_dir, plan)
    primary = _primary_proposed(results, initial_spec)

    effects = paired_effect_table(
        primary,
        group_columns=("dataset", "model"),
        pair_column="seed",
        method_column="method",
        reference_method="all_features",
        metric="mean_size",
        higher_is_better=False,
        confidence_level=float(initial_spec["statistics"]["confidence_level"]),
    )
    accuracy = paired_effect_table(
        primary,
        group_columns=("dataset", "model"),
        pair_column="seed",
        method_column="method",
        reference_method="all_features",
        metric="accuracy",
        higher_is_better=True,
        confidence_level=float(initial_spec["statistics"]["confidence_level"]),
    )
    standard_effects = _matched_standard_effects(results, initial_spec)
    loss_effects, subset_effects = _sensitivity_effects(results, choices, initial_spec)
    rank_pairs, rank_summary = _rank_stability_tables(output_dir, plan)
    subject_effects = _paired_subject_bootstrap_table(subjects, initial_spec)

    atomic_write_csv(output_dir / "phase8_all_results.csv", results)
    if not subjects.empty:
        atomic_write_csv(output_dir / "phase8_all_subject_results.csv", subjects)
    atomic_write_csv(output_dir / "phase8_paired_size_effects.csv", effects)
    atomic_write_csv(output_dir / "phase8_paired_accuracy_effects.csv", accuracy)
    atomic_write_csv(output_dir / "phase8_matched_standard_effects.csv", standard_effects)
    atomic_write_csv(
        output_dir / "phase8_accuracy_loss_sensitivity_effects.csv", loss_effects
    )
    atomic_write_csv(
        output_dir / "phase8_subset_size_sensitivity_effects.csv", subset_effects
    )
    atomic_write_csv(output_dir / "phase8_rank_stability_pairs.csv", rank_pairs)
    atomic_write_csv(output_dir / "phase8_rank_stability_summary.csv", rank_summary)
    if not subject_effects.empty:
        atomic_write_csv(output_dir / "phase8_har_subject_effects.csv", subject_effects)

    initial_decision = json.loads(initial_precision_path.read_text(encoding="utf-8"))
    atomic_write_json(
        output_dir / "phase8_precision_decision_initial10.json", initial_decision
    )
    final_precision = _final_precision_status(
        effects=effects,
        primary=primary,
        plan=plan,
        initial_decision=initial_decision,
    )
    atomic_write_json(output_dir / "phase8_precision_decision.json", final_precision)

    seed_counts = {
        str(k): int(v)
        for k, v in plan.groupby("dataset_key")["seed"].nunique().to_dict().items()
    }
    atomic_write_json(
        output_dir / "phase8_protocol.json",
        {
            "status": "PASS_PRECISION_EXTENSION_COMPLETE",
            "phase": 8,
            "protocol_version": int(initial_spec["protocol_version"]),
            "scientific_code_version": EXPECTED_SCIENTIFIC_VERSION,
            "completed_units": int(len(plan)),
            "paired_seeds_by_dataset": seed_counts,
            "extended_datasets": sorted(EXTENDED_DATASETS),
            "stopped_at_10_seeds": ["covertype"],
            "extension_seeds": list(EXTENSION_SEEDS),
            "scientific_grid_changed": False,
            "final_result_rows": int(len(results)),
            "har_subject_rows": int(len(subjects)),
            "exact_sign_flip_implementation": (
                "meet-in-the-middle exact enumeration; same 2^n randomization test"
            ),
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
            },
        },
    )
    atomic_write_json(
        output_dir / "phase8_precision_extension_aggregate.json",
        {
            "status": "PASS",
            **audit,
            "initial_completed_run_id": "34239393188",
            "scientific_code_version": EXPECTED_SCIENTIFIC_VERSION,
            "extension_control_commit": os.environ.get(
                "PHASE8_EXTENSION_CONTROL_SHA", "unknown"
            ),
            "scientific_grid_changed": False,
            "numerical_failure_handling": "retained_and_flagged_not_repaired",
            "precision_extension_protocol": "PHASE8_PRECISION_EXTENSION.md",
        },
    )
    print(
        "Final Phase 8 precision-extension aggregate complete: "
        f"seed_counts={seed_counts}, rows={len(results)}, HAR_subject_rows={len(subjects)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-config", type=Path, required=True)
    parser.add_argument("--extension-config", type=Path, required=True)
    parser.add_argument("--initial-precision-json", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    aggregate_final(
        initial_spec_path=args.initial_config.resolve(),
        extension_spec_path=args.extension_config.resolve(),
        initial_precision_path=args.initial_precision_json.resolve(),
        repository_root=args.repository_root.resolve(),
        output_dir=args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
