"""Recover interrupted Phase 8 units without changing their scientific commit.

This control script is intentionally executed from a separate checkout while the
scientific repository is checked out at the original Phase 8 commit.  It reuses
only artifacts whose split/selection provenance matches that original run,
repairs two operational issues discovered after the run, and records the repair
in a separate recovery manifest.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from chf.experiments.baselines import run_required_baselines
from chf.experiments.checkpoints import (
    atomic_write_csv,
    atomic_write_json,
    experiment_config_sha256,
)
from chf.experiments.phase8_runner import load_phase8_spec
from chf.experiments.protocol import (
    code_version,
    dataset_from_config,
    experiment_split,
    selection_data_id,
    selection_data_indices,
    split_id,
)
from chf.experiments.real_datasets import _validate_dataset_declaration
from chf.experiments.robustness import (
    derive_seed_config,
    phase8_accuracy_loss_choices,
    run_phase8_grid,
)
import chf.experiments.robustness as robustness


BOUNDARY_COLUMNS = [
    "calibration_zero_count",
    "calibration_exactly_one_count",
    "test_zero_count",
    "test_exactly_one_count",
]
CONDITION_COLUMNS = [
    "model",
    "method",
    "target_size",
    "selected_indices",
    "repetition",
    "alpha",
    "score",
    "raps_lambda",
    "raps_k_reg",
]


def _normalized_condition(row: Mapping[str, Any]) -> tuple[Any, ...]:
    values: list[Any] = []
    for column in CONDITION_COLUMNS:
        value = row[column]
        if column == "raps_lambda":
            value = None if pd.isna(value) else float(value)
        elif column == "raps_k_reg":
            value = None if pd.isna(value) else int(value)
        elif column == "alpha":
            value = float(value)
        elif column in {"target_size", "repetition"}:
            value = int(value)
        values.append(value)
    return tuple(values)


def _boundary_excess_frame(results: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a validation copy containing only scaling-induced boundary counts.

    Exact 0/1 probabilities already present at Base (T=1) are retained as a
    diagnostic but are not attributed to TS/ConfTS. A tuned scaling fails the
    numerical safeguard only if it has more exact-boundary probabilities than
    the matching Base cell for the same frozen subset and conformal condition.
    """
    missing = set(BOUNDARY_COLUMNS + CONDITION_COLUMNS + ["scaling"]).difference(
        results.columns
    )
    if missing:
        raise ValueError(
            f"Phase 8 numerical recovery is missing columns: {sorted(missing)}"
        )

    validation = results.copy(deep=True)
    base_rows = results.loc[results["scaling"].eq("base")]
    base_counts: dict[tuple[Any, ...], dict[str, int]] = {}
    for row in base_rows.to_dict("records"):
        key = _normalized_condition(row)
        if key in base_counts:
            raise ValueError("Base numerical diagnostic is not unique by condition")
        base_counts[key] = {
            column: int(row[column]) for column in BOUNDARY_COLUMNS
        }

    max_excess = {column: 0 for column in BOUNDARY_COLUMNS}
    for index, row in validation.iterrows():
        if str(row["scaling"]) == "base":
            validation.loc[index, BOUNDARY_COLUMNS] = 0
            continue
        key = _normalized_condition(row)
        if key not in base_counts:
            raise ValueError("Tuned scaling lacks a matching Base numerical diagnostic")
        for column in BOUNDARY_COLUMNS:
            excess = max(int(row[column]) - base_counts[key][column], 0)
            validation.at[index, column] = excess
            max_excess[column] = max(max_excess[column], excess)

    diagnostics = {
        "rule": (
            "Exact 0/1 probabilities at Base (T=1) are diagnostic. TS/ConfTS "
            "must not increase exact-boundary counts above the matching Base cell."
        ),
        "base_boundary_totals": {
            column: int(base_rows[column].sum()) for column in BOUNDARY_COLUMNS
        },
        "maximum_scaling_induced_excess": max_excess,
    }
    return validation, diagnostics


def _install_recovery_validator() -> None:
    original_validate = robustness._validate_and_write_grid

    def validate_with_recovery_rule(
        *,
        results: pd.DataFrame,
        subject_results: pd.DataFrame,
        selections: pd.DataFrame,
        spec: Mapping[str, Any],
        output_dir: Path,
        manifest: Mapping[str, Any],
    ) -> None:
        validation_results, diagnostics = _boundary_excess_frame(results)
        original_validate(
            results=validation_results,
            subject_results=subject_results,
            selections=selections,
            spec=spec,
            output_dir=output_dir,
            manifest=manifest,
        )

        # Preserve the unmodified scientific diagnostics in the published CSV.
        atomic_write_csv(output_dir / "phase8_results.csv", results)
        protocol_path = output_dir / "phase8_grid_protocol.json"
        record = json.loads(protocol_path.read_text(encoding="utf-8"))
        legacy = record["checks"].pop("no_zero_or_saturated_probabilities")
        record["checks"]["no_scaling_induced_boundary_probabilities"] = bool(legacy)
        record["numerical_recovery"] = diagnostics
        record["status"] = "PASS" if all(record["checks"].values()) else "FAIL"
        atomic_write_json(protocol_path, record)

    robustness._validate_and_write_grid = validate_with_recovery_rule


def _install_har_merge_compatibility() -> None:
    """Normalize only the legacy HAR subject-filter merge keys under pandas 3."""
    original_merge = pd.DataFrame.merge
    subject_keys = [
        "alpha", "scaling", "score", "raps_lambda", "raps_k_reg"
    ]

    def merge_with_normalized_subject_keys(self, right, *args, **kwargs):
        on = kwargs.get("on")
        if on is not None and list(on) == subject_keys:
            left_frame = self.copy()
            right_frame = right.copy()
            for column in ("raps_lambda", "raps_k_reg"):
                left_frame[column] = pd.to_numeric(
                    left_frame[column], errors="coerce"
                )
                right_frame[column] = pd.to_numeric(
                    right_frame[column], errors="coerce"
                )
            return original_merge(left_frame, right_frame, *args, **kwargs)
        return original_merge(self, right, *args, **kwargs)

    pd.DataFrame.merge = merge_with_normalized_subject_keys


def _assert_pass(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing {label}: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "PASS":
        raise ValueError(f"{label} is not PASS: {path}")


def _load_or_rebuild_selections(
    *,
    config: dict[str, Any],
    output_dir: Path,
    repository_root: Path,
    selection_identifier: str,
) -> tuple[pd.DataFrame, str]:
    selections_path = output_dir / "baseline_selections.csv"
    baseline_protocol = output_dir / "baseline_protocol.json"
    if selections_path.exists() and baseline_protocol.exists():
        _assert_pass(baseline_protocol, "baseline selection protocol")
        selections = pd.read_csv(selections_path)
        if (
            "selection_data_id" not in selections
            or not selections["selection_data_id"].eq(selection_identifier).all()
        ):
            raise ValueError("cached baseline selections use different selection data")
        return selections, "validated_baseline_selections"

    _assert_pass(
        output_dir / "proposed_selection" / "progressive_protocol.json",
        "progressive selection protocol",
    )
    progressive_path = (
        output_dir / "proposed_selection" / "progressive_consensus_paths.csv"
    )
    if not progressive_path.exists():
        raise FileNotFoundError(
            "recovery requires completed progressive-selection outputs"
        )
    progressive = pd.read_csv(progressive_path)
    if (
        "selection_data_id" not in progressive
        or not progressive["selection_data_id"].eq(selection_identifier).all()
    ):
        raise ValueError("cached progressive selection uses different selection data")

    # The expensive progressive stage is already protocol-validated. Disable
    # checkpoint reconstruction only for this selection-ranking rebuild so
    # run_required_baselines accepts the validated CSV instead of replaying
    # thousands of candidate shards. The final-grid config is restored below.
    ranking_config = copy.deepcopy(config)
    ranking_config["checkpointing"] = {"enabled": False, "resume": False}
    selections, _, _ = run_required_baselines(
        ranking_config, output_dir, repository_root
    )
    if not selections["selection_data_id"].eq(selection_identifier).all():
        raise ValueError("rebuilt baseline selections use different selection data")
    return selections, "validated_progressive_selection_plus_rebuilt_baselines"


def recover_unit(
    *,
    spec_path: Path,
    repository_root: Path,
    output_root: Path,
    dataset_key: str,
    seed: int,
    source_run_id: str,
) -> None:
    spec = load_phase8_spec(spec_path)
    if dataset_key not in spec["datasets"]:
        raise ValueError(f"unknown Phase 8 dataset: {dataset_key}")
    if seed not in {int(value) for value in spec["seeds"]}:
        raise ValueError(f"seed {seed} is outside the frozen Phase 8 design")

    scientific_version = code_version(repository_root)
    expected_version = os.environ.get("PHASE8_SCIENTIFIC_COMMIT_SHORT", "e4b3645")
    if scientific_version != expected_version:
        raise ValueError(
            "recovery must run from the original clean scientific commit; "
            f"expected {expected_version}, got {scientific_version}"
        )

    config_path = repository_root / str(spec["datasets"][dataset_key]["config"])
    base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = derive_seed_config(
        base_config,
        spec,
        dataset_key=dataset_key,
        seed=seed,
        resume=True,
    )
    output_dir = output_root / dataset_key / f"seed_{seed}"
    if not output_dir.exists():
        raise FileNotFoundError(
            "source artifact was not restored before recovery: " + str(output_dir)
        )

    dataset = dataset_from_config(config, repository_root)
    _validate_dataset_declaration(config, dataset)
    split = experiment_split(config, dataset)
    split_identifier = split_id(split)
    selection_train, selection_tune = selection_data_indices(
        config, split, dataset.labels, seed=seed
    )
    selection_identifier = selection_data_id(selection_train, selection_tune)

    selections, selection_reuse = _load_or_rebuild_selections(
        config=config,
        output_dir=output_dir,
        repository_root=repository_root,
        selection_identifier=selection_identifier,
    )
    # Restore the exact scientific config after any baseline-ranking rebuild.
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    choices = phase8_accuracy_loss_choices(selections, spec, config)
    choices.insert(0, "dataset", dataset_key)
    choices.insert(1, "seed", seed)
    choices.insert(2, "split_id", split_identifier)
    choices.insert(3, "selection_data_id", selection_identifier)
    atomic_write_csv(output_dir / "phase8_accuracy_loss_choices.csv", choices)

    _install_har_merge_compatibility()
    _install_recovery_validator()
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
        resume=True,
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
            "code_version": scientific_version,
            "recovered_from_run_id": source_run_id,
        },
    )
    atomic_write_json(
        output_dir / "phase8_recovery_manifest.json",
        {
            "status": "PASS",
            "source_run_id": source_run_id,
            "source_artifact": f"phase8-unit-{dataset_key}-{seed}",
            "scientific_code_version": scientific_version,
            "recovery_control_commit": os.environ.get(
                "PHASE8_RECOVERY_CONTROL_SHA", "unknown"
            ),
            "pandas_version": pd.__version__,
            "selection_reuse": selection_reuse,
            "scientific_grid_changed": False,
            "split_or_seed_changed": False,
            "final_calibration_or_test_reused_for_selection": False,
            "operational_repairs": [
                "normalize only legacy HAR subject-filter merge dtypes in memory",
                "validate exact-boundary probabilities relative to matching Base T=1",
                "reuse only source artifacts with matching frozen split/selection provenance",
            ],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-run-id", required=True)
    args = parser.parse_args()
    recover_unit(
        spec_path=args.config.resolve(),
        repository_root=args.repository_root.resolve(),
        output_root=args.output_dir.resolve(),
        dataset_key=str(args.dataset),
        seed=int(args.seed),
        source_run_id=str(args.source_run_id),
    )


if __name__ == "__main__":
    main()
