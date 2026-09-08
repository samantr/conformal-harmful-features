"""Run the preregistered Phase 8 10->20 seed extension without changing science.

This control script is executed from the current control checkout while the
scientific package is installed from the original Phase 8 commit ``e4b3645``.
The extension seed specification still contains exactly ten seeds, so the
original Phase 8 scientific validator remains untouched.  The only operational
compatibility layers installed are the already documented pandas-3 HAR merge
normalization and the numerical-boundary retention policy from the completed
10-seed recovery.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from chf.experiments.checkpoints import atomic_write_json
from chf.experiments.phase8_runner import load_phase8_spec, run_phase8_unit
from chf.experiments.protocol import code_version


HERE = Path(__file__).resolve().parent
V1_PATH = HERE / "13_phase8_recover.py"
V3_PATH = HERE / "15_phase8_recover_v3.py"
EXPECTED_SCIENTIFIC_VERSION = "e4b3645"
EXPECTED_EXTENSION_SEEDS = tuple(range(53, 63))
EXPECTED_EXTENSION_DATASETS = {
    "dry_bean",
    "human_activity_recognition",
}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load control module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def validate_extension_design(
    *, extension_spec_path: Path, repository_root: Path
) -> dict[str, Any]:
    """Prove that the extension changes only the seed batch/execution metadata."""
    extension = _load_yaml(extension_spec_path)
    initial = _load_yaml(repository_root / "configs" / "phase8_robustness.yaml")

    # Exercise the original scientific validator: the extension is deliberately
    # another 10-seed batch, not a mutation of the initial validator.
    load_phase8_spec(extension_spec_path)

    seeds = tuple(int(value) for value in extension.get("seeds", ()))
    if seeds != EXPECTED_EXTENSION_SEEDS:
        raise ValueError(
            f"extension seeds must be exactly {EXPECTED_EXTENSION_SEEDS}, got {seeds}"
        )
    meta = extension.get("precision_extension", {})
    if set(meta.get("extended_datasets", ())) != EXPECTED_EXTENSION_DATASETS:
        raise ValueError("precision extension must run Dry Bean and HAR only")
    if list(meta.get("stopped_at_10_seeds", ())) != ["covertype"]:
        raise ValueError("Covertype must remain stopped at ten seeds")

    frozen_equal_keys = (
        "phase",
        "protocol_version",
        "frozen_parent_commit",
        "datasets",
        "selection",
        "conformal_grid",
        "statistics",
        "frozen_artifact_policy",
    )
    changed = [key for key in frozen_equal_keys if extension.get(key) != initial.get(key)]
    if changed:
        raise ValueError(
            "extension changed frozen scientific fields: " + ", ".join(changed)
        )

    if int(initial["statistics"]["precision_escalation"]["maximum_seeds"]) != 20:
        raise ValueError("initial Phase 8 protocol did not preregister a 20-seed maximum")
    return extension


def _install_completed_run_compatibility() -> None:
    """Install only compatibility layers already documented after the 10-seed run."""
    v1 = _load_module(V1_PATH, "phase8_extension_v1_compat")
    v3 = _load_module(V3_PATH, "phase8_extension_v3_compat")
    v1._install_har_merge_compatibility()
    v3._install_diagnostic_retention_validator(v1)


def run_extension_unit(
    *,
    extension_spec_path: Path,
    repository_root: Path,
    output_root: Path,
    dataset: str,
    seed: int,
) -> None:
    spec = validate_extension_design(
        extension_spec_path=extension_spec_path,
        repository_root=repository_root,
    )
    if dataset not in EXPECTED_EXTENSION_DATASETS:
        raise ValueError(f"dataset {dataset!r} did not trigger the precision extension")
    if seed not in EXPECTED_EXTENSION_SEEDS:
        raise ValueError(f"seed {seed} is not an extension seed")

    scientific_version = code_version(repository_root)
    if scientific_version != EXPECTED_SCIENTIFIC_VERSION:
        raise ValueError(
            "precision extension must use the original clean scientific commit; "
            f"expected {EXPECTED_SCIENTIFIC_VERSION}, got {scientific_version}"
        )

    _install_completed_run_compatibility()
    run_phase8_unit(
        spec=spec,
        repository_root=repository_root,
        output_root=output_root,
        dataset_key=dataset,
        seed=seed,
        resume=True,
    )

    unit = output_root / dataset / f"seed_{seed}"
    completion_path = unit / "phase8_unit_complete.json"
    if not completion_path.exists():
        raise FileNotFoundError(f"extension unit did not write completion record: {unit}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "PASS":
        raise ValueError(f"extension unit is not complete: {unit}")

    result_path = unit / "phase8_results.csv"
    results = pd.read_csv(result_path, low_memory=False)
    raw_flags = int(
        results.get("numerical_boundary_flag", pd.Series(False, index=results.index))
        .fillna(False)
        .astype(bool)
        .sum()
    )
    induced_flags = int(
        results.get(
            "scaling_induced_boundary_flag", pd.Series(False, index=results.index)
        )
        .fillna(False)
        .astype(bool)
        .sum()
    )
    grid_protocol = json.loads(
        (unit / "phase8_grid_protocol.json").read_text(encoding="utf-8")
    )
    atomic_write_json(
        unit / "phase8_precision_extension_manifest.json",
        {
            "status": "PASS",
            "dataset": dataset,
            "seed": int(seed),
            "initial_precision_decision_run_id": str(
                spec["precision_extension"]["triggered_by_completed_run_id"]
            ),
            "scientific_code_version": scientific_version,
            "extension_control_commit": os.environ.get(
                "PHASE8_EXTENSION_CONTROL_SHA", "unknown"
            ),
            "grid_protocol_status": grid_protocol.get("status"),
            "raw_boundary_flagged_rows": raw_flags,
            "scaling_induced_boundary_flagged_rows": induced_flags,
            "numerical_failure_handling": "retained_and_flagged_not_repaired",
            "post_run_amendment": "PHASE8_POSTRUN_AMENDMENT.md",
            "precision_extension_protocol": "PHASE8_PRECISION_EXTENSION.md",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    spec = validate_extension_design(
        extension_spec_path=args.config.resolve(),
        repository_root=args.repository_root.resolve(),
    )
    if args.validate_only:
        print(
            "Phase 8 precision extension validated: "
            f"seeds={spec['seeds']}, datasets={spec['precision_extension']['extended_datasets']}"
        )
        return

    run_extension_unit(
        extension_spec_path=args.config.resolve(),
        repository_root=args.repository_root.resolve(),
        output_root=args.output_dir.resolve(),
        dataset=str(args.dataset),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
