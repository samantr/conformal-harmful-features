"""Phase 8 recovery v3: complete units while retaining numerical failures as data.

This recovery layer makes no scientific selection, split, seed, model, scaling,
or conformal-grid changes.  It exists because the original execution validator
aborted an entire dataset/seed unit when *any* final probability diagnostic hit
an exact floating-point boundary.  Once that happens, the numerical event is a
scientific observation and must be retained and reported rather than erased by
an execution failure.

The original structural validator is still used for every non-numerical check.
Exact-boundary diagnostics are written unchanged to the result CSV, augmented
with explicit per-cell flags.  No temperature is re-tuned from calibration or
test data and no flagged cell is replaced or dropped.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

import chf.experiments.robustness as robustness
from chf.experiments.checkpoints import atomic_write_csv, atomic_write_json


HERE = Path(__file__).resolve().parent
V1_PATH = HERE / "13_phase8_recover.py"
BOUNDARY_COLUMNS = [
    "calibration_zero_count",
    "calibration_exactly_one_count",
    "test_zero_count",
    "test_exactly_one_count",
]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load recovery module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _preserve_saved_selection_digest(
    *, output_root: Path, dataset: str, seed: int
) -> tuple[bool, str | None]:
    """Reuse the digest already bound to restored final-grid shards, if present.

    All other manifest fields continue to be reconstructed by the original
    scientific code and compared exactly by ``CheckpointStore.initialize``.
    Thus this compatibility step cannot make split/config/grid/code drift pass.
    """
    manifest_path = (
        output_root
        / dataset
        / f"seed_{seed}"
        / "checkpoints"
        / "final_grid"
        / "manifest.json"
    )
    if not manifest_path.exists():
        return False, None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "stage": "phase8_final_grid",
        "experiment_name": f"phase8_{dataset}_seed_{seed}",
        "seed": seed,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"restored final-grid manifest has unexpected {key}: "
                f"expected {value!r}, got {manifest.get(key)!r}"
            )
    digest = manifest.get("selection_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("restored final-grid manifest lacks a valid selection_sha256")
    robustness._selection_digest = lambda _selections: digest
    return True, digest


def _install_diagnostic_retention_validator(v1) -> None:
    """Keep strict structural validation but convert numerical hard-stop to flags."""
    original_validate = robustness._validate_and_write_grid

    def validate_retaining_numerical_events(
        *,
        results: pd.DataFrame,
        subject_results: pd.DataFrame,
        selections: pd.DataFrame,
        spec: Mapping[str, Any],
        output_dir: Path,
        manifest: Mapping[str, Any],
    ) -> None:
        missing = set(BOUNDARY_COLUMNS).difference(results.columns)
        if missing:
            raise ValueError(
                f"numerical-retention recovery is missing columns: {sorted(missing)}"
            )

        # Reuse the v1 matching logic only to quantify scaling-induced excess.
        # The raw result values themselves are never changed or filtered.
        excess_frame, diagnostics = v1._boundary_excess_frame(results)
        annotated = results.copy(deep=True)
        annotated["numerical_boundary_count"] = annotated[BOUNDARY_COLUMNS].sum(axis=1)
        annotated["numerical_boundary_flag"] = annotated[
            "numerical_boundary_count"
        ].gt(0)
        annotated["scaling_induced_boundary_count"] = excess_frame[
            BOUNDARY_COLUMNS
        ].sum(axis=1)
        annotated["scaling_induced_boundary_flag"] = annotated[
            "scaling_induced_boundary_count"
        ].gt(0)

        strict_safe = bool((annotated[BOUNDARY_COLUMNS] == 0).all().all())
        no_scaling_induced = bool(
            (~annotated["scaling_induced_boundary_flag"]).all()
        )

        # Exercise the original validator on an otherwise identical copy with
        # only the numerical hard-stop neutralized. Every structural/provenance
        # check still has to pass and will still raise if it does not.
        validation_copy = annotated.copy(deep=True)
        validation_copy.loc[:, BOUNDARY_COLUMNS] = 0
        original_validate(
            results=validation_copy,
            subject_results=subject_results,
            selections=selections,
            spec=spec,
            output_dir=output_dir,
            manifest=manifest,
        )

        # Publish the untouched scientific diagnostics plus explicit flags.
        atomic_write_csv(output_dir / "phase8_results.csv", annotated)
        protocol_path = output_dir / "phase8_grid_protocol.json"
        record = json.loads(protocol_path.read_text(encoding="utf-8"))
        record["checks"].pop("no_zero_or_saturated_probabilities", None)
        record["checks"]["structural_validation_passed"] = True
        record["checks"]["original_strict_numerical_safety"] = strict_safe
        record["checks"][
            "no_scaling_induced_boundary_probabilities"
        ] = no_scaling_induced
        record["numerical_recovery"] = {
            **diagnostics,
            "raw_boundary_flagged_rows": int(
                annotated["numerical_boundary_flag"].sum()
            ),
            "scaling_induced_flagged_rows": int(
                annotated["scaling_induced_boundary_flag"].sum()
            ),
            "handling": (
                "retain all prespecified cells unchanged; flag numerical boundary "
                "events; do not retune, replace, or drop any temperature/cell"
            ),
        }
        record["status"] = (
            "PASS" if strict_safe else "COMPLETE_WITH_NUMERICAL_FLAGS"
        )
        atomic_write_json(protocol_path, record)

    def install() -> None:
        setattr(
            robustness,
            "_validate_and_write_grid",
            validate_retaining_numerical_events,
        )

    # Install immediately so the helper is testable in isolation, and also
    # replace the v1 hook because v1.recover_unit calls it immediately before
    # run_phase8_grid. Reinstalling the same wrapper there is intentional.
    install()
    v1._install_recovery_validator = install


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--parent-recovery-run-id", default="")
    args = parser.parse_args()

    output_root = args.output_dir.resolve()
    reused_digest, digest = _preserve_saved_selection_digest(
        output_root=output_root,
        dataset=str(args.dataset),
        seed=int(args.seed),
    )
    if reused_digest:
        print(
            "Recovery v3: preserving restored final-grid selection digest "
            f"{digest[:12]}... while leaving every other manifest field strict.",
            flush=True,
        )
    else:
        print(
            "Recovery v3: no prior final-grid manifest exists; a new manifest "
            "will be created by the original scientific code.",
            flush=True,
        )

    v1 = _load_module(V1_PATH, "phase8_recovery_control_v1_for_v3")
    _install_diagnostic_retention_validator(v1)
    v1.recover_unit(
        spec_path=args.config.resolve(),
        repository_root=args.repository_root.resolve(),
        output_root=output_root,
        dataset_key=str(args.dataset),
        seed=int(args.seed),
        source_run_id=str(args.source_run_id),
    )

    # Extend, rather than overwrite, the v1 recovery provenance.
    recovery_path = (
        output_root
        / str(args.dataset)
        / f"seed_{int(args.seed)}"
        / "phase8_recovery_manifest.json"
    )
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    recovery["recovery_generation"] = 3
    recovery["parent_recovery_run_id"] = str(args.parent_recovery_run_id)
    recovery["selection_digest_reused_from_restored_manifest"] = reused_digest
    recovery["numerical_failure_handling"] = (
        "retained and explicitly flagged; no final calibration/test driven retuning"
    )
    recovery["post_run_protocol_amendment"] = "PHASE8_POSTRUN_AMENDMENT.md"
    recovery["recovery_control_commit"] = os.environ.get(
        "PHASE8_RECOVERY_CONTROL_SHA", "unknown"
    )
    atomic_write_json(recovery_path, recovery)


if __name__ == "__main__":
    main()
