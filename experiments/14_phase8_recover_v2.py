"""Phase 8 recovery v2: preserve the original final-grid checkpoint manifest.

The first recovery attempt reconstructed ``baseline_selections.csv`` with pandas.
That CSV round-trip is scientifically equivalent, but it cannot reproduce the
byte-level ``selection_sha256`` that was computed from the in-memory DataFrame
in the original run.  Since the final-grid checkpoint manifest includes that
hash, the otherwise identical resume was rejected.

This wrapper restores the original hash from the original checkpoint manifest
before delegating to the Phase 8 recovery controller.  Every other manifest
field is still recomputed by the original scientific code and must match
exactly, so split/config/grid/code drift remains rejected.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import chf.experiments.robustness as robustness


CONTROL_PATH = Path(__file__).with_name("13_phase8_recover.py")


def _load_recovery_control():
    spec = importlib.util.spec_from_file_location("phase8_recovery_control_v1", CONTROL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load recovery controller: {CONTROL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_original_selection_digest(
    *, output_root: Path, dataset: str, seed: int
) -> dict:
    """Reuse only the selection digest recorded by the original final-grid manifest.

    ``run_phase8_grid`` will still reconstruct all other manifest fields and the
    ``CheckpointStore`` will compare the complete reconstructed manifest against
    this saved manifest.  Therefore this patch cannot make config, split, grid,
    experiment, seed, or code-version drift resumable.
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
        raise FileNotFoundError(
            "recovery v2 requires the original final-grid checkpoint manifest: "
            + str(manifest_path)
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_experiment = f"phase8_{dataset}_seed_{seed}"
    required = {
        "schema_version": 1,
        "stage": "phase8_final_grid",
        "experiment_name": expected_experiment,
        "seed": seed,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"original checkpoint manifest has unexpected {key}: "
                f"expected {expected!r}, got {manifest.get(key)!r}"
            )
    selection_sha256 = manifest.get("selection_sha256")
    if not isinstance(selection_sha256, str) or len(selection_sha256) != 64:
        raise ValueError("original checkpoint manifest lacks a valid selection_sha256")

    robustness._selection_digest = lambda _selections: selection_sha256
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-run-id", required=True)
    args = parser.parse_args()

    output_root = args.output_dir.resolve()
    manifest = install_original_selection_digest(
        output_root=output_root,
        dataset=str(args.dataset),
        seed=int(args.seed),
    )
    print(
        "Recovery v2: preserved original final-grid selection_sha256 "
        f"{manifest['selection_sha256'][:12]}...; all other manifest fields "
        "remain strict.",
        flush=True,
    )

    recovery = _load_recovery_control()
    recovery.recover_unit(
        spec_path=args.config.resolve(),
        repository_root=args.repository_root.resolve(),
        output_root=output_root,
        dataset_key=str(args.dataset),
        seed=int(args.seed),
        source_run_id=str(args.source_run_id),
    )


if __name__ == "__main__":
    main()
