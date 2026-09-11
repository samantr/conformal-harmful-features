"""Strict provenance preflight before Phase 8 aggregate-only inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chf.experiments.phase8_runner import load_phase8_spec, phase8_run_plan


SCIENTIFIC_VERSION = "e4b3645"


def audit(*, spec_path: Path, repository_root: Path, output_dir: Path) -> None:
    spec = load_phase8_spec(spec_path)
    plan = phase8_run_plan(spec, repository_root)
    failures: list[str] = []
    recovered = 0
    for row in plan.to_dict("records"):
        dataset = str(row["dataset_key"])
        seed = int(row["seed"])
        unit = output_dir / str(row["output_subdirectory"])
        completion_path = unit / "phase8_unit_complete.json"
        results_path = unit / "phase8_results.csv"
        if not completion_path.exists() or not results_path.exists():
            failures.append(f"{dataset}/seed_{seed}: missing completion/results")
            continue
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        expected = {
            "status": "PASS",
            "dataset": dataset,
            "seed": seed,
            "config_sha256": str(row["config_sha256"]),
            "code_version": SCIENTIFIC_VERSION,
        }
        for key, value in expected.items():
            if completion.get(key) != value:
                failures.append(
                    f"{dataset}/seed_{seed}: {key} expected {value!r}, "
                    f"got {completion.get(key)!r}"
                )
        if dataset == "human_activity_recognition" and not (
            unit / "phase8_subject_results.csv"
        ).exists():
            failures.append(f"{dataset}/seed_{seed}: missing subject results")
        recovery_path = unit / "phase8_recovery_manifest.json"
        if recovery_path.exists():
            recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
            if recovery.get("status") != "PASS":
                failures.append(f"{dataset}/seed_{seed}: recovery manifest not PASS")
            recovered += 1

    if recovered != 12:
        failures.append(f"expected 12 recovered units, found {recovered}")
    if failures:
        raise RuntimeError("Phase 8 aggregate provenance audit failed:\n- " + "\n- ".join(failures))
    print(
        f"Phase 8 aggregate provenance audit PASS: {len(plan)} units, "
        f"{recovered} recovered, scientific version {SCIENTIFIC_VERSION}.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit(
        spec_path=args.config.resolve(),
        repository_root=args.repository_root.resolve(),
        output_dir=args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
