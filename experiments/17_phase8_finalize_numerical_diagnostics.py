"""Normalize Phase 8 numerical diagnostics across original and recovered units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from chf.experiments.checkpoints import atomic_write_csv, atomic_write_json


BOUNDARY_COLUMNS = [
    "calibration_zero_count",
    "calibration_exactly_one_count",
    "test_zero_count",
    "test_exactly_one_count",
]
CONDITION_COLUMNS = [
    "dataset",
    "seed",
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


def finalize(output_dir: Path) -> None:
    path = output_dir / "phase8_all_results.csv"
    frame = pd.read_csv(path)
    for column in ("raps_lambda", "raps_k_reg"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["numerical_boundary_count"] = frame[BOUNDARY_COLUMNS].sum(axis=1).astype(int)
    frame["numerical_boundary_flag"] = frame["numerical_boundary_count"].gt(0)

    base = frame.loc[frame["scaling"].eq("base"), CONDITION_COLUMNS + BOUNDARY_COLUMNS].copy()
    if base.duplicated(CONDITION_COLUMNS).any():
        raise ValueError("Base numerical reference is not unique by condition")
    base = base.rename(columns={c: f"base_{c}" for c in BOUNDARY_COLUMNS})
    merged = frame.merge(base, on=CONDITION_COLUMNS, how="left", validate="many_to_one")

    # Base rows are their own native reference. A scaled row lacking Base is a
    # protocol error because every full-grid scaled condition has matching Base.
    scaled_missing = merged.loc[
        ~merged["scaling"].eq("base"), [f"base_{c}" for c in BOUNDARY_COLUMNS]
    ].isna().any(axis=1)
    if scaled_missing.any():
        raise ValueError("a scaled Phase 8 cell lacks its matching Base diagnostic")

    excess_columns = []
    for column in BOUNDARY_COLUMNS:
        reference = merged[f"base_{column}"].fillna(merged[column])
        excess = (merged[column] - reference).clip(lower=0).astype(int)
        name = f"{column}_excess_vs_base"
        merged[name] = excess
        excess_columns.append(name)
    merged["scaling_induced_boundary_count"] = merged[excess_columns].sum(axis=1).astype(int)
    merged["scaling_induced_boundary_flag"] = merged[
        "scaling_induced_boundary_count"
    ].gt(0)
    merged = merged.drop(columns=[f"base_{c}" for c in BOUNDARY_COLUMNS])
    atomic_write_csv(path, merged)

    grouped = (
        merged.groupby(["dataset", "scaling"], sort=False, dropna=False)
        .agg(
            rows=("scaling", "size"),
            raw_boundary_flagged_rows=("numerical_boundary_flag", "sum"),
            scaling_induced_flagged_rows=("scaling_induced_boundary_flag", "sum"),
            maximum_boundary_count=("numerical_boundary_count", "max"),
            maximum_scaling_induced_count=("scaling_induced_boundary_count", "max"),
            mean_coverage=("coverage", "mean"),
            mean_empty_set_rate=("empty_set_rate", "mean"),
        )
        .reset_index()
    )
    atomic_write_csv(output_dir / "phase8_numerical_diagnostics.csv", grouped)
    atomic_write_json(
        output_dir / "phase8_numerical_diagnostics.json",
        {
            "status": "REPORTED_NOT_REPAIRED",
            "handling": (
                "all prespecified cells retained; exact-boundary events flagged; "
                "no calibration/test-driven temperature retuning or cell replacement"
            ),
            "raw_boundary_flagged_rows": int(merged["numerical_boundary_flag"].sum()),
            "scaling_induced_flagged_rows": int(
                merged["scaling_induced_boundary_flag"].sum()
            ),
            "datasets_with_scaling_induced_boundaries": sorted(
                merged.loc[
                    merged["scaling_induced_boundary_flag"], "dataset"
                ].astype(str).unique().tolist()
            ),
            "post_run_amendment": "PHASE8_POSTRUN_AMENDMENT.md",
        },
    )
    print(
        "Phase 8 numerical diagnostics finalized: "
        f"{int(merged['numerical_boundary_flag'].sum())} raw flagged rows, "
        f"{int(merged['scaling_induced_boundary_flag'].sum())} scaling-induced flagged rows.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    finalize(args.output_dir.resolve())


if __name__ == "__main__":
    main()
