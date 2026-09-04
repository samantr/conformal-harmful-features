import argparse
from pathlib import Path
from typing import Any, Mapping

import yaml

from chf.experiments.coverage_validation import coverage_validation_mode
from chf.experiments.real_datasets import run_real_dataset_benchmark


def _validate_har_config(config: Mapping[str, Any]) -> None:
    dataset = config.get("dataset", {})
    split = config.get("split", {})
    if dataset.get("kind") != "human_activity_recognition":
        raise ValueError("Phase 7C runner requires the human_activity_recognition dataset")
    if not str(dataset.get("archive_sha256", "")).strip():
        raise ValueError("Phase 7C requires a pinned dataset.archive_sha256")
    if split.get("unit") != "groups":
        raise ValueError("Phase 7C HAR requires split.unit: groups")
    if coverage_validation_mode(config) != "grouped":
        raise ValueError("Phase 7C HAR requires grouped coverage validation")
    expected_subjects = int(dataset.get("n_subjects", 0))
    configured_subjects = sum(
        int(split[name]) for name in ("train", "tune", "calibration", "test")
    )
    if expected_subjects <= 0 or configured_subjects != expected_subjects:
        raise ValueError(
            "Phase 7C group counts must sum to dataset.n_subjects"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    replay = parser.add_mutually_exclusive_group()
    replay.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse Phase 4-5 and compatible Phase 6 artifacts; legacy Phase 6 "
            "results receive only the required Phase 6 refit"
        ),
    )
    replay.add_argument(
        "--revalidate-only",
        action="store_true",
        help="Regenerate validation/reports without fitting any classifier",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    _validate_har_config(config)
    repository_root = Path(__file__).resolve().parents[1]
    output_dir = (
        args.output_dir
        or repository_root / "outputs" / config["experiment_name"]
    )
    selections, factorial, interactions, main_table = run_real_dataset_benchmark(
        config,
        output_dir,
        repository_root,
        resume=args.resume,
        revalidate_only=args.revalidate_only,
    )

    proposed = main_table.loc[
        main_table["selection_type"].isin(
            ["all_features", "proposed_selection"]
        ),
        [
            "model",
            "method",
            "n_features",
            "scaling",
            "score",
            "accuracy",
            "coverage",
            "group_macro_coverage",
            "group_coverage_ci_lower",
            "group_coverage_ci_upper",
            "group_coverage_status",
            "mean_size",
            "sscv",
        ],
    ]
    print("\nHAR all-feature and proposed-selection results:")
    print(proposed.round(4).to_string(index=False))
    print(
        f"\nSaved {len(selections)} selections, {len(factorial)} factorial "
        f"rows, and {len(interactions)} interaction rows: {output_dir}"
    )


if __name__ == "__main__":
    main()
