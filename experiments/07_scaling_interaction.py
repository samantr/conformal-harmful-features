import argparse
from pathlib import Path

import yaml

from chf.experiments.scaling_interaction import run_scaling_interaction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--selections",
        type=Path,
        help="Frozen baseline_selections.csv from Phase 5",
    )
    replay = parser.add_mutually_exclusive_group()
    replay.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse compatible Phase 6 artifacts, or refit Phase 6 only when "
            "legacy artifacts lack grouped coverage metrics"
        ),
    )
    replay.add_argument(
        "--revalidate-only",
        action="store_true",
        help="Rebuild validation and reports from complete frozen Phase 6 artifacts",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    repository_root = Path(__file__).resolve().parents[1]
    output_dir = (
        args.output_dir
        or repository_root / "outputs" / config["experiment_name"]
    )
    results, interactions, rank_stability, summary = run_scaling_interaction(
        config,
        output_dir,
        repository_root,
        selections_path=args.selections,
        resume=args.resume,
        revalidate_only=args.revalidate_only,
    )
    proposed = interactions.loc[
        interactions["selection_type"] == "proposed_selection",
        [
            "model",
            "method",
            "score",
            "scaling",
            "feature_gain_at_base",
            "scaling_gain_on_all_features",
            "observed_joint_gain",
            "interaction_size_gain",
            "interaction_label",
        ],
    ]
    print("\nProposed feature-selection/scaling interactions:")
    print(proposed.round(4).to_string(index=False))
    print("\nAggregate descriptive interaction summary:")
    print(summary.round(4).to_string(index=False))
    print(
        f"Saved {len(results)} factorial rows, {len(interactions)} interaction "
        f"rows, and {len(rank_stability)} rank comparisons: {output_dir}"
    )


if __name__ == "__main__":
    main()
