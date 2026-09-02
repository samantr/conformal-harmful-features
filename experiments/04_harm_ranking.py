import argparse
from pathlib import Path

import yaml

from chf.experiments.harm_ranking import run_harm_ranking


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    repository_root = Path(__file__).resolve().parents[1]
    output_dir = (
        args.output_dir
        or repository_root / "outputs" / config["experiment_name"]
    )
    rankings, stability, consensus = run_harm_ranking(
        config, output_dir, repository_root
    )
    print("\nTop consensus features:")
    print(
        consensus[
            [
                "feature_name",
                "feature_role",
                "constrained_eligible_fraction",
                "constrained_harm_candidate_fraction",
                "mean_efficiency_gain",
                "mean_accuracy_loss",
                "constrained_consensus_rank",
                "weighted_consensus_rank",
                "pareto_consensus_rank",
            ]
        ]
        .head(10)
        .round(4)
        .to_string(index=False)
    )
    print("\nRank stability by formulation:")
    print(
        stability.groupby("formulation")["mean_rank_correlation"]
        .agg(["mean", "min"])
        .round(4)
        .to_string()
    )
    print(f"Saved {len(rankings)} Phase 3 ranking rows: {output_dir}")


if __name__ == "__main__":
    main()
