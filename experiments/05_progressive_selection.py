import argparse
from pathlib import Path

import yaml

from chf.experiments.progressive_selection import run_progressive_selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    repository_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or repository_root / "outputs" / config["experiment_name"]
    paths, consensus, final = run_progressive_selection(
        config, output_dir, repository_root
    )
    print("\nFrozen tuning-selected subsets:")
    print(consensus.loc[consensus["selected_for_final"], [
        "model", "method", "n_features", "mean_accuracy_loss",
        "cumulative_efficiency_gain", "mean_conditional_violation",
    ]].round(4).to_string(index=False))
    print("\nFinal Base APS results:")
    print(final.loc[(final["scaling"] == "base") & (final["score"] == "aps"), [
        "model", "method", "n_features", "accuracy", "coverage", "mean_size",
        "sscv", "class_coverage_max_deviation",
    ]].round(4).to_string(index=False))
    print(f"Saved {len(paths)} pipeline path rows and {len(final)} final rows: {output_dir}")


if __name__ == "__main__":
    main()
