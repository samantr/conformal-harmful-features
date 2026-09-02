import argparse
from pathlib import Path

import yaml

from chf.experiments.baselines import run_required_baselines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    repository_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or repository_root / "outputs" / config["experiment_name"]
    selections, results, summary = run_required_baselines(config, output_dir, repository_root)
    print("\nMatched Base APS baseline summary:")
    print(summary[["model", "method", "target_size", "accuracy", "coverage", "mean_size"]].round(4).to_string(index=False))
    print(f"Saved {len(selections)} selections and {len(results)} final rows: {output_dir}")


if __name__ == "__main__":
    main()
