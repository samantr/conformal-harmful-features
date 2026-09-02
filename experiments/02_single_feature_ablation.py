import argparse
from pathlib import Path

import yaml

from chf.experiments.single_feature import run_retrain_ablation


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
    results, summary = run_retrain_ablation(config, output_dir, repository_root)
    print(
        summary[
            [
                "feature_name",
                "feature_role",
                "max_accuracy_loss",
                "mean_base_aps_size_reduction",
                "max_abs_coverage_delta",
                "descriptive_pattern_pass",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print(f"Saved {len(results)} Phase 2 retrain-ablation rows: {output_dir}")


if __name__ == "__main__":
    main()
