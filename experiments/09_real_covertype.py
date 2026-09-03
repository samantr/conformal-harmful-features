import argparse
from pathlib import Path

import yaml

from chf.experiments.real_datasets import run_real_dataset_benchmark


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
    selections, factorial, interactions, main_table = run_real_dataset_benchmark(
        config, output_dir, repository_root
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
            "mean_size",
            "sscv",
        ],
    ]
    print("\nCovertype all-feature and proposed-selection results:")
    print(proposed.round(4).to_string(index=False))
    print(
        f"\nSaved {len(selections)} selections, {len(factorial)} factorial "
        f"rows, and {len(interactions)} interaction rows: {output_dir}"
    )


if __name__ == "__main__":
    main()
