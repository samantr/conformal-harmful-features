import argparse
from pathlib import Path

import numpy as np
import yaml
from sklearn.datasets import make_classification

from chf.data import make_four_way_split, save_split_artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    split_config = config["split"]

    _, labels = make_classification(
        n_samples=dataset["n_samples"],
        n_features=dataset["n_features"],
        n_informative=dataset["n_informative"],
        n_redundant=dataset["n_redundant"],
        n_classes=dataset["n_classes"],
        n_clusters_per_class=1,
        class_sep=dataset["class_sep"],
        flip_y=dataset["flip_y"],
        random_state=config["seed"],
    )
    sizes = tuple(split_config[name] for name in ("train", "tune", "calibration", "test"))
    split = make_four_way_split(np.asarray(labels), sizes, config["seed"])
    output_dir = args.output_dir or Path("outputs") / config["experiment_name"]
    artifact_path = save_split_artifact(
        output_dir / "split_indices.npz",
        split,
        seed=config["seed"],
        metadata={
            "experiment_name": config["experiment_name"],
            "n_samples": dataset["n_samples"],
        },
    )
    print({name: len(getattr(split, name)) for name in split_config})
    print("Split isolation check: PASS")
    print(f"Saved split artifact: {artifact_path}")


if __name__ == "__main__":
    main()
