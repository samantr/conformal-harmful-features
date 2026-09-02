import argparse
from pathlib import Path

import numpy as np
import yaml
from sklearn.datasets import make_classification

from chf.data import make_four_way_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
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
    print({name: len(getattr(split, name)) for name in split_config})
    print("Split isolation check: PASS")


if __name__ == "__main__":
    main()

