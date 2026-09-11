import json
from pathlib import Path

import numpy as np
import yaml

from chf.experiments.protocol import dataset_from_config, experiment_split, split_id


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "human_activity_recognition.yaml"
OUTPUT_DIR = REPOSITORY_ROOT / "outputs" / "phase7c_har_provenance"


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    dataset = dataset_from_config(config, REPOSITORY_ROOT)
    split = experiment_split(config, dataset)
    split.assert_disjoint()

    partition_subjects = {}
    partition_rows = {}
    partition_classes = {}
    for name in ("train", "tune", "calibration", "test"):
        indices = getattr(split, name)
        partition_subjects[name] = sorted(
            int(value) for value in np.unique(dataset.groups[indices])
        )
        partition_rows[name] = int(len(indices))
        partition_classes[name] = sorted(
            int(value) for value in np.unique(dataset.labels[indices])
        )

    subject_sets = [set(values) for values in partition_subjects.values()]
    subject_disjoint = all(
        left.isdisjoint(right)
        for left_index, left in enumerate(subject_sets)
        for right in subject_sets[left_index + 1 :]
    )
    all_classes_present = all(
        values == list(range(len(dataset.class_names)))
        for values in partition_classes.values()
    )

    record = {
        "dataset": dataset.name,
        "source_url": dataset.source_url,
        "archive_sha256": dataset.archive_sha256,
        "shape": list(dataset.features.shape),
        "n_classes": int(len(np.unique(dataset.labels))),
        "n_subjects": int(len(np.unique(dataset.groups))),
        "split_id": split_id(split),
        "partition_rows": partition_rows,
        "partition_subjects": partition_subjects,
        "partition_classes": partition_classes,
        "subject_disjoint": subject_disjoint,
        "all_classes_present_in_every_partition": all_classes_present,
        "finite_features": bool(np.isfinite(dataset.features).all()),
    }
    if dataset.features.shape != (
        int(config["dataset"]["n_samples"]),
        int(config["dataset"]["n_features"]),
    ):
        raise RuntimeError("HAR shape differs from the frozen Phase 7C declaration")
    if not subject_disjoint:
        raise RuntimeError("HAR subject leakage detected across outer partitions")
    if not all_classes_present:
        raise RuntimeError("a HAR partition is missing at least one activity class")
    if not record["finite_features"]:
        raise RuntimeError("HAR contains non-finite features")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "har_provenance.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    print(
        "\nPin dataset.archive_sha256 in configs/human_activity_recognition.yaml "
        "to the value above before running the final Phase 7C benchmark."
    )


if __name__ == "__main__":
    main()
