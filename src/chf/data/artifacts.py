import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .splits import FourWaySplit


@dataclass(frozen=True)
class SplitArtifact:
    split: FourWaySplit
    seed: int
    metadata: dict[str, Any]


def save_split_artifact(
    path: str | Path,
    split: FourWaySplit,
    seed: int,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Persist exact split indices, their seed, and JSON audit metadata."""
    split.assert_disjoint()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(dict(metadata or {}), sort_keys=True)
    np.savez_compressed(
        destination,
        train=split.train,
        tune=split.tune,
        calibration=split.calibration,
        test=split.test,
        seed=np.asarray(seed, dtype=np.int64),
        metadata=np.asarray(metadata_json),
    )
    return destination


def load_split_artifact(path: str | Path) -> SplitArtifact:
    """Load and validate a split artifact without permitting pickled data."""
    with np.load(Path(path), allow_pickle=False) as stored:
        required = {"train", "tune", "calibration", "test", "seed", "metadata"}
        missing = required.difference(stored.files)
        if missing:
            raise ValueError(f"split artifact is missing fields: {sorted(missing)}")
        split = FourWaySplit(
            train=np.asarray(stored["train"], dtype=np.int64),
            tune=np.asarray(stored["tune"], dtype=np.int64),
            calibration=np.asarray(stored["calibration"], dtype=np.int64),
            test=np.asarray(stored["test"], dtype=np.int64),
        )
        seed = int(stored["seed"])
        metadata = json.loads(str(stored["metadata"]))
    split.assert_disjoint()
    return SplitArtifact(split=split, seed=seed, metadata=metadata)
