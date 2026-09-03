from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class FourWaySplit:
    train: np.ndarray
    tune: np.ndarray
    calibration: np.ndarray
    test: np.ndarray

    def assert_disjoint(self) -> None:
        parts = [self.train, self.tune, self.calibration, self.test]
        for left_index, left in enumerate(parts):
            for right in parts[left_index + 1 :]:
                if np.intersect1d(left, right).size:
                    raise ValueError("Four-way split contains overlapping indices")


def make_four_way_split(
    labels: np.ndarray,
    sizes: tuple[int, int, int, int],
    seed: int,
) -> FourWaySplit:
    """Create stratified train/tune/calibration/test index partitions."""
    if sum(sizes) != len(labels) or any(size <= 0 for size in sizes):
        raise ValueError("Split sizes must be positive and sum to dataset size")

    indices = np.arange(len(labels))
    train_size, tune_size, calibration_size, test_size = sizes
    train, remainder = train_test_split(
        indices,
        train_size=train_size,
        test_size=tune_size + calibration_size + test_size,
        stratify=labels,
        random_state=seed,
    )
    tune, remainder = train_test_split(
        remainder,
        train_size=tune_size,
        test_size=calibration_size + test_size,
        stratify=labels[remainder],
        random_state=seed + 1,
    )
    calibration, test = train_test_split(
        remainder,
        train_size=calibration_size,
        test_size=test_size,
        stratify=labels[remainder],
        random_state=seed + 2,
    )
    result = FourWaySplit(train, tune, calibration, test)
    result.assert_disjoint()
    return result


def stratified_subsample(
    indices: np.ndarray,
    labels: np.ndarray,
    *,
    max_samples: int | None,
    seed: int,
) -> np.ndarray:
    """Select a deterministic class-stratified subset of existing indices."""
    source_indices = np.asarray(indices, dtype=np.int64)
    all_labels = np.asarray(labels)
    if source_indices.ndim != 1 or len(np.unique(source_indices)) != len(source_indices):
        raise ValueError("source indices must be a one-dimensional unique array")
    if len(source_indices) == 0:
        raise ValueError("source indices must not be empty")
    if source_indices.min() < 0 or source_indices.max() >= len(all_labels):
        raise ValueError("source indices are out of bounds for labels")
    if max_samples is None or max_samples >= len(source_indices):
        return source_indices.copy()
    if max_samples <= 0:
        raise ValueError("max_samples must be positive when specified")
    selected, _ = train_test_split(
        source_indices,
        train_size=max_samples,
        stratify=all_labels[source_indices],
        random_state=seed,
    )
    return np.sort(np.asarray(selected, dtype=np.int64))
