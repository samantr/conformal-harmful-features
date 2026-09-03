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


def make_group_four_way_split(
    labels: np.ndarray,
    groups: np.ndarray,
    group_counts: tuple[int, int, int, int],
    seed: int,
    *,
    candidate_permutations: int = 4096,
) -> FourWaySplit:
    """Create a deterministic four-way split with complete group separation.

    Candidate group assignments are sampled only from ``seed``. Among valid
    assignments that retain every observed class in every partition, the one
    minimizing row-count and class-proportion imbalance is selected. This is
    intended for grouped datasets such as UCI HAR where overlapping windows
    from the same participant must never cross experimental partitions.
    """
    y = np.asarray(labels)
    group_values = np.asarray(groups)
    if y.ndim != 1 or group_values.ndim != 1 or len(y) != len(group_values):
        raise ValueError("labels and groups must be one-dimensional arrays of equal length")
    unique_groups = np.unique(group_values)
    if sum(group_counts) != len(unique_groups) or any(count <= 0 for count in group_counts):
        raise ValueError("group counts must be positive and sum to the number of groups")
    if candidate_permutations <= 0:
        raise ValueError("candidate_permutations must be positive")

    classes = np.unique(y)
    global_proportions = np.asarray([(y == cls).mean() for cls in classes])
    target_row_fractions = np.asarray(group_counts, dtype=np.float64) / len(unique_groups)
    names = ("train", "tune", "calibration", "test")
    rng = np.random.default_rng(seed)
    best_score = np.inf
    best_parts: tuple[np.ndarray, ...] | None = None

    for _ in range(candidate_permutations):
        shuffled = rng.permutation(unique_groups)
        boundaries = np.cumsum((0, *group_counts))
        group_parts = tuple(
            shuffled[boundaries[index] : boundaries[index + 1]]
            for index in range(4)
        )
        index_parts = tuple(
            np.flatnonzero(np.isin(group_values, partition_groups))
            for partition_groups in group_parts
        )
        if any(len(np.unique(y[indices])) != len(classes) for indices in index_parts):
            continue

        row_fractions = np.asarray([len(indices) / len(y) for indices in index_parts])
        row_penalty = float(np.square(row_fractions - target_row_fractions).sum())
        class_penalty = 0.0
        for indices in index_parts:
            proportions = np.asarray([(y[indices] == cls).mean() for cls in classes])
            class_penalty += float(np.square(proportions - global_proportions).sum())
        score = row_penalty + class_penalty
        if score < best_score:
            best_score = score
            best_parts = index_parts

    if best_parts is None:
        raise ValueError(
            "could not find a group-disjoint split containing every class in every partition"
        )
    result = FourWaySplit(
        train=np.sort(best_parts[0]),
        tune=np.sort(best_parts[1]),
        calibration=np.sort(best_parts[2]),
        test=np.sort(best_parts[3]),
    )
    result.assert_disjoint()
    assigned = np.concatenate([getattr(result, name) for name in names])
    if len(assigned) != len(y) or len(np.unique(assigned)) != len(y):
        raise RuntimeError("group split failed to assign every row exactly once")
    partition_groups = [np.unique(group_values[getattr(result, name)]) for name in names]
    for left_index, left in enumerate(partition_groups):
        for right in partition_groups[left_index + 1 :]:
            if np.intersect1d(left, right).size:
                raise RuntimeError("group split contains a group in multiple partitions")
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
