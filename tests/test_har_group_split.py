from types import SimpleNamespace

import numpy as np

from chf.data import make_group_four_way_split
from chf.experiments.protocol import experiment_split


def _grouped_fixture() -> tuple[np.ndarray, np.ndarray]:
    labels = []
    groups = []
    for subject in range(1, 31):
        for activity in range(6):
            labels.extend([activity, activity])
            groups.extend([subject, subject])
    return np.asarray(labels, dtype=np.int64), np.asarray(groups, dtype=np.int64)


def _assert_subject_partitioning(
    split: object, labels: np.ndarray, groups: np.ndarray
) -> None:
    expected_group_counts = {"train": 15, "tune": 5, "calibration": 5, "test": 5}
    observed_group_sets = {}
    for name, expected_count in expected_group_counts.items():
        indices = getattr(split, name)
        np.testing.assert_array_equal(np.unique(labels[indices]), np.arange(6))
        subject_ids = set(np.unique(groups[indices]).tolist())
        assert len(subject_ids) == expected_count
        observed_group_sets[name] = subject_ids

    names = tuple(observed_group_sets)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            assert observed_group_sets[left_name].isdisjoint(
                observed_group_sets[right_name]
            )
    assert set.union(*observed_group_sets.values()) == set(range(1, 31))


def test_group_four_way_split_is_subject_disjoint_and_reproducible() -> None:
    labels, groups = _grouped_fixture()
    first = make_group_four_way_split(labels, groups, (15, 5, 5, 5), seed=45)
    second = make_group_four_way_split(labels, groups, (15, 5, 5, 5), seed=45)

    for name in ("train", "tune", "calibration", "test"):
        assert np.array_equal(getattr(first, name), getattr(second, name))

    _assert_subject_partitioning(first, labels, groups)


def test_experiment_split_routes_group_config_to_subject_disjoint_split() -> None:
    labels, groups = _grouped_fixture()
    dataset = SimpleNamespace(labels=labels, groups=groups)
    config = {
        "seed": 45,
        "split": {
            "unit": "groups",
            "train": 15,
            "tune": 5,
            "calibration": 5,
            "test": 5,
            "candidate_permutations": 128,
        },
    }

    split = experiment_split(config, dataset)

    _assert_subject_partitioning(split, labels, groups)


def test_group_four_way_split_rejects_invalid_group_budget() -> None:
    labels, groups = _grouped_fixture()
    try:
        make_group_four_way_split(labels, groups, (14, 5, 5, 5), seed=45)
    except ValueError as error:
        assert "sum to the number of groups" in str(error)
    else:
        raise AssertionError("invalid group budget was accepted")
