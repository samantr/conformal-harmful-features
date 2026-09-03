import numpy as np

from chf.data import make_group_four_way_split


def _grouped_fixture() -> tuple[np.ndarray, np.ndarray]:
    labels = []
    groups = []
    for subject in range(1, 31):
        for activity in range(6):
            labels.extend([activity, activity])
            groups.extend([subject, subject])
    return np.asarray(labels, dtype=np.int64), np.asarray(groups, dtype=np.int64)


def test_group_four_way_split_is_subject_disjoint_and_reproducible() -> None:
    labels, groups = _grouped_fixture()
    first = make_group_four_way_split(labels, groups, (15, 5, 5, 5), seed=45)
    second = make_group_four_way_split(labels, groups, (15, 5, 5, 5), seed=45)

    for name in ("train", "tune", "calibration", "test"):
        assert np.array_equal(getattr(first, name), getattr(second, name))
        assert np.array_equal(np.unique(labels[getattr(first, name)]), np.arange(6))

    expected_group_counts = {"train": 15, "tune": 5, "calibration": 5, "test": 5}
    observed_group_sets = {}
    for name, expected_count in expected_group_counts.items():
        subject_ids = set(np.unique(groups[getattr(first, name)]).tolist())
        assert len(subject_ids) == expected_count
        observed_group_sets[name] = subject_ids

    names = tuple(observed_group_sets)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            assert observed_group_sets[left_name].isdisjoint(
                observed_group_sets[right_name]
            )
    assert set.union(*observed_group_sets.values()) == set(range(1, 31))


def test_group_four_way_split_rejects_invalid_group_budget() -> None:
    labels, groups = _grouped_fixture()
    try:
        make_group_four_way_split(labels, groups, (14, 5, 5, 5), seed=45)
    except ValueError as error:
        assert "sum to the number of groups" in str(error)
    else:
        raise AssertionError("invalid group budget was accepted")
