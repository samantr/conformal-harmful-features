import numpy as np

from chf.data import make_four_way_split


def test_four_way_split_is_disjoint_complete_and_reproducible() -> None:
    labels = np.repeat(np.arange(10), 100)
    first = make_four_way_split(labels, (500, 200, 150, 150), seed=42)
    second = make_four_way_split(labels, (500, 200, 150, 150), seed=42)
    combined = np.concatenate([first.train, first.tune, first.calibration, first.test])
    assert len(np.unique(combined)) == len(labels)
    assert np.array_equal(first.train, second.train)
    first.assert_disjoint()

