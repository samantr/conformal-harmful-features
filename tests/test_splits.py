import numpy as np

from chf.data import make_four_way_split, stratified_subsample


def test_four_way_split_is_disjoint_complete_and_reproducible() -> None:
    labels = np.repeat(np.arange(10), 100)
    first = make_four_way_split(labels, (500, 200, 150, 150), seed=42)
    second = make_four_way_split(labels, (500, 200, 150, 150), seed=42)
    combined = np.concatenate([first.train, first.tune, first.calibration, first.test])
    assert len(np.unique(combined)) == len(labels)
    assert np.array_equal(first.train, second.train)
    first.assert_disjoint()


def test_stratified_subsample_is_deterministic_and_stays_inside_source() -> None:
    labels = np.repeat(np.arange(4), 25)
    source = np.arange(10, 90)

    first = stratified_subsample(source, labels, max_samples=40, seed=17)
    second = stratified_subsample(source, labels, max_samples=40, seed=17)

    np.testing.assert_array_equal(first, second)
    assert len(first) == 40
    assert np.isin(first, source).all()
    assert set(labels[first]) == set(labels[source])
