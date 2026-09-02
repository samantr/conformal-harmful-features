import numpy as np
import pytest

from chf.interventions import intervene_feature, remove_feature


def test_remove_feature_drops_only_the_requested_column() -> None:
    features = np.arange(20, dtype=float).reshape(5, 4)
    expected = features[:, [0, 2, 3]]

    actual = remove_feature(features, 1)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(features, np.arange(20, dtype=float).reshape(5, 4))


def test_mean_mask_uses_training_statistic_without_mutating_partition() -> None:
    features = np.arange(12, dtype=float).reshape(4, 3)
    original = features.copy()

    masked = intervene_feature(features, 1, "mean_mask", training_mean=7.5)

    np.testing.assert_array_equal(masked[:, 1], np.full(4, 7.5))
    np.testing.assert_array_equal(masked[:, [0, 2]], original[:, [0, 2]])
    np.testing.assert_array_equal(features, original)


def test_permutation_is_partition_local_and_reproducible() -> None:
    features = np.arange(24, dtype=float).reshape(8, 3)
    first = intervene_feature(
        features, 2, "permutation", rng=np.random.default_rng(91)
    )
    second = intervene_feature(
        features, 2, "permutation", rng=np.random.default_rng(91)
    )

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.sort(first[:, 2]), np.sort(features[:, 2]))
    np.testing.assert_array_equal(first[:, :2], features[:, :2])


def test_interventions_require_explicit_safe_inputs() -> None:
    features = np.ones((4, 2))
    with pytest.raises(ValueError, match="training_mean"):
        intervene_feature(features, 0, "mean_mask")
    with pytest.raises(ValueError, match="explicit rng"):
        intervene_feature(features, 0, "permutation")
    with pytest.raises(ValueError, match="outside"):
        remove_feature(features, 2)
