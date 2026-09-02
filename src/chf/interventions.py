from typing import Literal

import numpy as np


MaskingMethod = Literal["mean_mask", "permutation"]


def remove_feature(features: np.ndarray, feature_index: int) -> np.ndarray:
    """Return a copy with one feature column removed."""
    values = _validated_features(features, feature_index)
    return np.delete(values, feature_index, axis=1)


def intervene_feature(
    features: np.ndarray,
    feature_index: int,
    method: MaskingMethod,
    *,
    training_mean: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply one fixed-model feature intervention without modifying the input.

    Mean masking uses a statistic learned from training data only. Permutation
    shuffles within the supplied partition, so values never cross split boundaries.
    """
    values = _validated_features(features, feature_index)
    result = values.copy()
    if method == "mean_mask":
        if training_mean is None or not np.isfinite(training_mean):
            raise ValueError("mean_mask requires a finite training_mean")
        result[:, feature_index] = float(training_mean)
    elif method == "permutation":
        if rng is None:
            raise ValueError("permutation requires an explicit rng")
        result[:, feature_index] = rng.permutation(result[:, feature_index])
    else:
        raise ValueError("method must be 'mean_mask' or 'permutation'")
    return result


def _validated_features(features: np.ndarray, feature_index: int) -> np.ndarray:
    values = np.asarray(features)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("features must be a non-empty two-dimensional array")
    if not isinstance(feature_index, (int, np.integer)) or not (
        0 <= feature_index < values.shape[1]
    ):
        raise ValueError("feature_index is outside the feature matrix")
    return values
