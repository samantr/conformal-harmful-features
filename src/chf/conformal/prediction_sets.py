from dataclasses import dataclass

import numpy as np

from .quantiles import finite_sample_quantile
from .scores import ScoreName, randomized_score_matrix


@dataclass(frozen=True)
class PredictionSetEvaluation:
    coverage: float
    mean_size: float
    median_size: float
    size_p90: float
    empty_set_rate: float
    full_set_rate: float
    included: np.ndarray


def _uniforms(
    n_samples: int,
    rng: np.random.Generator | None,
    u_values: np.ndarray | None,
) -> np.ndarray:
    if u_values is not None:
        uniforms = np.asarray(u_values, dtype=float)
    elif rng is not None:
        uniforms = rng.random(n_samples)
    else:
        raise ValueError("provide rng or explicit u_values for randomized prediction sets")
    if uniforms.shape != (n_samples,):
        raise ValueError("u_values must contain one value per sample")
    return uniforms


def _validate_labels(labels: np.ndarray, n_samples: int, n_classes: int) -> np.ndarray:
    values = np.asarray(labels)
    if values.shape != (n_samples,) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("labels must be a one-dimensional integer array")
    if np.any((values < 0) | (values >= n_classes)):
        raise ValueError("labels contain a class outside the probability matrix")
    return values.astype(np.int64, copy=False)


def calibrate_threshold(
    probability_calibration: np.ndarray,
    labels_calibration: np.ndarray,
    alpha: float,
    score_name: ScoreName,
    rng: np.random.Generator | None = None,
    *,
    u_values: np.ndarray | None = None,
    k_reg: int = 1,
    lambda_reg: float = 0.001,
) -> float:
    """Calibrate a randomized APS/RAPS threshold on a held-out partition."""
    probabilities = np.asarray(probability_calibration)
    if probabilities.ndim != 2:
        raise ValueError("probability_calibration must be two-dimensional")
    labels = _validate_labels(labels_calibration, *probabilities.shape)
    uniforms = _uniforms(len(labels), rng, u_values)
    all_scores = randomized_score_matrix(
        probabilities,
        uniforms,
        score_name,
        k_reg=k_reg,
        lambda_reg=lambda_reg,
    )
    true_scores = all_scores[np.arange(len(labels)), labels]
    return finite_sample_quantile(true_scores, alpha)


def evaluate_prediction_sets(
    probability_test: np.ndarray,
    labels_test: np.ndarray,
    tau: float,
    score_name: ScoreName,
    rng: np.random.Generator | None = None,
    *,
    u_values: np.ndarray | None = None,
    k_reg: int = 1,
    lambda_reg: float = 0.001,
) -> PredictionSetEvaluation:
    """Generate randomized APS/RAPS sets and summarize their test behavior."""
    probabilities = np.asarray(probability_test)
    if probabilities.ndim != 2:
        raise ValueError("probability_test must be two-dimensional")
    if not np.isfinite(tau):
        raise ValueError("tau must be finite")
    labels = _validate_labels(labels_test, *probabilities.shape)
    uniforms = _uniforms(len(labels), rng, u_values)
    all_scores = randomized_score_matrix(
        probabilities,
        uniforms,
        score_name,
        k_reg=k_reg,
        lambda_reg=lambda_reg,
    )
    included = all_scores <= tau
    sizes = included.sum(axis=1)
    return PredictionSetEvaluation(
        coverage=float(included[np.arange(len(labels)), labels].mean()),
        mean_size=float(sizes.mean()),
        median_size=float(np.median(sizes)),
        size_p90=float(np.quantile(sizes, 0.9)),
        empty_set_rate=float(np.mean(sizes == 0)),
        full_set_rate=float(np.mean(sizes == probabilities.shape[1])),
        included=included,
    )
