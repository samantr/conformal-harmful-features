from typing import Literal

import numpy as np


ScoreName = Literal["aps", "raps"]


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("probabilities must have shape (n_samples, n_classes>=2)")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("each probability row must sum to one")
    return values


def _validate_raps_parameters(k_reg: int, lambda_reg: float) -> None:
    if not isinstance(k_reg, (int, np.integer)) or k_reg < 0:
        raise ValueError("k_reg must be a non-negative integer")
    if not np.isfinite(lambda_reg) or lambda_reg < 0:
        raise ValueError("lambda_reg must be finite and non-negative")


def _sorted_components(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Stable sorting gives deterministic behavior when classes have equal probability.
    order = np.argsort(-probabilities, axis=1, kind="stable")
    sorted_probabilities = np.take_along_axis(probabilities, order, axis=1)
    return order, sorted_probabilities


def nonrandomized_score_matrix(
    probabilities: np.ndarray,
    score_name: ScoreName = "aps",
    *,
    k_reg: int = 1,
    lambda_reg: float = 0.001,
) -> np.ndarray:
    """Return non-randomized APS/RAPS scores for every candidate label.

    Non-randomized scores include the full probability of the candidate class.
    They are used for ConfTS tuning, not for final randomized calibration.
    """
    values = _validate_probabilities(probabilities)
    if score_name not in ("aps", "raps"):
        raise ValueError("score_name must be 'aps' or 'raps'")
    _validate_raps_parameters(k_reg, lambda_reg)

    order, sorted_probabilities = _sorted_components(values)
    sorted_scores = np.cumsum(sorted_probabilities, axis=1)
    if score_name == "raps":
        ranks = np.arange(1, values.shape[1] + 1)
        penalties = lambda_reg * np.maximum(ranks - k_reg, 0)
        sorted_scores = sorted_scores + penalties[None, :]

    scores = np.empty_like(values)
    np.put_along_axis(scores, order, sorted_scores, axis=1)
    return scores


def randomized_score_matrix(
    probabilities: np.ndarray,
    u_values: np.ndarray,
    score_name: ScoreName = "aps",
    *,
    k_reg: int = 1,
    lambda_reg: float = 0.001,
) -> np.ndarray:
    """Return standard randomized APS/RAPS scores for every candidate label.

    One uniform random value is shared by all candidate labels for a sample:
    mass before candidate + u * candidate probability + optional RAPS penalty.
    """
    values = _validate_probabilities(probabilities)
    uniforms = np.asarray(u_values, dtype=float)
    if uniforms.shape != (values.shape[0],):
        raise ValueError("u_values must contain one value per sample")
    if not np.all(np.isfinite(uniforms)) or np.any((uniforms < 0) | (uniforms > 1)):
        raise ValueError("u_values must be finite and lie in [0, 1]")
    if score_name not in ("aps", "raps"):
        raise ValueError("score_name must be 'aps' or 'raps'")
    _validate_raps_parameters(k_reg, lambda_reg)

    order, sorted_probabilities = _sorted_components(values)
    cumulative_before = np.cumsum(sorted_probabilities, axis=1) - sorted_probabilities
    sorted_scores = cumulative_before + uniforms[:, None] * sorted_probabilities
    if score_name == "raps":
        ranks = np.arange(1, values.shape[1] + 1)
        penalties = lambda_reg * np.maximum(ranks - k_reg, 0)
        sorted_scores = sorted_scores + penalties[None, :]

    scores = np.empty_like(values)
    np.put_along_axis(scores, order, sorted_scores, axis=1)
    return scores
