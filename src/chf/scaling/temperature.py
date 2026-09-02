from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.model_selection import train_test_split

from chf.conformal import ScoreName, finite_sample_quantile, nonrandomized_score_matrix

BASE_TEMPERATURE = 1.0


@dataclass(frozen=True)
class ProbabilityDiagnostics:
    zero_count: int
    zero_fraction: float
    exactly_one_count: int
    exactly_one_fraction: float
    mean_max_probability: float


@dataclass(frozen=True)
class TemperatureTuningResult:
    temperature: float
    nll: float


@dataclass(frozen=True)
class ConfTSTuningResult:
    temperature: float
    loss: float
    tuning_threshold: float
    loss_indices: np.ndarray
    threshold_indices: np.ndarray
    rejected_temperatures: tuple[float, ...]


def _validate_logits_and_labels(
    logits: np.ndarray, labels: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("logits must have shape (n_samples, n_classes>=2)")
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("logits must be finite")
    if labels is None:
        return values, None
    label_values = np.asarray(labels)
    if label_values.shape != (values.shape[0],) or not np.issubdtype(
        label_values.dtype, np.integer
    ):
        raise ValueError("labels must be a one-dimensional integer array")
    if np.any((label_values < 0) | (label_values >= values.shape[1])):
        raise ValueError("labels contain a class outside the logits matrix")
    return values, label_values.astype(np.int64, copy=False)


def _temperature_grid(temperatures: Iterable[float]) -> tuple[float, ...]:
    candidates = tuple(float(value) for value in temperatures)
    if not candidates or any(not np.isfinite(value) or value <= 0 for value in candidates):
        raise ValueError("temperatures must be a non-empty sequence of positive values")
    return candidates


def probabilities_from_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling with a max-shifted softmax.

    Floating-point precision is preserved intentionally so diagnostics can detect
    float32 underflow exactly as in the frozen reproduction scripts.
    """
    values, _ = _validate_logits_and_labels(logits)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    scaled = values / float(temperature)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exponentials = np.exp(scaled)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def probability_diagnostics(probabilities: np.ndarray) -> ProbabilityDiagnostics:
    values = np.asarray(probabilities)
    if values.ndim != 2 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("probabilities must be a finite two-dimensional array")
    maxima = values.max(axis=1)
    zero_count = int(np.count_nonzero(values == 0))
    one_count = int(np.count_nonzero(maxima == 1))
    return ProbabilityDiagnostics(
        zero_count=zero_count,
        zero_fraction=zero_count / values.size,
        exactly_one_count=one_count,
        exactly_one_fraction=one_count / len(values),
        mean_max_probability=float(maxima.mean()),
    )


def tune_temperature(
    logits_tune: np.ndarray,
    labels_tune: np.ndarray,
    temperatures: Iterable[float],
) -> TemperatureTuningResult:
    """Choose ordinary temperature scaling by minimum tuning-set NLL."""
    logits, labels = _validate_logits_and_labels(logits_tune, labels_tune)
    assert labels is not None
    best: TemperatureTuningResult | None = None
    for temperature in _temperature_grid(temperatures):
        probabilities = probabilities_from_logits(logits, temperature)
        true_probabilities = probabilities[np.arange(len(labels)), labels]
        tiny = np.finfo(probabilities.dtype).tiny
        nll = float(-np.log(np.clip(true_probabilities, tiny, 1.0)).mean())
        candidate = TemperatureTuningResult(temperature=temperature, nll=nll)
        if best is None or candidate.nll < best.nll:
            best = candidate
    assert best is not None
    return best


def tune_confts(
    logits_tune: np.ndarray,
    labels_tune: np.ndarray,
    alpha: float,
    score_name: ScoreName,
    temperatures: Iterable[float],
    *,
    seed: int,
    threshold_fraction: float = 0.5,
    k_reg: int = 1,
    lambda_reg: float = 0.001,
    reject_zero_probabilities: bool = True,
) -> ConfTSTuningResult:
    """Tune ConfTS using two disjoint subsets of the tuning partition.

    The threshold subset estimates tau(T); the loss subset minimizes the mean
    squared efficiency gap. Final conformal calibration is never used here.
    """
    logits, labels = _validate_logits_and_labels(logits_tune, labels_tune)
    assert labels is not None
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    if not 0 < threshold_fraction < 1:
        raise ValueError("threshold_fraction must lie strictly between zero and one")

    all_indices = np.arange(len(labels))
    loss_indices, threshold_indices = train_test_split(
        all_indices,
        test_size=threshold_fraction,
        random_state=seed,
        stratify=labels,
    )
    candidates = _temperature_grid(temperatures)
    rejected: list[float] = []
    best_temperature: float | None = None
    best_loss = float("inf")
    best_threshold = float("nan")

    for temperature in candidates:
        probabilities = probabilities_from_logits(logits, temperature)
        diagnostics = probability_diagnostics(probabilities)
        if reject_zero_probabilities and diagnostics.zero_count:
            rejected.append(temperature)
            continue

        threshold_scores = nonrandomized_score_matrix(
            probabilities[threshold_indices],
            score_name,
            k_reg=k_reg,
            lambda_reg=lambda_reg,
        )
        threshold_true_scores = threshold_scores[
            np.arange(len(threshold_indices)), labels[threshold_indices]
        ]
        tau = finite_sample_quantile(threshold_true_scores, alpha)

        loss_scores = nonrandomized_score_matrix(
            probabilities[loss_indices],
            score_name,
            k_reg=k_reg,
            lambda_reg=lambda_reg,
        )
        loss_true_scores = loss_scores[
            np.arange(len(loss_indices)), labels[loss_indices]
        ]
        loss = float(np.mean((tau - loss_true_scores) ** 2))
        if loss < best_loss:
            best_temperature = temperature
            best_loss = loss
            best_threshold = tau

    if best_temperature is None:
        raise ValueError("all candidate temperatures failed the numerical safety rule")
    return ConfTSTuningResult(
        temperature=best_temperature,
        loss=best_loss,
        tuning_threshold=best_threshold,
        loss_indices=np.asarray(loss_indices),
        threshold_indices=np.asarray(threshold_indices),
        rejected_temperatures=tuple(rejected),
    )
