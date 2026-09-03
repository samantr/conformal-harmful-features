import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from chf.conformal import finite_sample_quantile
from chf.scaling import (
    probabilities_from_logits,
    probability_diagnostics,
    tune_confts,
    tune_temperature,
)


def _legacy_aps(probabilities: np.ndarray) -> np.ndarray:
    order = np.argsort(-probabilities, axis=1)
    sorted_probabilities = np.take_along_axis(probabilities, order, axis=1)
    cumulative = np.cumsum(sorted_probabilities, axis=1)
    result = np.empty_like(probabilities)
    np.put_along_axis(result, order, cumulative, axis=1)
    return result


def test_temperature_softmax_matches_hand_calculation() -> None:
    probabilities = probabilities_from_logits(np.array([[3.0, 1.5, 0.5]]), 1.0)
    np.testing.assert_allclose(probabilities, [[0.76615721, 0.17095278, 0.06289001]])


def test_ordinary_temperature_scaling_minimizes_grid_nll() -> None:
    logits = np.array([[4.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 0.5]])
    labels = np.array([0, 1, 1, 0])
    grid = [0.5, 1.0, 2.0, 4.0]
    result = tune_temperature(logits, labels, grid)

    reference_losses = []
    for temperature in grid:
        probabilities = probabilities_from_logits(logits, temperature)
        reference_losses.append(
            -np.log(probabilities[np.arange(len(labels)), labels]).mean()
        )
    assert result.temperature == grid[int(np.argmin(reference_losses))]
    assert result.nll == pytest.approx(min(reference_losses))


def test_confts_grid_matches_frozen_reproduction_formula() -> None:
    rng = np.random.default_rng(42)
    logits = rng.normal(size=(60, 5))
    labels = np.repeat(np.arange(5), 12)
    grid = [0.5, 0.8, 1.0, 1.3]
    alpha = 0.1
    seed = 17

    result = tune_confts(
        logits,
        labels,
        alpha,
        "aps",
        grid,
        seed=seed,
        reject_zero_probabilities=True,
    )

    indices = np.arange(len(labels))
    loss_indices, threshold_indices = train_test_split(
        indices, test_size=0.5, random_state=seed, stratify=labels
    )
    reference = []
    for temperature in grid:
        probabilities = probabilities_from_logits(logits, temperature)
        threshold_matrix = _legacy_aps(probabilities[threshold_indices])
        threshold_scores = threshold_matrix[
            np.arange(len(threshold_indices)), labels[threshold_indices]
        ]
        tau = finite_sample_quantile(threshold_scores, alpha)
        loss_matrix = _legacy_aps(probabilities[loss_indices])
        loss_scores = loss_matrix[np.arange(len(loss_indices)), labels[loss_indices]]
        reference.append((np.mean((tau - loss_scores) ** 2), temperature, tau))
    expected_loss, expected_temperature, expected_tau = min(reference)

    assert result.temperature == expected_temperature
    assert result.loss == pytest.approx(expected_loss, rel=1e-12, abs=1e-12)
    assert result.tuning_threshold == pytest.approx(expected_tau)
    np.testing.assert_array_equal(result.loss_indices, loss_indices)
    np.testing.assert_array_equal(result.threshold_indices, threshold_indices)


def test_diagnostics_detect_float32_underflow_and_saturation() -> None:
    logits = np.array([[0.0, -1000.0, -2000.0]], dtype=np.float32)
    diagnostics = probability_diagnostics(probabilities_from_logits(logits, 1.0))
    assert diagnostics.zero_count == 2
    assert diagnostics.exactly_one_count == 1
    assert diagnostics.mean_max_probability == 1.0


def test_confts_can_reject_saturated_candidates_without_changing_default() -> None:
    logits = np.array(
        [[100.0, 0.0], [0.0, 100.0], [80.0, 0.0], [0.0, 80.0]]
    )
    labels = np.array([0, 1, 0, 1])
    result = tune_confts(
        logits,
        labels,
        0.1,
        "aps",
        [0.5, 10.0],
        seed=7,
        reject_zero_probabilities=True,
        reject_saturated_probabilities=True,
    )

    assert result.temperature == 10.0
    assert result.rejected_temperatures == (0.5,)
