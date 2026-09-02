import numpy as np
import pytest

from chf.conformal import calibrate_threshold, evaluate_prediction_sets


def test_calibration_and_evaluation_use_explicit_randomization() -> None:
    probabilities = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.6, 0.3],
            [0.2, 0.3, 0.5],
            [0.6, 0.25, 0.15],
            [0.15, 0.55, 0.3],
        ]
    )
    labels = np.array([0, 1, 2, 0, 1])
    uniforms = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

    tau = calibrate_threshold(
        probabilities, labels, alpha=0.4, score_name="aps", u_values=uniforms
    )
    result = evaluate_prediction_sets(
        probabilities, labels, tau, "aps", u_values=uniforms
    )

    # True-label scores are [0.35, 0.30, 0.25, 0.30, 0.275].
    # ceil((5 + 1) * (1 - 0.4)) = 4, so the fourth order statistic is 0.30.
    assert tau == pytest.approx(0.30)
    assert result.coverage == 0.8
    assert result.mean_size == 0.8
    assert result.empty_set_rate == 0.2
    assert result.full_set_rate == 0.0


def test_randomization_source_is_required() -> None:
    probabilities = np.array([[0.6, 0.4]])
    labels = np.array([0])
    with pytest.raises(ValueError, match="provide rng"):
        calibrate_threshold(probabilities, labels, 0.1, "aps")
