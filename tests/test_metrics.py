import numpy as np
import pytest

from chf.metrics import (
    classification_metrics,
    conditional_coverage_metrics,
    expected_calibration_error,
)


def test_classification_metrics_match_hand_calculation() -> None:
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.4, 0.3, 0.3],
            [0.2, 0.2, 0.6],
        ]
    )
    labels = np.array([0, 1, 2, 2])
    result = classification_metrics(probabilities, labels, ece_bins=2)

    assert result.accuracy == 0.75
    assert result.macro_f1 == pytest.approx((2 / 3 + 1 + 2 / 3) / 3)
    assert result.nll == pytest.approx(-np.log([0.8, 0.7, 0.3, 0.6]).mean())
    assert result.ece == pytest.approx(
        expected_calibration_error(probabilities, labels, n_bins=2)
    )


def test_conditional_metrics_report_sscv_and_class_gap() -> None:
    included = np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
            [1, 0, 0],
        ],
        dtype=bool,
    )
    labels = np.array([0, 0, 1, 1, 2, 2])
    result = conditional_coverage_metrics(
        included,
        labels,
        target_coverage=0.75,
        min_sscv_stratum_size=1,
    )

    assert result.class_coverages == (1.0, 1.0, 0.5)
    assert result.class_coverage_gap == 0.5
    assert result.class_coverage_max_deviation == 0.25
    assert result.sscv == pytest.approx(0.25)
