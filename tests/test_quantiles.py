import numpy as np
import pytest

from chf.conformal import finite_sample_quantile


def test_finite_sample_quantile_uses_conformal_rank() -> None:
    scores = np.arange(1, 11, dtype=float) / 10
    assert finite_sample_quantile(scores, alpha=0.2) == pytest.approx(0.9)


def test_finite_sample_quantile_caps_rank_at_n() -> None:
    assert finite_sample_quantile(np.array([0.1, 0.4, 0.2]), alpha=0.1) == 0.4

