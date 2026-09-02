import math

import numpy as np


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile using ceil((n + 1)(1-alpha))/n."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    rank = math.ceil((values.size + 1) * (1 - alpha))
    rank = min(rank, values.size)
    return float(np.partition(values, rank - 1)[rank - 1])
