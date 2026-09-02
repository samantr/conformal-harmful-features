from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class ProgressiveCandidate:
    feature_index: int
    feature_name: str
    cumulative_efficiency_gain: float
    incremental_efficiency_gain: float
    max_accuracy_loss: float
    max_coverage_shortfall: float
    mean_conditional_violation: float
    eligible: bool


def rank_progressive_candidates(
    candidates: Iterable[ProgressiveCandidate],
) -> list[ProgressiveCandidate]:
    """Order candidates using the constrained-efficiency Phase 3 rule.

    Eligible candidates come first. Within that validity gate, recursive
    selection maximizes the incremental reduction in prediction-set size.
    Deterministic safeguards make equal or ineligible candidates reproducible.
    """
    return sorted(
        candidates,
        key=lambda candidate: (
            not candidate.eligible,
            -candidate.incremental_efficiency_gain,
            candidate.max_accuracy_loss,
            candidate.max_coverage_shortfall,
            candidate.mean_conditional_violation,
            candidate.feature_name,
        ),
    )


def choose_progressive_step(
    candidates: Iterable[ProgressiveCandidate],
    *,
    require_positive_gain: bool = True,
) -> ProgressiveCandidate | None:
    """Return the best safe next removal, or ``None`` when selection stops."""
    ordered = rank_progressive_candidates(candidates)
    if not ordered or not ordered[0].eligible:
        return None
    if require_positive_gain and ordered[0].incremental_efficiency_gain <= 0:
        return None
    return ordered[0]


def choose_subset_size(
    steps: Iterable[Mapping[str, float | int | bool]],
) -> int:
    """Choose a frozen path step from tuning evidence only.

    Among valid steps, maximize cumulative efficiency, then prefer lower
    conditional violation and fewer removals. Step zero (all features) should be
    supplied by callers so the selector can safely choose no intervention.
    """
    values = list(steps)
    eligible = [row for row in values if bool(row["eligible"])]
    if not eligible:
        raise ValueError("at least one eligible progressive step is required")
    selected = min(
        eligible,
        key=lambda row: (
            -float(row["cumulative_efficiency_gain"]),
            float(row["mean_conditional_violation"]),
            int(row["n_removed"]),
        ),
    )
    return int(selected["step"])


def non_dominated_steps(
    accuracy_loss: np.ndarray,
    mean_size: np.ndarray,
    conditional_violation: np.ndarray,
) -> np.ndarray:
    """Return a mask for the accuracy/size/conditional Pareto frontier."""
    objectives = np.column_stack(
        (
            np.asarray(accuracy_loss, dtype=float),
            np.asarray(mean_size, dtype=float),
            np.asarray(conditional_violation, dtype=float),
        )
    )
    if objectives.ndim != 2 or objectives.shape[1] != 3 or len(objectives) == 0:
        raise ValueError("progressive objectives must be non-empty vectors")
    if not np.isfinite(objectives).all():
        raise ValueError("progressive objectives must be finite")
    frontier = np.ones(len(objectives), dtype=bool)
    for index, candidate in enumerate(objectives):
        frontier[index] = not np.any(
            np.all(objectives <= candidate, axis=1)
            & np.any(objectives < candidate, axis=1)
        )
    return frontier
