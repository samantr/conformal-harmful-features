import numpy as np

from chf.selection import (
    ProgressiveCandidate,
    choose_progressive_step,
    choose_subset_size,
    non_dominated_steps,
)


def _candidate(name: str, gain: float, *, eligible: bool = True):
    return ProgressiveCandidate(
        feature_index=0,
        feature_name=name,
        cumulative_efficiency_gain=gain,
        incremental_efficiency_gain=gain,
        max_accuracy_loss=0.001,
        max_coverage_shortfall=0.0,
        mean_conditional_violation=0.03,
        eligible=eligible,
    )


def test_progressive_choice_obeys_gate_and_positive_gain() -> None:
    chosen = choose_progressive_step(
        [_candidate("unsafe", 1.0, eligible=False), _candidate("safe", 0.2)]
    )
    assert chosen is not None and chosen.feature_name == "safe"
    assert choose_progressive_step([_candidate("negative", -0.1)]) is None


def test_subset_size_uses_tuning_efficiency_with_safe_tiebreaks() -> None:
    steps = [
        {"step": 0, "n_removed": 0, "eligible": True,
         "cumulative_efficiency_gain": 0.0, "mean_conditional_violation": 0.04},
        {"step": 1, "n_removed": 1, "eligible": True,
         "cumulative_efficiency_gain": 0.2, "mean_conditional_violation": 0.05},
        {"step": 2, "n_removed": 2, "eligible": True,
         "cumulative_efficiency_gain": 0.2, "mean_conditional_violation": 0.03},
        {"step": 3, "n_removed": 3, "eligible": False,
         "cumulative_efficiency_gain": 0.5, "mean_conditional_violation": 0.01},
    ]
    assert choose_subset_size(steps) == 2


def test_progressive_pareto_mask_preserves_tradeoffs() -> None:
    mask = non_dominated_steps(
        np.array([0.0, 0.01, 0.02, 0.03]),
        np.array([2.0, 1.7, 1.8, 2.1]),
        np.array([0.05, 0.06, 0.04, 0.08]),
    )
    np.testing.assert_array_equal(mask, [True, True, True, False])
