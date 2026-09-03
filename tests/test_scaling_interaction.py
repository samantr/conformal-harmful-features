import json

import pandas as pd
import pytest

from chf.experiments.scaling_interaction import (
    interaction_decomposition,
    scaling_rank_stability,
)


def _result_row(
    method: str,
    scaling: str,
    mean_size: float,
    *,
    target_size: int,
) -> dict[str, object]:
    return {
        "model": "model",
        "method": method,
        "target_size": target_size,
        "selection_type": (
            "all_features" if method == "all_features" else "standard_selection"
        ),
        "scaling": scaling,
        "score": "aps",
        "mean_size": mean_size,
        "temperature": {"base": 1.0, "ts": 1.5, "confts": 0.5}[scaling],
        "threshold": 0.9,
        "test_mean_max_probability": 0.7,
        "test_mean_entropy": 0.8,
    }


def test_interaction_decomposition_matches_difference_in_differences() -> None:
    rows = [
        _result_row("all_features", "base", 4.0, target_size=20),
        _result_row("all_features", "ts", 5.0, target_size=20),
        _result_row("all_features", "confts", 3.0, target_size=20),
        _result_row("selected", "base", 3.5, target_size=15),
        _result_row("selected", "ts", 4.2, target_size=15),
        _result_row("selected", "confts", 2.8, target_size=15),
    ]

    result = interaction_decomposition(pd.DataFrame(rows), tolerance=0.01)
    ts = result.loc[result["scaling"] == "ts"].iloc[0]
    confts = result.loc[result["scaling"] == "confts"].iloc[0]

    assert ts["feature_gain_at_base"] == pytest.approx(0.5)
    assert ts["scaling_gain_on_all_features"] == pytest.approx(-1.0)
    assert ts["expected_additive_gain"] == pytest.approx(-0.5)
    assert ts["observed_joint_gain"] == pytest.approx(-0.2)
    assert ts["interaction_size_gain"] == pytest.approx(0.3)
    assert ts["interaction_label"] == "synergistic"
    assert confts["interaction_size_gain"] == pytest.approx(-0.3)
    assert confts["interaction_label"] == "redundant_or_antagonistic"


def test_scaling_rank_stability_detects_reversal() -> None:
    rows = []
    sizes = {
        "method_a": {"base": 1.0, "ts": 2.0, "confts": 1.0},
        "method_b": {"base": 2.0, "ts": 1.0, "confts": 2.0},
    }
    for method, by_scaling in sizes.items():
        for scaling, mean_size in by_scaling.items():
            rows.append(
                _result_row(method, scaling, mean_size, target_size=15)
            )

    result = scaling_rank_stability(pd.DataFrame(rows))
    base_ts = result.loc[
        (result["first_scaling"] == "base")
        & (result["second_scaling"] == "ts")
    ].iloc[0]

    assert base_ts["spearman_rank_correlation"] == pytest.approx(-1.0)
    assert base_ts["methods_with_changed_rank"] == 2
    assert json.loads(base_ts["first_best_methods"]) == ["method_a"]
    assert json.loads(base_ts["second_best_methods"]) == ["method_b"]
    assert bool(base_ts["best_method_changed"])
