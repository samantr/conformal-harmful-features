import pandas as pd
import pytest

from chf.experiments.protocol import REFERENCE_METRICS, attach_reference_deltas


def test_reference_deltas_pair_on_explicit_experimental_keys() -> None:
    reference_metrics = {metric: 0.5 for metric in REFERENCE_METRICS}
    intervention_metrics = dict(reference_metrics)
    intervention_metrics.update(accuracy=0.49, mean_size=0.45, coverage=0.89)
    rows = [
        {
            "model": "model",
            "scaling": "base",
            "score": "aps",
            "selection_resample": 2,
            "intervention": "reference",
            "alpha": 0.1,
            **reference_metrics,
        },
        {
            "model": "model",
            "scaling": "base",
            "score": "aps",
            "selection_resample": 2,
            "intervention": "retrain_ablation",
            "alpha": 0.1,
            **intervention_metrics,
        },
    ]

    result = attach_reference_deltas(
        pd.DataFrame(rows),
        key_columns=("model", "scaling", "score", "selection_resample"),
    )
    intervention = result.loc[result["intervention"] == "retrain_ablation"].iloc[0]

    assert intervention["accuracy_loss"] == pytest.approx(0.01)
    assert intervention["mean_size_reduction"] == pytest.approx(0.05)
    assert intervention["coverage_target_deviation"] == pytest.approx(0.01)
