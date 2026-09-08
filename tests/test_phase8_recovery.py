import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "13_phase8_recover.py"
SPEC = importlib.util.spec_from_file_location("phase8_recovery_control", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def _row(*, scaling: str, base_count: int) -> dict:
    return {
        "model": "small_neural_network",
        "method": "all_features",
        "target_size": 4,
        "selected_indices": "[0, 1, 2, 3]",
        "repetition": -1,
        "alpha": 0.1,
        "scaling": scaling,
        "score": "aps",
        "raps_lambda": np.nan,
        "raps_k_reg": np.nan,
        "calibration_zero_count": 0,
        "calibration_exactly_one_count": 0,
        "test_zero_count": 0,
        "test_exactly_one_count": base_count,
    }


def test_boundary_recovery_ignores_native_base_saturation_but_not_new_saturation() -> None:
    inherited = pd.DataFrame([
        _row(scaling="base", base_count=2),
        _row(scaling="confts", base_count=2),
    ])

    validation, diagnostics = recovery._boundary_excess_frame(inherited)

    assert (validation[recovery.BOUNDARY_COLUMNS] == 0).all().all()
    assert diagnostics["base_boundary_totals"]["test_exactly_one_count"] == 2

    induced = inherited.copy()
    induced.loc[induced["scaling"].eq("confts"), "test_exactly_one_count"] = 3
    validation, diagnostics = recovery._boundary_excess_frame(induced)

    tuned = validation.loc[validation["scaling"].eq("confts")]
    assert int(tuned.iloc[0]["test_exactly_one_count"]) == 1
    assert diagnostics["maximum_scaling_induced_excess"]["test_exactly_one_count"] == 1


def test_har_merge_compatibility_normalizes_only_subject_filter_keys() -> None:
    recovery._install_har_merge_compatibility()
    keys = ["alpha", "scaling", "score", "raps_lambda", "raps_k_reg"]
    left = pd.DataFrame({
        "alpha": [0.1],
        "scaling": ["base"],
        "score": ["aps"],
        "raps_lambda": pd.Series([np.nan], dtype="float64"),
        "raps_k_reg": pd.Series([np.nan], dtype="float64"),
        "subject_id": [7],
    })
    right = pd.DataFrame({
        "alpha": [0.1],
        "scaling": ["base"],
        "score": ["aps"],
        "raps_lambda": pd.Series([None], dtype="object"),
        "raps_k_reg": pd.Series([None], dtype="object"),
    })

    merged = left.merge(right, on=keys, how="inner")

    assert len(merged) == 1
    assert int(merged.iloc[0]["subject_id"]) == 7
