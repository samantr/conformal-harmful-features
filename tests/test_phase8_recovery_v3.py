import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "15_phase8_recover_v3.py"
SPEC = importlib.util.spec_from_file_location("phase8_recovery_control_v3", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
v3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v3)


def test_saved_digest_is_optional_but_wrong_identity_remains_rejected(tmp_path) -> None:
    reused, digest = v3._preserve_saved_selection_digest(
        output_root=tmp_path, dataset="covertype", seed=43
    )
    assert reused is False
    assert digest is None

    root = tmp_path / "covertype" / "seed_43" / "checkpoints" / "final_grid"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({
            "schema_version": 1,
            "stage": "phase8_final_grid",
            "experiment_name": "phase8_covertype_seed_50",
            "seed": 50,
            "selection_sha256": "a" * 64,
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="experiment_name|seed"):
        v3._preserve_saved_selection_digest(
            output_root=tmp_path, dataset="covertype", seed=43
        )


def test_diagnostic_retention_keeps_raw_counts_and_marks_scaling_excess(tmp_path, monkeypatch) -> None:
    class V1:
        @staticmethod
        def _boundary_excess_frame(results):
            excess = results.copy(deep=True)
            excess.loc[:, v3.BOUNDARY_COLUMNS] = 0
            # Only the ConfTS test exact-one event is scaling-induced.
            excess.loc[excess["scaling"].eq("confts"), "test_exactly_one_count"] = 3
            return excess, {"rule": "test"}

    captured = {}

    def structural_validator(*, results, subject_results, selections, spec, output_dir, manifest):
        # v3 may neutralize only numerical columns before this call.
        assert (results[v3.BOUNDARY_COLUMNS] == 0).all().all()
        captured["called"] = True
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "phase8_grid_protocol.json").write_text(
            json.dumps({
                "status": "PASS",
                "checks": {
                    "result_keys_unique": True,
                    "no_zero_or_saturated_probabilities": True,
                },
            }),
            encoding="utf-8",
        )
        results.to_csv(output_dir / "phase8_results.csv", index=False)

    monkeypatch.setattr(v3.robustness, "_validate_and_write_grid", structural_validator)
    v3._install_diagnostic_retention_validator(V1)

    rows = []
    for scaling, exact_one in (("base", 1), ("confts", 4)):
        row = {
            "scaling": scaling,
            "calibration_zero_count": 0,
            "calibration_exactly_one_count": 0,
            "test_zero_count": 0,
            "test_exactly_one_count": exact_one,
        }
        rows.append(row)
    raw = pd.DataFrame(rows)

    v3.robustness._validate_and_write_grid(
        results=raw,
        subject_results=pd.DataFrame(),
        selections=pd.DataFrame(),
        spec={},
        output_dir=tmp_path / "out",
        manifest={},
    )

    assert captured["called"] is True
    saved = pd.read_csv(tmp_path / "out" / "phase8_results.csv")
    assert saved["test_exactly_one_count"].tolist() == [1, 4]
    assert saved["numerical_boundary_flag"].astype(bool).tolist() == [True, True]
    assert saved["scaling_induced_boundary_count"].tolist() == [0, 3]
    protocol = json.loads(
        (tmp_path / "out" / "phase8_grid_protocol.json").read_text(encoding="utf-8")
    )
    assert protocol["status"] == "COMPLETE_WITH_NUMERICAL_FLAGS"
    assert protocol["checks"]["structural_validation_passed"] is True
    assert protocol["checks"]["original_strict_numerical_safety"] is False
    assert protocol["checks"]["no_scaling_induced_boundary_probabilities"] is False
