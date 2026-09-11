import importlib.util
import json
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "14_phase8_recover_v2.py"
SPEC = importlib.util.spec_from_file_location("phase8_recovery_control_v2", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
recovery_v2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery_v2)


def test_original_selection_digest_is_reused_without_relaxing_other_manifest_fields(tmp_path) -> None:
    unit = tmp_path / "covertype" / "seed_43" / "checkpoints" / "final_grid"
    unit.mkdir(parents=True)
    original_hash = "a" * 64
    manifest = {
        "schema_version": 1,
        "stage": "phase8_final_grid",
        "experiment_name": "phase8_covertype_seed_43",
        "seed": 43,
        "selection_sha256": original_hash,
        "config_sha256": "b" * 64,
        "split_id": "split",
        "selection_data_id": "selection",
        "code_version": "e4b3645",
        "phase8_grid": [],
    }
    (unit / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    loaded = recovery_v2.install_original_selection_digest(
        output_root=tmp_path, dataset="covertype", seed=43
    )

    assert loaded == manifest
    assert recovery_v2.robustness._selection_digest(pd.DataFrame({"x": [1]})) == original_hash


def test_original_selection_digest_rejects_wrong_unit_identity(tmp_path) -> None:
    unit = tmp_path / "covertype" / "seed_43" / "checkpoints" / "final_grid"
    unit.mkdir(parents=True)
    (unit / "manifest.json").write_text(
        json.dumps({
            "schema_version": 1,
            "stage": "phase8_final_grid",
            "experiment_name": "phase8_covertype_seed_50",
            "seed": 50,
            "selection_sha256": "a" * 64,
        }),
        encoding="utf-8",
    )

    try:
        recovery_v2.install_original_selection_digest(
            output_root=tmp_path, dataset="covertype", seed=43
        )
    except ValueError as exc:
        assert "experiment_name" in str(exc) or "seed" in str(exc)
    else:
        raise AssertionError("wrong-unit manifest must be rejected")
