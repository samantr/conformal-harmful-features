from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from chf.experiments.statistics import exact_sign_flip_pvalue


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "19_phase8_precision_extension_aggregate.py"
RUNNER = ROOT / "experiments" / "18_phase8_precision_extension.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extension_spec_changes_only_the_seed_batch_and_execution_metadata():
    runner = _load(RUNNER, "phase8_precision_extension_runner_test")
    spec = runner.validate_extension_design(
        extension_spec_path=ROOT / "configs" / "phase8_robustness_extension_seeds.yaml",
        repository_root=ROOT,
    )
    assert spec["seeds"] == list(range(53, 63))
    assert set(spec["precision_extension"]["extended_datasets"]) == {
        "dry_bean",
        "human_activity_recognition",
    }
    assert spec["precision_extension"]["stopped_at_10_seeds"] == ["covertype"]


def test_meet_in_the_middle_exact_sign_flip_matches_frozen_enumeration():
    aggregate = _load(SCRIPT, "phase8_precision_extension_aggregate_test")
    rng = np.random.default_rng(20260908)
    cases = [
        np.array([1.0, -0.5]),
        np.array([0.1, 0.1, -0.2, 0.4]),
        np.zeros(6),
        rng.normal(size=8),
        rng.normal(size=10),
    ]
    for values in cases:
        expected = exact_sign_flip_pvalue(values)
        actual = aggregate.exact_sign_flip_pvalue_mitm(values)
        assert actual == expected


def test_meet_in_the_middle_supports_twenty_pairs_without_approximation():
    aggregate = _load(SCRIPT, "phase8_precision_extension_aggregate_20_test")
    values = np.linspace(-0.017, 0.023, 20)
    pvalue = aggregate.exact_sign_flip_pvalue_mitm(values)
    assert 0.0 <= pvalue <= 1.0
    # Every exact p-value is an integer multiple of 1 / 2^20.
    scaled = pvalue * (1 << 20)
    assert abs(scaled - round(scaled)) < 1e-9
