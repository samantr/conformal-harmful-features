from types import SimpleNamespace

import numpy as np
import pytest

from chf.experiments import baselines, progressive_selection, real_datasets, scaling_interaction


class _ExperimentSplitReached(RuntimeError):
    pass


def _probe_dataset() -> SimpleNamespace:
    return SimpleNamespace(
        features=np.zeros((12, 3), dtype=np.float64),
        labels=np.tile(np.arange(3, dtype=np.int64), 4),
        groups=np.repeat(np.arange(1, 5, dtype=np.int64), 3),
    )


def _probe_config() -> dict[str, object]:
    return {
        "experiment_name": "har_split_probe",
        "phase": 7,
        "seed": 45,
        "dataset": {
            "n_samples": 12,
            "n_features": 3,
            "n_classes": 3,
        },
        "split": {
            "unit": "groups",
            "train": 1,
            "tune": 1,
            "calibration": 1,
            "test": 1,
        },
    }


@pytest.mark.parametrize(
    ("module", "runner_name"),
    [
        (progressive_selection, "run_progressive_selection"),
        (baselines, "run_required_baselines"),
        (scaling_interaction, "run_scaling_interaction"),
        (real_datasets, "run_real_dataset_benchmark"),
    ],
)
def test_phase4_to_phase7_entrypoints_delegate_outer_split_to_experiment_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    module,
    runner_name: str,
) -> None:
    dataset = _probe_dataset()
    config = _probe_config()

    monkeypatch.setattr(module, "dataset_from_config", lambda *_: dataset)

    def split_probe(config_arg, dataset_arg):
        assert config_arg is config
        assert dataset_arg is dataset
        assert config_arg["split"]["unit"] == "groups"
        raise _ExperimentSplitReached

    monkeypatch.setattr(module, "experiment_split", split_probe)

    with pytest.raises(_ExperimentSplitReached):
        getattr(module, runner_name)(config, tmp_path, tmp_path)
