import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

from chf.experiments.phase8_runner import phase8_run_plan
from chf.experiments.protocol import code_version
from chf.experiments.robustness import (
    derive_seed_config,
    phase8_accuracy_loss_choices,
    phase8_grid_rows,
    validate_phase8_spec,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _spec() -> dict:
    return yaml.safe_load(
        (REPOSITORY_ROOT / "configs" / "phase8_robustness.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_frozen_phase8_plan_has_ten_paired_seeds_and_sixty_grid_cells() -> None:
    spec = _spec()

    validate_phase8_spec(spec)
    plan = phase8_run_plan(spec, REPOSITORY_ROOT)
    grid = pd.DataFrame(phase8_grid_rows(spec))

    assert len(plan) == 30
    assert plan.groupby("dataset_key")["seed"].nunique().eq(10).all()
    assert plan.loc[
        plan["dataset_key"] == "human_activity_recognition", "split_unit"
    ].eq("groups").all()
    assert len(grid) == 60
    assert len(grid.loc[grid["score"] == "aps"]) == 6
    assert len(grid.loc[grid["score"] == "raps"]) == 54


def test_seed_derivation_does_not_mutate_frozen_phase7_config() -> None:
    spec = _spec()
    base_path = REPOSITORY_ROOT / "configs" / "human_activity_recognition.yaml"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    before = json.dumps(base, sort_keys=True)

    derived = derive_seed_config(
        base,
        spec,
        dataset_key="human_activity_recognition",
        seed=52,
        resume=True,
    )

    assert json.dumps(base, sort_keys=True) == before
    assert derived["seed"] == 52
    assert derived["split"]["unit"] == "groups"
    assert derived["progressive_selection"]["max_removals"] == 5
    assert derived["progressive_selection"]["continue_path_after_stop"] is True
    assert derived["checkpointing"] == {"enabled": True, "resume": True}
    assert derived["selection"]["max_accuracy_loss"] == 0.01


def test_accuracy_loss_choices_are_made_from_tuning_path_only() -> None:
    rows = []
    for method in ("conformal_harm_one_shot", "conformal_harm_recursive"):
        for removed, loss, gain in (
            (0, 0.0, 0.0),
            (1, 0.004, 0.2),
            (2, 0.009, 0.3),
            (3, 0.018, 0.4),
        ):
            rows.append(
                {
                    "model": "small_neural_network",
                    "method": method,
                    "selected_indices": json.dumps(list(range(6 - removed))),
                    "selected_features": json.dumps(list(range(6 - removed))),
                    "n_features": 6 - removed,
                    "tuning_n_removed": removed,
                    "tuning_max_accuracy_loss": loss,
                    "tuning_max_coverage_shortfall": 0.0,
                    "tuning_efficiency_gain": gain,
                    "tuning_conditional_violation": 0.01,
                }
            )
    config = {"harm": {"constraints": {"max_coverage_shortfall": 0.03}}}

    choices = phase8_accuracy_loss_choices(pd.DataFrame(rows), _spec(), config)

    one_shot = choices.loc[
        choices["method"] == "conformal_harm_one_shot"
    ].set_index("accuracy_loss_limit")
    assert one_shot.loc[0.0, "n_removed"] == 0
    assert one_shot.loc[0.005, "n_removed"] == 1
    assert one_shot.loc[0.01, "n_removed"] == 2
    assert one_shot.loc[0.02, "n_removed"] == 3
    assert choices["choice_partition"].eq("outer_tune_crossfit_only").all()


def test_repository_hygiene_patterns_do_not_dirty_provenance(tmp_path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "phase8@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Phase 8 test"],
        cwd=tmp_path,
        check=True,
    )
    gitignore = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8")
    (tmp_path / ".gitignore").write_text(gitignore, encoding="utf-8")
    (tmp_path / "tracked.txt").write_text("frozen\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "frozen"], cwd=tmp_path, check=True)

    (tmp_path / ".idea").mkdir()
    (tmp_path / ".idea" / "workspace.xml").write_text("local", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "local.txt").write_text("local", encoding="utf-8")
    (tmp_path / "phase8-results.zip").write_bytes(b"local")

    assert not code_version(tmp_path).endswith("-dirty")
    (tmp_path / "tracked.txt").write_text("changed\n", encoding="utf-8")
    assert code_version(tmp_path).endswith("-dirty")


def test_phase8_spec_rejects_seed_or_grid_drift() -> None:
    spec = _spec()
    spec["seeds"] = spec["seeds"][:-1]
    with pytest.raises(ValueError, match="ten unique seeds"):
        validate_phase8_spec(spec)
