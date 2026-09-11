import json

import numpy as np
import pandas as pd
import pytest

from chf.experiments.coverage_validation import (
    grouped_coverage_metrics,
    validate_coverage_results,
)
from chf.experiments.scaling_interaction import subject_coverage_table
from chf.experiments.scaling_interaction import (
    _assert_recomputed_results_match,
    _validate_replay_config,
)


def _grouped_metrics(covered_counts: list[int]) -> dict[str, object]:
    samples_per_group = 10
    groups = np.repeat(np.arange(len(covered_counts)), samples_per_group)
    labels = np.zeros(len(groups), dtype=np.int64)
    included = np.zeros((len(groups), 2), dtype=bool)
    offset = 0
    for covered_count in covered_counts:
        included[offset : offset + covered_count, 0] = True
        offset += samples_per_group
    return grouped_coverage_metrics(
        included,
        labels,
        groups,
        target=0.9,
        confidence_level=0.95,
    )


def _group_config() -> dict[str, object]:
    return {
        "split": {"unit": "groups"},
        "conformal": {"alpha": 0.1},
        "coverage_validation": {"mode": "grouped", "confidence_level": 0.95},
    }


def _serialized_group_row(metrics: dict[str, object]) -> dict[str, object]:
    row = dict(metrics)
    for column in (
        "group_coverages",
        "group_covered_counts",
        "group_sample_counts",
    ):
        row[column] = json.dumps(row[column], sort_keys=True)
    return row


@pytest.mark.parametrize(
    ("covered_counts", "expected_status"),
    [
        ([8, 8, 8, 8, 8], "UNDERCOVERAGE"),
        ([7, 8, 9, 10, 10], "WITHIN_GROUPED_UNCERTAINTY"),
        ([10, 10, 10, 10, 10], "OVERCONSERVATIVE"),
    ],
)
def test_grouped_coverage_uses_subject_level_student_t_interval(
    covered_counts: list[int], expected_status: str
) -> None:
    result = _grouped_metrics(covered_counts)

    assert result["group_coverage_count"] == 5
    assert result["group_coverage_status"] == expected_status


def test_grouped_validation_hard_fails_only_supported_undercoverage() -> None:
    under = _serialized_group_row(_grouped_metrics([8, 8, 8, 8, 8]))
    over = _serialized_group_row(_grouped_metrics([10, 10, 10, 10, 10]))

    under_result = validate_coverage_results(pd.DataFrame([under]), _group_config())
    over_result = validate_coverage_results(pd.DataFrame([over]), _group_config())

    assert under_result.check_name == "no_grouped_undercoverage"
    assert not under_result.hard_pass
    assert under_result.scientific_status == "UNDERCOVERAGE"
    assert over_result.hard_pass
    assert over_result.scientific_status == "OVERCONSERVATIVE"
    assert over_result.diagnostics["all_rows_within_grouped_uncertainty"] is False


def test_fixed_validation_preserves_existing_row_tolerance() -> None:
    config = {
        "split": {"unit": "rows"},
        "conformal": {"alpha": 0.1},
        "stability": {"max_coverage_deviation": 0.02},
    }

    passing = validate_coverage_results(
        pd.DataFrame({"coverage": [0.88, 0.92]}), config
    )
    failing = validate_coverage_results(
        pd.DataFrame({"coverage": [0.920001]}), config
    )

    assert passing.check_name == "coverage_within_stability_tolerance"
    assert passing.hard_pass
    assert not failing.hard_pass
    assert failing.diagnostics["failing_rows"] == 1


def test_subject_coverage_artifact_expands_replayable_details() -> None:
    grouped = _serialized_group_row(_grouped_metrics([7, 8, 9, 10, 10]))
    result_row = {
        "experiment": "har",
        "dataset": "human_activity_recognition",
        "seed": 45,
        "split_id": "split",
        "selection_data_id": "selection",
        "code_version": "test",
        "model": "model",
        "method": "all_features",
        "selection_type": "all_features",
        "target_size": 3,
        "n_features": 3,
        "selected_indices": "[0, 1, 2]",
        "scaling": "base",
        "score": "aps",
        "alpha": 0.1,
        **grouped,
    }

    artifact = subject_coverage_table(pd.DataFrame([result_row]))

    assert len(artifact) == 5
    assert artifact["window_count"].eq(10).all()
    assert artifact["covered_window_count"].tolist() == [7, 8, 9, 10, 10]
    assert artifact["group_coverage_status"].eq(
        "WITHIN_GROUPED_UNCERTAINTY"
    ).all()


def test_grouped_revalidation_rejects_legacy_aggregate_only_results() -> None:
    with pytest.raises(ValueError, match="requires saved group metrics"):
        validate_coverage_results(
            pd.DataFrame({"coverage": [0.94]}), _group_config()
        )


def test_grouped_revalidation_rejects_status_that_disagrees_with_interval() -> None:
    row = _serialized_group_row(_grouped_metrics([10, 10, 10, 10, 10]))
    row["group_coverage_status"] = "WITHIN_GROUPED_UNCERTAINTY"

    with pytest.raises(ValueError, match="disagree with intervals"):
        validate_coverage_results(pd.DataFrame([row]), _group_config())


def test_phase6_resume_requires_recomputed_aggregates_to_match_frozen_results() -> None:
    key = {
        "model": "model",
        "method": "all_features",
        "target_size": 3,
        "selected_indices": "[0, 1, 2]",
        "scaling": "base",
        "score": "aps",
    }
    saved = pd.DataFrame([{**key, "coverage": 0.91, "mean_size": 1.5}])
    matching = saved.copy()
    changed = pd.DataFrame([{**key, "coverage": 0.92, "mean_size": 1.5}])

    _assert_recomputed_results_match(saved, matching)
    with pytest.raises(RuntimeError, match="existing artifacts were not overwritten"):
        _assert_recomputed_results_match(saved, changed)


def test_phase6_replay_allows_only_validation_policy_config_upgrade(tmp_path) -> None:
    saved_config = {
        "seed": 45,
        "split": {"unit": "groups"},
        "conformal": {"alpha": 0.1},
    }
    config_path = tmp_path / "resolved.yaml"
    config_path.write_text(
        "seed: 45\nsplit:\n  unit: groups\nconformal:\n  alpha: 0.1\n",
        encoding="utf-8",
    )
    upgraded = {
        **saved_config,
        "coverage_validation": {"mode": "grouped", "confidence_level": 0.95},
    }

    _validate_replay_config(config_path, upgraded)
    with pytest.raises(ValueError, match="different config"):
        _validate_replay_config(config_path, {**upgraded, "seed": 46})
