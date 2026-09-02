import numpy as np
import pytest

from chf.selection import (
    HarmConstraints,
    HarmWeights,
    compute_harm_metrics,
    make_tuning_evidence_folds,
    make_tuning_evidence_split,
    pareto_fronts,
    passes_constraints,
    weighted_harm_score,
)


def test_harm_components_have_explicit_safe_directions() -> None:
    metrics = compute_harm_metrics(
        reference_accuracy=0.80,
        intervention_accuracy=0.795,
        reference_mean_size=2.0,
        intervention_mean_size=1.8,
        intervention_coverage=0.88,
        target_coverage=0.90,
        intervention_sscv=0.04,
        intervention_class_coverage_max_deviation=0.07,
    )

    assert metrics.efficiency_gain == pytest.approx(0.2)
    assert metrics.accuracy_loss == pytest.approx(0.005)
    assert metrics.coverage_shortfall == pytest.approx(0.02)
    assert metrics.coverage_deviation == pytest.approx(0.02)
    assert metrics.conditional_violation == pytest.approx(0.07)
    assert passes_constraints(metrics, HarmConstraints(0.01, 0.03))

    overcovered = compute_harm_metrics(
        reference_accuracy=0.80,
        intervention_accuracy=0.80,
        reference_mean_size=2.0,
        intervention_mean_size=2.1,
        intervention_coverage=0.93,
        target_coverage=0.90,
        intervention_sscv=0.03,
        intervention_class_coverage_max_deviation=0.03,
    )
    assert overcovered.coverage_shortfall == 0.0
    assert overcovered.coverage_deviation == pytest.approx(0.03)


def test_weighted_score_penalizes_accuracy_coverage_and_conditional_harm() -> None:
    safe = compute_harm_metrics(
        reference_accuracy=0.8,
        intervention_accuracy=0.8,
        reference_mean_size=2.0,
        intervention_mean_size=1.9,
        intervention_coverage=0.9,
        target_coverage=0.9,
        intervention_sscv=0.02,
        intervention_class_coverage_max_deviation=0.02,
    )
    unsafe = compute_harm_metrics(
        reference_accuracy=0.8,
        intervention_accuracy=0.75,
        reference_mean_size=2.0,
        intervention_mean_size=1.9,
        intervention_coverage=0.84,
        target_coverage=0.9,
        intervention_sscv=0.12,
        intervention_class_coverage_max_deviation=0.10,
    )
    weights = HarmWeights(beta=4, gamma=10, eta=1)

    assert weighted_harm_score(safe, weights) > weighted_harm_score(unsafe, weights)
    assert not passes_constraints(unsafe, HarmConstraints(0.01, 0.03))


def test_pareto_fronts_preserve_tradeoffs_and_demote_dominated_points() -> None:
    # Columns are accuracy loss, negative efficiency gain, conditional violation.
    objectives = np.array(
        [
            [0.00, -0.05, 0.04],
            [0.01, -0.08, 0.03],
            [0.02, -0.02, 0.08],
            [0.03, -0.01, 0.09],
        ]
    )

    np.testing.assert_array_equal(pareto_fronts(objectives), [1, 1, 2, 3])


def test_tuning_evidence_split_is_stratified_disjoint_and_complete() -> None:
    labels = np.repeat(np.arange(5), 40)
    split = make_tuning_evidence_split(
        labels,
        scale_fraction=0.4,
        calibration_fraction=0.3,
        evaluation_fraction=0.3,
        seed=19,
    )

    assert tuple(map(len, (split.scale_tuning, split.calibration, split.evaluation))) == (
        80,
        60,
        60,
    )
    for indices in (split.scale_tuning, split.calibration, split.evaluation):
        expected = np.full(5, 12 if len(indices) == 60 else 16)
        np.testing.assert_array_equal(np.bincount(labels[indices]), expected)
    split.assert_valid(len(labels))


def test_crossfit_evidence_evaluates_each_tuning_row_once() -> None:
    labels = np.repeat(np.arange(5), 40)
    folds = make_tuning_evidence_folds(
        labels,
        n_folds=4,
        scale_share_of_remainder=0.6,
        seed=31,
    )

    assert len(folds) == 4
    assert all(len(fold.evaluation) == 50 for fold in folds)
    combined = np.concatenate([fold.evaluation for fold in folds])
    np.testing.assert_array_equal(np.sort(combined), np.arange(len(labels)))
    for fold in folds:
        fold.assert_valid(len(labels))
