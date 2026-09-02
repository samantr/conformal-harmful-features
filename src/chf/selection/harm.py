from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass(frozen=True)
class HarmConstraints:
    max_accuracy_loss: float = 0.01
    max_coverage_shortfall: float = 0.03

    def __post_init__(self) -> None:
        if self.max_accuracy_loss < 0 or not np.isfinite(self.max_accuracy_loss):
            raise ValueError("max_accuracy_loss must be finite and non-negative")
        if self.max_coverage_shortfall < 0 or not np.isfinite(
            self.max_coverage_shortfall
        ):
            raise ValueError(
                "max_coverage_shortfall must be finite and non-negative"
            )


@dataclass(frozen=True)
class HarmWeights:
    beta: float = 4.0
    gamma: float = 10.0
    eta: float = 1.0

    def __post_init__(self) -> None:
        values = (self.beta, self.gamma, self.eta)
        if any(value < 0 or not np.isfinite(value) for value in values):
            raise ValueError("harm weights must be finite and non-negative")


@dataclass(frozen=True)
class HarmMetrics:
    efficiency_gain: float
    accuracy_loss: float
    coverage_shortfall: float
    coverage_deviation: float
    conditional_violation: float


@dataclass(frozen=True)
class TuningEvidenceSplit:
    scale_tuning: np.ndarray
    calibration: np.ndarray
    evaluation: np.ndarray

    def assert_valid(self, n_samples: int) -> None:
        parts = (self.scale_tuning, self.calibration, self.evaluation)
        if any(part.ndim != 1 or len(part) == 0 for part in parts):
            raise ValueError("tuning-evidence partitions must be non-empty vectors")
        combined = np.concatenate(parts)
        if len(combined) != n_samples or not np.array_equal(
            np.sort(combined), np.arange(n_samples)
        ):
            raise ValueError(
                "tuning-evidence partitions must cover each tuning row exactly once"
            )


def make_tuning_evidence_split(
    labels: np.ndarray,
    *,
    scale_fraction: float,
    calibration_fraction: float,
    evaluation_fraction: float,
    seed: int,
) -> TuningEvidenceSplit:
    """Split the outer tuning partition for leakage-free harm estimation."""
    label_values = np.asarray(labels)
    if label_values.ndim != 1 or len(label_values) == 0:
        raise ValueError("labels must be a non-empty vector")
    fractions = np.asarray(
        [scale_fraction, calibration_fraction, evaluation_fraction], dtype=float
    )
    if (
        not np.all(np.isfinite(fractions))
        or np.any(fractions <= 0)
        or not np.isclose(fractions.sum(), 1.0)
    ):
        raise ValueError("tuning-evidence fractions must be positive and sum to one")

    indices = np.arange(len(label_values))
    scale_tuning, remainder = train_test_split(
        indices,
        train_size=float(scale_fraction),
        random_state=seed,
        stratify=label_values,
    )
    calibration_share = calibration_fraction / (
        calibration_fraction + evaluation_fraction
    )
    calibration, evaluation = train_test_split(
        remainder,
        train_size=float(calibration_share),
        random_state=seed + 1,
        stratify=label_values[remainder],
    )
    result = TuningEvidenceSplit(
        scale_tuning=np.asarray(scale_tuning),
        calibration=np.asarray(calibration),
        evaluation=np.asarray(evaluation),
    )
    result.assert_valid(len(label_values))
    return result


def make_tuning_evidence_folds(
    labels: np.ndarray,
    *,
    n_folds: int,
    scale_share_of_remainder: float,
    seed: int,
) -> tuple[TuningEvidenceSplit, ...]:
    """Create repeated cross-fitting folds wholly inside outer tuning data.

    Each row is evaluated exactly once. For a given fold, all other rows are
    divided into scaling-tuning and conformal-calibration partitions, so the
    fold's evaluation labels cannot influence either fitted temperature or tau.
    """
    label_values = np.asarray(labels)
    if label_values.ndim != 1 or len(label_values) == 0:
        raise ValueError("labels must be a non-empty vector")
    if not isinstance(n_folds, (int, np.integer)) or n_folds < 3:
        raise ValueError("n_folds must be an integer of at least three")
    if not 0 < scale_share_of_remainder < 1:
        raise ValueError("scale_share_of_remainder must lie strictly between 0 and 1")

    all_indices = np.arange(len(label_values))
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds: list[TuningEvidenceSplit] = []
    for fold_index, (_, evaluation) in enumerate(
        splitter.split(all_indices, label_values)
    ):
        remainder = np.setdiff1d(all_indices, evaluation, assume_unique=True)
        scale_tuning, calibration = train_test_split(
            remainder,
            train_size=float(scale_share_of_remainder),
            random_state=seed + fold_index + 1,
            stratify=label_values[remainder],
        )
        fold = TuningEvidenceSplit(
            scale_tuning=np.asarray(scale_tuning),
            calibration=np.asarray(calibration),
            evaluation=np.asarray(evaluation),
        )
        fold.assert_valid(len(label_values))
        folds.append(fold)

    combined_evaluation = np.concatenate([fold.evaluation for fold in folds])
    if not np.array_equal(np.sort(combined_evaluation), all_indices):
        raise RuntimeError("cross-fitting evaluation folds must cover tuning data once")
    return tuple(folds)


def compute_harm_metrics(
    *,
    reference_accuracy: float,
    intervention_accuracy: float,
    reference_mean_size: float,
    intervention_mean_size: float,
    intervention_coverage: float,
    target_coverage: float,
    intervention_sscv: float,
    intervention_class_coverage_max_deviation: float,
) -> HarmMetrics:
    """Compute directionally explicit components of conformal harm.

    Marginal validity uses a one-sided shortfall: overcoverage is not a validity
    failure and is already reflected in prediction-set size. Conditional harm is
    the worse of size-stratified and class-conditional target deviations.
    """
    values = np.asarray(
        [
            reference_accuracy,
            intervention_accuracy,
            reference_mean_size,
            intervention_mean_size,
            intervention_coverage,
            target_coverage,
            intervention_sscv,
            intervention_class_coverage_max_deviation,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("harm inputs must be finite")
    if not 0 < target_coverage < 1 or not 0 <= intervention_coverage <= 1:
        raise ValueError("coverage values must lie in their valid ranges")
    if intervention_sscv < 0 or intervention_class_coverage_max_deviation < 0:
        raise ValueError("conditional deviations must be non-negative")
    return HarmMetrics(
        efficiency_gain=reference_mean_size - intervention_mean_size,
        accuracy_loss=reference_accuracy - intervention_accuracy,
        coverage_shortfall=max(0.0, target_coverage - intervention_coverage),
        coverage_deviation=abs(intervention_coverage - target_coverage),
        conditional_violation=max(
            intervention_sscv, intervention_class_coverage_max_deviation
        ),
    )


def passes_constraints(metrics: HarmMetrics, constraints: HarmConstraints) -> bool:
    """Return whether accuracy and marginal-validity safeguards both pass."""
    return bool(
        metrics.accuracy_loss <= constraints.max_accuracy_loss
        and metrics.coverage_shortfall <= constraints.max_coverage_shortfall
    )


def weighted_harm_score(metrics: HarmMetrics, weights: HarmWeights) -> float:
    """Return the preregistered weighted conformal-harm score."""
    return float(
        metrics.efficiency_gain
        - weights.beta * metrics.accuracy_loss
        - weights.gamma * metrics.coverage_shortfall
        - weights.eta * metrics.conditional_violation
    )


def pareto_fronts(objectives: np.ndarray) -> np.ndarray:
    """Assign non-dominated front numbers for objectives that are minimized."""
    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2 or len(values) == 0 or values.shape[1] == 0:
        raise ValueError("objectives must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("objectives must be finite")

    remaining = list(range(len(values)))
    fronts = np.zeros(len(values), dtype=np.int64)
    front_number = 1
    while remaining:
        current: list[int] = []
        for candidate in remaining:
            dominated = any(
                other != candidate
                and np.all(values[other] <= values[candidate])
                and np.any(values[other] < values[candidate])
                for other in remaining
            )
            if not dominated:
                current.append(candidate)
        if not current:
            raise RuntimeError("failed to identify a non-dominated Pareto front")
        fronts[current] = front_number
        current_set = set(current)
        remaining = [index for index in remaining if index not in current_set]
        front_number += 1
    return fronts
