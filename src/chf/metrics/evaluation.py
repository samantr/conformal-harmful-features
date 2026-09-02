from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    nll: float
    ece: float


@dataclass(frozen=True)
class ConditionalCoverageMetrics:
    sscv: float
    class_coverage_gap: float
    class_coverage_max_deviation: float
    class_coverages: tuple[float, ...]


def _validate_probabilities_and_labels(
    probabilities: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    probability_values = np.asarray(probabilities, dtype=np.float64)
    label_values = np.asarray(labels)
    if probability_values.ndim != 2 or probability_values.shape[1] < 2:
        raise ValueError("probabilities must have shape (n_samples, n_classes>=2)")
    if label_values.shape != (len(probability_values),):
        raise ValueError("labels must contain one value per sample")
    if not np.issubdtype(label_values.dtype, np.integer):
        raise ValueError("labels must be integers")
    if np.any((label_values < 0) | (label_values >= probability_values.shape[1])):
        raise ValueError("labels contain an invalid class")
    if not np.all(np.isfinite(probability_values)) or np.any(probability_values < 0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(probability_values.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("probability rows must sum to one")
    return probability_values, label_values.astype(np.int64, copy=False)


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, *, n_bins: int = 15
) -> float:
    probability_values, label_values = _validate_probabilities_and_labels(
        probabilities, labels
    )
    if not isinstance(n_bins, (int, np.integer)) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")
    predictions = probability_values.argmax(axis=1)
    confidences = probability_values.max(axis=1)
    correct = predictions == label_values
    bin_indices = np.minimum((confidences * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for bin_index in range(n_bins):
        mask = bin_indices == bin_index
        if np.any(mask):
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidences[mask].mean())
            )
    return ece


def classification_metrics(
    probabilities: np.ndarray, labels: np.ndarray, *, ece_bins: int = 15
) -> ClassificationMetrics:
    probability_values, label_values = _validate_probabilities_and_labels(
        probabilities, labels
    )
    predictions = probability_values.argmax(axis=1)
    return ClassificationMetrics(
        accuracy=float(accuracy_score(label_values, predictions)),
        macro_f1=float(f1_score(label_values, predictions, average="macro")),
        nll=float(
            log_loss(
                label_values,
                probability_values,
                labels=np.arange(probability_values.shape[1]),
            )
        ),
        ece=expected_calibration_error(probability_values, label_values, n_bins=ece_bins),
    )


def conditional_coverage_metrics(
    included: np.ndarray,
    labels: np.ndarray,
    *,
    target_coverage: float,
    min_sscv_stratum_size: int = 25,
    size_strata: tuple[tuple[int, int], ...] | None = None,
) -> ConditionalCoverageMetrics:
    inclusion_values = np.asarray(included, dtype=bool)
    label_values = np.asarray(labels)
    if inclusion_values.ndim != 2 or label_values.shape != (len(inclusion_values),):
        raise ValueError("included and labels have incompatible shapes")
    if not 0 < target_coverage < 1:
        raise ValueError("target_coverage must lie strictly between zero and one")
    if min_sscv_stratum_size <= 0:
        raise ValueError("min_sscv_stratum_size must be positive")
    n_classes = inclusion_values.shape[1]
    if np.any((label_values < 0) | (label_values >= n_classes)):
        raise ValueError("labels contain an invalid class")

    covered = inclusion_values[np.arange(len(label_values)), label_values]
    sizes = inclusion_values.sum(axis=1)
    if size_strata is None:
        upper_bounds = (1, 3, 10, 100, 1000)
        lower_bound = 0
        generated_strata = []
        for upper_bound in upper_bounds:
            generated_strata.append((lower_bound, min(upper_bound, n_classes)))
            lower_bound = upper_bound + 1
            if upper_bound >= n_classes:
                break
        size_strata = tuple(generated_strata)
    stratum_deviations = [
        abs(float(covered[(sizes >= lower) & (sizes <= upper)].mean()) - target_coverage)
        for lower, upper in size_strata
        if int(np.count_nonzero((sizes >= lower) & (sizes <= upper)))
        >= min_sscv_stratum_size
    ]
    if not stratum_deviations:
        stratum_deviations = [abs(float(covered.mean()) - target_coverage)]

    class_coverages = tuple(
        float(covered[label_values == class_index].mean())
        for class_index in range(n_classes)
        if np.any(label_values == class_index)
    )
    return ConditionalCoverageMetrics(
        sscv=max(stratum_deviations),
        class_coverage_gap=max(class_coverages) - min(class_coverages),
        class_coverage_max_deviation=max(
            abs(coverage - target_coverage) for coverage in class_coverages
        ),
        class_coverages=class_coverages,
    )
