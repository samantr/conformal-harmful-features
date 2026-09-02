"""Feature-ranking baselines used by the matched-subset comparison."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def descending_order(values: np.ndarray) -> np.ndarray:
    """Return a deterministic descending feature order."""
    scores = np.asarray(values, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("feature scores must be a finite vector")
    return np.lexsort((np.arange(len(scores)), -scores))


def mutual_information_order(
    features_train: np.ndarray, labels_train: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    scores = mutual_info_classif(
        np.asarray(features_train), np.asarray(labels_train), random_state=seed
    )
    return descending_order(scores), scores


def permutation_order(
    estimator: object,
    features_tune: np.ndarray,
    labels_tune: np.ndarray,
    *,
    seed: int,
    repeats: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features_tune)
    labels = np.asarray(labels_tune)
    if repeats < 1:
        raise ValueError("permutation repeats must be positive")
    baseline = np.mean(np.asarray(estimator.predict(features)) == labels)
    rng = np.random.default_rng(seed)
    importances = np.empty((features.shape[1], repeats), dtype=float)
    for feature_index in range(features.shape[1]):
        for repetition in range(repeats):
            permuted = features.copy()
            permuted[:, feature_index] = permuted[
                rng.permutation(len(permuted)), feature_index
            ]
            accuracy = np.mean(np.asarray(estimator.predict(permuted)) == labels)
            importances[feature_index, repetition] = baseline - accuracy
    scores = importances.mean(axis=1)
    return descending_order(scores), scores


def rfe_order(
    features_train: np.ndarray, labels_train: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Classical coefficient-based RFE using a shared linear selector."""
    scaler = StandardScaler().fit(features_train)
    estimator = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)
    selector = RFE(estimator, n_features_to_select=1, step=1)
    selector.fit(scaler.transform(features_train), labels_train)
    # sklearn rank 1 is the final survivor, hence lower is better.
    importance = len(selector.ranking_) + 1 - selector.ranking_.astype(float)
    return descending_order(importance), importance


def crfe_order(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_tune: np.ndarray,
    labels_tune: np.ndarray,
    *,
    seed: int,
    lambda_value: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce CRFE's multiclass beta elimination rule.

    The outer training split fits each linear SVM and outer tune supplies CRFE's
    held-out conformity evidence. The returned order is best-to-worst, so every
    requested matched size is a prefix of one recursive run.
    """
    x_train = np.asarray(features_train, dtype=float)
    x_tune = np.asarray(features_tune, dtype=float)
    y_train = np.asarray(labels_train)
    y_tune = np.asarray(labels_tune)
    if not 0 <= lambda_value <= 1:
        raise ValueError("CRFE lambda must lie in [0, 1]")
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_tune = scaler.transform(x_tune)
    classes = np.unique(y_train)
    class_to_index = {value: index for index, value in enumerate(classes)}
    encoded_tune = np.array([class_to_index[value] for value in y_tune], dtype=int)
    remaining = list(range(x_train.shape[1]))
    eliminated: list[int] = []
    elimination_beta: dict[int, float] = {}
    lambda_prime = (1.0 - lambda_value) / (len(classes) - 1)
    base = LinearSVC(dual="auto", random_state=seed, max_iter=10_000)
    while len(remaining) > 1:
        model = OneVsRestClassifier(clone(base), n_jobs=1)
        model.fit(x_train[:, remaining], y_train)
        weights = np.array([estimator.coef_[0] for estimator in model.estimators_])
        columns = x_tune[:, remaining]
        true_weights = weights[encoded_tune]
        other_weights = weights.sum(axis=0)[None, :] - true_weights
        beta = -np.sum(
            lambda_value * true_weights * columns
            - lambda_prime * other_weights * columns,
            axis=0,
        )
        deleted_position = int(np.argmax(beta))
        deleted = remaining.pop(deleted_position)
        eliminated.append(deleted)
        elimination_beta[deleted] = float(beta[deleted_position])
    # The survivor is strongest; the last eliminated is next strongest.
    order = np.array(remaining + eliminated[::-1], dtype=int)
    scores = np.empty(x_train.shape[1], dtype=float)
    scores[order] = np.arange(len(order), 0, -1, dtype=float)
    return order, scores


def shap_order(
    predict_probability: Callable[[np.ndarray], np.ndarray],
    background: np.ndarray,
    evaluation: np.ndarray,
    *,
    seed: int,
    max_evaluations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Model-agnostic permutation SHAP ranking with an explicit compute cap."""
    try:
        import shap
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("SHAP baseline requires the optional 'shap' package") from error
    background_values = np.asarray(background, dtype=float)
    evaluation_values = np.asarray(evaluation, dtype=float)
    masker = shap.maskers.Independent(background_values, max_samples=len(background_values))
    explainer = shap.Explainer(
        predict_probability,
        masker,
        algorithm="permutation",
        seed=seed,
    )
    explanation = explainer(
        evaluation_values,
        max_evals=max_evaluations,
        silent=True,
    )
    values = np.asarray(explanation.values)
    # Multiclass shape is (samples, features, classes).
    feature_axis = 1 if values.ndim == 3 else -1
    reduce_axes = tuple(axis for axis in range(values.ndim) if axis != feature_axis)
    scores = np.mean(np.abs(values), axis=reduce_axes)
    return descending_order(scores), scores
