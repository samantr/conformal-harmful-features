import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from chf.models import fit_classifier
from chf.scaling import probabilities_from_logits


def test_logistic_classifier_returns_multiclass_logits() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(150, 5))
    labels = np.repeat(np.arange(5), 30)
    fitted = fit_classifier(
        features,
        labels,
        {"type": "logistic_regression", "max_iter": 200},
        seed=7,
    )

    logits = fitted.logits(features[:11])
    probabilities = probabilities_from_logits(logits, 1.0)
    assert logits.shape == (11, 5)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_small_neural_network_logits_reproduce_its_probabilities() -> None:
    rng = np.random.default_rng(9)
    labels = np.repeat(np.arange(5), 30)
    features = rng.normal(size=(150, 6)) + labels[:, None] * 0.1
    with pytest.warns(ConvergenceWarning):
        fitted = fit_classifier(
            features,
            labels,
            {
                "type": "mlp",
                "hidden_layers": [8],
                "max_epochs": 5,
                "early_stopping": False,
                "batch_size": 32,
            },
            seed=9,
        )
    transformed = fitted.scaler.transform(features[:13])
    expected = fitted.estimator.predict_proba(transformed)
    actual = probabilities_from_logits(fitted.logits(features[:13]), 1.0)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_binary_indicators_can_be_preserved_during_standardization() -> None:
    continuous = np.linspace(10.0, 30.0, 30)
    indicator = np.tile([0.0, 1.0, 0.0], 10)
    features = np.column_stack((continuous, indicator))
    labels = np.tile(np.arange(3), 10)
    fitted = fit_classifier(
        features,
        labels,
        {
            "type": "logistic_regression",
            "max_iter": 200,
            "preserve_binary_indicators": True,
        },
        seed=12,
    )

    transformed = fitted.scaler.transform(features)
    np.testing.assert_array_equal(transformed[:, 1], indicator)
    assert fitted.scaler.mean_[0] == pytest.approx(continuous.mean())
