from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class FittedClassifier:
    name: str
    scaler: StandardScaler
    estimator: LogisticRegression | MLPClassifier

    def logits(self, features: np.ndarray) -> np.ndarray:
        transformed = self.scaler.transform(np.asarray(features))
        if isinstance(self.estimator, LogisticRegression):
            values = np.asarray(self.estimator.decision_function(transformed))
            if values.ndim == 1:
                values = np.column_stack((-values, values))
            return values.astype(np.float64, copy=False)

        probabilities = np.asarray(self.estimator.predict_proba(transformed))
        tiny = np.finfo(np.float64).tiny
        # log(p) is a valid set of logits because softmax(log(p)) == p.
        return np.log(np.clip(probabilities, tiny, 1.0))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        logits = self.logits(features)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.estimator.classes_[np.argmax(self.predict_proba(features), axis=1)]


def fit_classifier(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    config: Mapping[str, Any],
    *,
    seed: int,
) -> FittedClassifier:
    """Fit a classifier after learning standardization on training data only."""
    features = np.asarray(features_train)
    labels = np.asarray(labels_train)
    if features.ndim != 2 or labels.shape != (len(features),):
        raise ValueError("training features and labels have incompatible shapes")
    model_type = str(config.get("type", ""))
    scaler = StandardScaler().fit(features)
    transformed = scaler.transform(features)

    if model_type == "logistic_regression":
        estimator: LogisticRegression | MLPClassifier = LogisticRegression(
            C=float(config.get("C", 1.0)),
            max_iter=int(config.get("max_iter", 500)),
            solver="lbfgs",
            random_state=seed,
        )
    elif model_type == "mlp":
        hidden_layers = tuple(int(width) for width in config.get("hidden_layers", [64, 32]))
        if not hidden_layers or any(width <= 0 for width in hidden_layers):
            raise ValueError("mlp hidden_layers must contain positive integers")
        estimator = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=str(config.get("activation", "relu")),
            alpha=float(config.get("weight_decay", 1e-4)),
            batch_size=int(config.get("batch_size", 128)),
            learning_rate_init=float(config.get("learning_rate", 1e-3)),
            max_iter=int(config.get("max_epochs", 200)),
            early_stopping=bool(config.get("early_stopping", True)),
            validation_fraction=float(config.get("validation_fraction", 0.1)),
            n_iter_no_change=int(config.get("patience", 15)),
            random_state=seed,
        )
    else:
        raise ValueError("model type must be 'logistic_regression' or 'mlp'")

    estimator.fit(transformed, labels)
    return FittedClassifier(name=model_type, scaler=scaler, estimator=estimator)
