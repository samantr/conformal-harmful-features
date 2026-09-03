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

        # Reconstruct the final affine layer rather than taking log(predict_proba).
        # The latter loses information after softmax rounds confident outputs to
        # exact 0/1, which can make every otherwise safe temperature unusable.
        activation = transformed
        for weights, intercepts in zip(
            self.estimator.coefs_[:-1],
            self.estimator.intercepts_[:-1],
            strict=True,
        ):
            activation = activation @ weights + intercepts
            if self.estimator.activation == "identity":
                continue
            if self.estimator.activation == "logistic":
                positive = activation >= 0
                negative_values = np.exp(activation[~positive])
                activation[positive] = 1.0 / (
                    1.0 + np.exp(-activation[positive])
                )
                activation[~positive] = negative_values / (
                    1.0 + negative_values
                )
            elif self.estimator.activation == "tanh":
                np.tanh(activation, out=activation)
            elif self.estimator.activation == "relu":
                np.maximum(activation, 0.0, out=activation)
            else:  # pragma: no cover - sklearn rejects unknown activations
                raise ValueError(
                    f"unsupported MLP activation: {self.estimator.activation}"
                )
        values = activation @ self.estimator.coefs_[-1]
        values = values + self.estimator.intercepts_[-1]
        if values.shape[1] == 1:
            values = np.column_stack((np.zeros(len(values)), values[:, 0]))
        return np.asarray(values, dtype=np.float64)

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
    if bool(config.get("preserve_binary_indicators", False)):
        binary_columns = np.logical_or(features == 0, features == 1).all(axis=0)
        scaler.mean_[binary_columns] = 0.0
        scaler.var_[binary_columns] = 1.0
        scaler.scale_[binary_columns] = 1.0
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
