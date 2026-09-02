from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticFeature:
    index: int
    name: str
    role: str
    signal_scale: float
    source_feature: str | None = None


@dataclass(frozen=True)
class ControlledSyntheticDataset:
    features: np.ndarray
    labels: np.ndarray
    feature_manifest: tuple[SyntheticFeature, ...]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.feature_manifest)


def class_separation_ratios(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return per-column between-class variance divided by total variance."""
    feature_values = np.asarray(features, dtype=np.float64)
    label_values = np.asarray(labels)
    if feature_values.ndim != 2 or label_values.shape != (len(feature_values),):
        raise ValueError("features and labels have incompatible shapes")
    overall_means = feature_values.mean(axis=0)
    between = np.zeros(feature_values.shape[1], dtype=np.float64)
    for class_label in np.unique(label_values):
        class_values = feature_values[label_values == class_label]
        between += len(class_values) * (class_values.mean(axis=0) - overall_means) ** 2
    between /= len(feature_values)
    total = feature_values.var(axis=0)
    return np.divide(between, total, out=np.zeros_like(between), where=total > 0)


def _balanced_labels(n_samples: int, n_classes: int, rng: np.random.Generator) -> np.ndarray:
    labels = np.resize(np.arange(n_classes, dtype=np.int64), n_samples)
    rng.shuffle(labels)
    return labels


def _class_means(
    n_classes: int,
    n_features: int,
    signal_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_features == 0:
        return np.empty((n_classes, 0), dtype=np.float64)
    means = rng.normal(size=(n_classes, n_features))
    means -= means.mean(axis=0, keepdims=True)
    standard_deviations = means.std(axis=0, keepdims=True)
    if np.any(standard_deviations == 0):
        raise RuntimeError("failed to generate non-degenerate class means")
    return signal_scale * means / standard_deviations


def make_controlled_multiclass(
    *,
    n_samples: int,
    n_classes: int,
    n_strong: int,
    n_weak: int,
    n_redundant: int,
    n_noise: int,
    seed: int,
    strong_signal: float = 1.0,
    weak_signal: float = 0.25,
    redundant_noise: float = 0.15,
) -> ControlledSyntheticDataset:
    """Generate balanced multiclass data with explicit ground-truth feature roles.

    Strong and weak features are noisy observations of class-specific means.
    Redundant features are noisy copies of strong features, while pure-noise
    features are independent of the labels. Feature order is intentionally
    fixed so the ground-truth role of every column is auditable.
    """
    counts = (n_strong, n_weak, n_redundant, n_noise)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if not 5 <= n_classes <= 20:
        raise ValueError("n_classes must be between 5 and 20")
    if any(not isinstance(count, (int, np.integer)) or count < 0 for count in counts):
        raise ValueError("feature counts must be non-negative integers")
    if n_strong == 0:
        raise ValueError("at least one strong feature is required")
    if sum(counts) == 0:
        raise ValueError("at least one feature is required")
    if strong_signal <= 0 or weak_signal < 0 or redundant_noise < 0:
        raise ValueError("signal scales must be non-negative and strong_signal positive")

    rng = np.random.default_rng(seed)
    labels = _balanced_labels(n_samples, n_classes, rng)

    strong_means = _class_means(n_classes, n_strong, strong_signal, rng)
    weak_means = _class_means(n_classes, n_weak, weak_signal, rng)
    strong = strong_means[labels] + rng.normal(size=(n_samples, n_strong))
    weak = weak_means[labels] + rng.normal(size=(n_samples, n_weak))

    redundant_columns: list[np.ndarray] = []
    redundant_sources: list[int] = []
    for redundant_index in range(n_redundant):
        source_index = redundant_index % n_strong
        copied = strong[:, source_index] + redundant_noise * rng.normal(size=n_samples)
        redundant_columns.append(copied)
        redundant_sources.append(source_index)
    redundant = (
        np.column_stack(redundant_columns)
        if redundant_columns
        else np.empty((n_samples, 0), dtype=np.float64)
    )
    noise = rng.normal(size=(n_samples, n_noise))
    features = np.column_stack((strong, weak, redundant, noise)).astype(
        np.float64, copy=False
    )

    manifest: list[SyntheticFeature] = []
    for role, count, signal_scale in (
        ("strong", n_strong, strong_signal),
        ("weak", n_weak, weak_signal),
    ):
        for role_index in range(count):
            manifest.append(
                SyntheticFeature(
                    index=len(manifest),
                    name=f"{role}_{role_index:02d}",
                    role=role,
                    signal_scale=signal_scale,
                )
            )
    for role_index, source_index in enumerate(redundant_sources):
        manifest.append(
            SyntheticFeature(
                index=len(manifest),
                name=f"redundant_{role_index:02d}",
                role="redundant",
                signal_scale=strong_signal,
                source_feature=f"strong_{source_index:02d}",
            )
        )
    for role_index in range(n_noise):
        manifest.append(
            SyntheticFeature(
                index=len(manifest),
                name=f"noise_{role_index:02d}",
                role="noise",
                signal_scale=0.0,
            )
        )

    return ControlledSyntheticDataset(
        features=features,
        labels=labels,
        feature_manifest=tuple(manifest),
    )
