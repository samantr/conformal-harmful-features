import numpy as np

from chf.data import class_separation_ratios, make_controlled_multiclass


def test_controlled_synthetic_roles_are_reproducible_and_recoverable() -> None:
    first = make_controlled_multiclass(
        n_samples=5000,
        n_classes=10,
        n_strong=6,
        n_weak=4,
        n_redundant=4,
        n_noise=6,
        seed=42,
    )
    second = make_controlled_multiclass(
        n_samples=5000,
        n_classes=10,
        n_strong=6,
        n_weak=4,
        n_redundant=4,
        n_noise=6,
        seed=42,
    )

    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.labels, second.labels)
    assert first.features.shape == (5000, 20)
    assert np.bincount(first.labels).tolist() == [500] * 10

    ratios = class_separation_ratios(first.features, first.labels)
    role_values = {
        role: ratios[
            [feature.index for feature in first.feature_manifest if feature.role == role]
        ]
        for role in ("strong", "weak", "redundant", "noise")
    }
    assert np.median(role_values["strong"]) > np.median(role_values["weak"])
    assert np.median(role_values["weak"]) > np.median(role_values["noise"])
    assert np.median(role_values["redundant"]) > np.median(role_values["weak"])

    by_name = {feature.name: feature.index for feature in first.feature_manifest}
    for feature in first.feature_manifest:
        if feature.role == "redundant":
            correlation = np.corrcoef(
                first.features[:, feature.index],
                first.features[:, by_name[feature.source_feature]],
            )[0, 1]
            assert abs(correlation) > 0.9


def test_synthetic_generator_rejects_out_of_scope_class_counts() -> None:
    common = dict(
        n_samples=100,
        n_strong=2,
        n_weak=1,
        n_redundant=1,
        n_noise=1,
        seed=1,
    )
    for n_classes in (4, 21):
        try:
            make_controlled_multiclass(n_classes=n_classes, **common)
        except ValueError as error:
            assert "between 5 and 20" in str(error)
        else:
            raise AssertionError("out-of-scope class count was accepted")
