import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from chf.selection import crfe_order, descending_order, permutation_order, rfe_order


def test_descending_order_is_deterministic_for_ties() -> None:
    np.testing.assert_array_equal(descending_order(np.array([0.2, 0.5, 0.5])), [1, 2, 0])


def test_rfe_and_crfe_return_complete_unique_orders() -> None:
    features, labels = make_classification(
        n_samples=180, n_features=6, n_informative=4, n_redundant=0,
        n_classes=3, random_state=4,
    )
    for order, scores in (
        rfe_order(features[:120], labels[:120], seed=4),
        crfe_order(features[:120], labels[:120], features[120:], labels[120:], seed=4),
    ):
        assert sorted(order.tolist()) == list(range(6))
        assert scores.shape == (6,)
        assert np.isfinite(scores).all()


def test_permutation_ranking_uses_fitted_estimator() -> None:
    rng = np.random.default_rng(3)
    features = rng.normal(size=(300, 3))
    labels = (features[:, 0] > 0).astype(int)
    model = LogisticRegression().fit(features[:200], labels[:200])
    order, scores = permutation_order(
        model, features[200:], labels[200:], seed=3, repeats=4
    )
    assert order[0] == 0
    assert scores[0] > scores[1:].max()
