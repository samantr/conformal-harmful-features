import numpy as np

from chf.conformal import nonrandomized_score_matrix, randomized_score_matrix


def _frozen_randomized_reference(probabilities: np.ndarray, u_values: np.ndarray) -> np.ndarray:
    """Formula copied from reproduction/step20_complete_synthetic.py."""
    order = np.argsort(-probabilities, axis=1)
    sorted_probabilities = np.take_along_axis(probabilities, order, axis=1)
    cumulative_before = np.cumsum(sorted_probabilities, axis=1) - sorted_probabilities
    randomized_sorted = cumulative_before + u_values[:, None] * sorted_probabilities
    scores = np.empty_like(probabilities)
    np.put_along_axis(scores, order, randomized_sorted, axis=1)
    return scores


def test_hand_calculated_aps_and_raps_scores() -> None:
    probabilities = np.array([[0.5, 0.3, 0.2]])
    u_values = np.array([0.4])

    aps = randomized_score_matrix(probabilities, u_values, "aps")
    raps = randomized_score_matrix(
        probabilities, u_values, "raps", k_reg=1, lambda_reg=0.1
    )

    np.testing.assert_allclose(aps, [[0.2, 0.62, 0.88]])
    np.testing.assert_allclose(raps, [[0.2, 0.72, 1.08]])


def test_nonrandomized_scores_include_candidate_probability() -> None:
    probabilities = np.array([[0.5, 0.3, 0.2]])
    np.testing.assert_allclose(
        nonrandomized_score_matrix(probabilities, "aps"), [[0.5, 0.8, 1.0]]
    )


def test_randomized_aps_matches_frozen_reproduction() -> None:
    rng = np.random.default_rng(90210)
    raw = rng.uniform(size=(31, 7))
    probabilities = raw / raw.sum(axis=1, keepdims=True)
    u_values = rng.random(len(probabilities))

    expected = _frozen_randomized_reference(probabilities, u_values)
    actual = randomized_score_matrix(probabilities, u_values, "aps")

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_randomized_raps_adds_rank_penalty_to_frozen_aps() -> None:
    rng = np.random.default_rng(8)
    raw = rng.uniform(size=(19, 5))
    probabilities = raw / raw.sum(axis=1, keepdims=True)
    uniforms = rng.random(len(probabilities))
    aps = _frozen_randomized_reference(probabilities, uniforms)

    order = np.argsort(-probabilities, axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(1, 6)[None, :], axis=1)
    expected = aps + 0.02 * np.maximum(ranks - 2, 0)

    actual = randomized_score_matrix(
        probabilities, uniforms, "raps", k_reg=2, lambda_reg=0.02
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
