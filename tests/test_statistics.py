import numpy as np
import pandas as pd
import pytest

from chf.experiments.statistics import (
    exact_sign_flip_pvalue,
    holm_adjust,
    matched_rank_biserial,
    paired_effect,
    paired_effect_table,
    rank_stability,
    removal_path_stability,
    two_way_cluster_bootstrap,
)


def test_paired_effect_reports_directional_effects_and_exact_test() -> None:
    differences = np.arange(1.0, 11.0)

    result = paired_effect(differences)

    assert result["n_pairs"] == 10
    assert result["mean_difference"] == pytest.approx(5.5)
    assert result["ci_lower"] > 0
    assert result["cohens_dz"] > 0
    assert result["rank_biserial"] == pytest.approx(1.0)
    assert result["sign_flip_pvalue"] == pytest.approx(2 / 2**10)
    assert exact_sign_flip_pvalue([0.0, 0.0]) == 1.0
    assert matched_rank_biserial([-1.0, 2.0]) == pytest.approx(1 / 3)


def test_holm_adjustment_is_monotone_in_sorted_pvalues() -> None:
    adjusted = holm_adjust([0.03, 0.01, 0.04])

    np.testing.assert_allclose(adjusted, [0.06, 0.03, 0.06])


def test_paired_effect_table_pairs_seeds_and_orients_smaller_size_as_better() -> None:
    rows = []
    for seed in range(4):
        rows.extend(
            [
                {"dataset": "data", "model": "mlp", "seed": seed,
                 "method": "all", "mean_size": 2.0},
                {"dataset": "data", "model": "mlp", "seed": seed,
                 "method": "selected", "mean_size": 1.8},
            ]
        )

    result = paired_effect_table(
        pd.DataFrame(rows),
        group_columns=("dataset", "model"),
        pair_column="seed",
        method_column="method",
        reference_method="all",
        metric="mean_size",
        higher_is_better=False,
    )

    assert len(result) == 1
    assert result.iloc[0]["mean_difference"] == pytest.approx(0.2)
    assert result.iloc[0]["positive_fraction"] == 1.0


def test_rank_stability_reports_reversal_and_top_k_overlap() -> None:
    rows = []
    for seed, order in [(1, ["a", "b", "c"]), (2, ["c", "b", "a"])]:
        for rank, feature in enumerate(order, start=1):
            rows.append(
                {"dataset": "data", "model": "mlp", "method": "harm",
                 "seed": seed, "feature": feature, "rank": rank}
            )

    result = rank_stability(pd.DataFrame(rows), top_ks=(1, 2))

    assert result["spearman"].eq(-1.0).all()
    top_one = result.loc[result["top_k"] == 1].iloc[0]
    top_two = result.loc[result["top_k"] == 2].iloc[0]
    assert top_one["jaccard"] == 0.0
    assert top_two["jaccard"] == pytest.approx(1 / 3)


def test_removal_path_stability_uses_declared_feature_universe() -> None:
    rows = []
    for seed, order in [(1, ["a", "b", "c"]), (2, ["a", "c", "d"])]:
        for rank, feature in enumerate(order, start=1):
            rows.append(
                {
                    "dataset": "data",
                    "model": "mlp",
                    "method": "recursive",
                    "seed": seed,
                    "feature": feature,
                    "rank": rank,
                    "universe_size": 10,
                }
            )

    result = removal_path_stability(pd.DataFrame(rows), top_ks=(1, 3))

    assert result.loc[result["top_k"] == 1, "jaccard"].iloc[0] == 1.0
    top_three = result.loc[result["top_k"] == 3].iloc[0]
    assert top_three["jaccard"] == 0.5
    assert top_three["kuncheva"] == pytest.approx((2 * 10 - 9) / (3 * 7))


def test_two_way_cluster_bootstrap_is_reproducible_and_contains_constant_effect() -> None:
    rows = [
        {"seed": seed, "subject_id": subject, "difference": 0.2}
        for seed in range(3)
        for subject in range(4)
    ]

    first = two_way_cluster_bootstrap(
        pd.DataFrame(rows), value_column="difference", repetitions=200,
        random_seed=123,
    )
    second = two_way_cluster_bootstrap(
        pd.DataFrame(rows), value_column="difference", repetitions=200,
        random_seed=123,
    )

    assert first == second
    assert first["n_seeds"] == 3
    assert first["n_subjects"] == 4
    assert first["ci_lower"] == pytest.approx(0.2)
    assert first["ci_upper"] == pytest.approx(0.2)
