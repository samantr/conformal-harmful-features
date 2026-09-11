"""Paired Phase 8 inference, rank stability, and grouped HAR uncertainty."""

from __future__ import annotations

from itertools import combinations, product
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, t as student_t, wilcoxon


def exact_sign_flip_pvalue(differences: Iterable[float]) -> float:
    """Two-sided exact randomization p-value for a paired mean difference."""
    values = np.asarray(list(differences), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("at least one finite paired difference is required")
    if len(values) > 20:
        raise ValueError("exact sign-flip enumeration is capped at 20 pairs")
    observed = abs(float(values.mean()))
    signs = np.asarray(list(product((-1.0, 1.0), repeat=len(values))))
    permuted = np.abs((signs * values).mean(axis=1))
    return float(np.mean(permuted >= observed - 1e-15))


def matched_rank_biserial(differences: Iterable[float]) -> float:
    """Matched-pairs rank-biserial correlation, with ties contributing zero."""
    values = np.asarray(list(differences), dtype=float)
    values = values[np.isfinite(values) & (values != 0)]
    if len(values) == 0:
        return 0.0
    ranks = rankdata(np.abs(values), method="average")
    total = float(ranks.sum())
    return float((ranks[values > 0].sum() - ranks[values < 0].sum()) / total)


def paired_effect(differences: Iterable[float], confidence_level: float = 0.95) -> dict[str, float | int]:
    """Return complementary parametric and rank-based paired summaries."""
    values = np.asarray(list(differences), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("paired inference requires at least two finite seeds")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between zero and one")
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=1))
    standard_error = standard_deviation / np.sqrt(len(values))
    critical = float(
        student_t.ppf(0.5 + confidence_level / 2.0, df=len(values) - 1)
    )
    if np.all(values == 0):
        wilcoxon_pvalue = 1.0
    else:
        wilcoxon_pvalue = float(
            wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue
        )
    return {
        "n_pairs": int(len(values)),
        "mean_difference": mean,
        "median_difference": float(np.median(values)),
        "standard_deviation": standard_deviation,
        "standard_error": float(standard_error),
        "ci_lower": float(mean - critical * standard_error),
        "ci_upper": float(mean + critical * standard_error),
        "cohens_dz": 0.0 if standard_deviation == 0 else mean / standard_deviation,
        "rank_biserial": matched_rank_biserial(values),
        "sign_flip_pvalue": exact_sign_flip_pvalue(values),
        "wilcoxon_pvalue": wilcoxon_pvalue,
        "positive_fraction": float(np.mean(values > 0)),
    }


def holm_adjust(pvalues: Iterable[float]) -> np.ndarray:
    """Holm family-wise adjusted p-values in the original order."""
    values = np.asarray(list(pvalues), dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a finite one-dimensional vector in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.maximum.accumulate(
        (len(values) - np.arange(len(values))) * values[order]
    )
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def paired_effect_table(
    results: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    pair_column: str,
    method_column: str,
    reference_method: str,
    metric: str,
    higher_is_better: bool,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Compute paired method-minus-reference effects for each analysis cell."""
    required = {*group_columns, pair_column, method_column, metric}
    if missing := required.difference(results.columns):
        raise ValueError(f"paired results are missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for keys, group in results.groupby(list(group_columns), sort=False, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        pivot = group.pivot(index=pair_column, columns=method_column, values=metric)
        if reference_method not in pivot:
            raise ValueError(f"missing reference method {reference_method!r}")
        for method in pivot.columns:
            if method == reference_method:
                continue
            paired = pivot[[method, reference_method]].dropna()
            raw = paired[method] - paired[reference_method]
            differences = raw if higher_is_better else -raw
            rows.append(
                {
                    **dict(zip(group_columns, key_values, strict=True)),
                    "method": method,
                    "reference_method": reference_method,
                    "metric": metric,
                    "positive_direction": "higher" if higher_is_better else "lower",
                    **paired_effect(differences, confidence_level),
                }
            )
    output = pd.DataFrame(rows)
    if not output.empty:
        for source, target in (
            ("sign_flip_pvalue", "holm_sign_flip_pvalue"),
            ("wilcoxon_pvalue", "holm_wilcoxon_pvalue"),
        ):
            output[target] = output.groupby(
                list(group_columns), sort=False, dropna=False
            )[source].transform(lambda values: holm_adjust(values.to_numpy()))
    return output


def _kuncheva(left: set[object], right: set[object], universe_size: int) -> float:
    if len(left) != len(right):
        raise ValueError("Kuncheva stability requires equal-size subsets")
    size = len(left)
    if size == 0 or size == universe_size:
        return 1.0 if left == right else 0.0
    return float((len(left & right) * universe_size - size**2) / (size * (universe_size - size)))


def rank_stability(
    rankings: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] = ("dataset", "model", "method"),
    seed_column: str = "seed",
    feature_column: str = "feature",
    rank_column: str = "rank",
    top_ks: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    """Compare complete rankings and top-k subsets across every seed pair."""
    required = {*group_columns, seed_column, feature_column, rank_column}
    if missing := required.difference(rankings.columns):
        raise ValueError(f"rankings are missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for keys, group in rankings.groupby(list(group_columns), sort=False, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        universe = set(group[feature_column])
        by_seed = {
            seed: values.set_index(feature_column)[rank_column]
            for seed, values in group.groupby(seed_column, sort=True)
        }
        for left_seed, right_seed in combinations(by_seed, 2):
            left = by_seed[left_seed]
            right = by_seed[right_seed]
            common = left.index.intersection(right.index)
            if set(common) != universe:
                raise ValueError("every seed must rank the same feature universe")
            correlation = float(spearmanr(left.loc[common], right.loc[common]).statistic)
            for top_k in top_ks:
                size = min(int(top_k), len(universe))
                left_top = set(left.nsmallest(size).index)
                right_top = set(right.nsmallest(size).index)
                union = left_top | right_top
                rows.append(
                    {
                        **dict(zip(group_columns, key_values, strict=True)),
                        "left_seed": left_seed,
                        "right_seed": right_seed,
                        "n_features": len(universe),
                        "top_k": size,
                        "spearman": correlation,
                        "jaccard": float(len(left_top & right_top) / len(union)),
                        "kuncheva": _kuncheva(left_top, right_top, len(universe)),
                    }
                )
    return pd.DataFrame(rows)


def removal_path_stability(
    rankings: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] = ("dataset", "model", "method"),
    seed_column: str = "seed",
    feature_column: str = "feature",
    rank_column: str = "rank",
    universe_size_column: str = "universe_size",
    top_ks: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    """Top-k stability for finite progressive-removal paths.

    Unlike :func:`rank_stability`, a recursive path ranks only the features it
    removes. Jaccard and Kuncheva therefore remain meaningful, while a
    complete-ranking correlation does not.
    """
    required = {
        *group_columns,
        seed_column,
        feature_column,
        rank_column,
        universe_size_column,
    }
    if missing := required.difference(rankings.columns):
        raise ValueError(f"removal paths are missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for keys, group in rankings.groupby(
        list(group_columns), sort=False, dropna=False
    ):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        universe_sizes = group[universe_size_column].drop_duplicates()
        if len(universe_sizes) != 1:
            raise ValueError("universe size must be constant within a ranking group")
        universe_size = int(universe_sizes.iloc[0])
        by_seed = {
            seed: values.sort_values(rank_column)[feature_column].tolist()
            for seed, values in group.groupby(seed_column, sort=True)
        }
        for left_seed, right_seed in combinations(by_seed, 2):
            left = by_seed[left_seed]
            right = by_seed[right_seed]
            if len(set(left)) != len(left) or len(set(right)) != len(right):
                raise ValueError("a removal path cannot rank a feature twice")
            for top_k in top_ks:
                size = int(top_k)
                if len(left) < size or len(right) < size:
                    continue
                left_top = set(left[:size])
                right_top = set(right[:size])
                union = left_top | right_top
                rows.append(
                    {
                        **dict(zip(group_columns, key_values, strict=True)),
                        "left_seed": left_seed,
                        "right_seed": right_seed,
                        "n_features": universe_size,
                        "top_k": size,
                        "spearman": np.nan,
                        "jaccard": float(len(left_top & right_top) / len(union)),
                        "kuncheva": _kuncheva(
                            left_top, right_top, universe_size
                        ),
                    }
                )
    return pd.DataFrame(rows)


def two_way_cluster_bootstrap(
    paired_subject_differences: pd.DataFrame,
    *,
    value_column: str,
    seed_column: str = "seed",
    subject_column: str = "subject_id",
    repetitions: int = 10_000,
    confidence_level: float = 0.95,
    random_seed: int = 800_001,
) -> dict[str, float | int]:
    """Pigeonhole bootstrap that preserves crossed seed/subject dependence."""
    required = {value_column, seed_column, subject_column}
    if missing := required.difference(paired_subject_differences.columns):
        raise ValueError(f"subject differences are missing columns: {sorted(missing)}")
    if repetitions < 100:
        raise ValueError("at least 100 bootstrap repetitions are required")
    values = paired_subject_differences[list(required)].dropna().copy()
    if values.empty:
        raise ValueError("subject bootstrap requires finite paired differences")
    seeds = values[seed_column].drop_duplicates().to_numpy()
    subjects = values[subject_column].drop_duplicates().to_numpy()
    if len(seeds) < 2 or len(subjects) < 2:
        raise ValueError("subject bootstrap requires at least two seeds and subjects")
    rng = np.random.default_rng(random_seed)
    estimates = np.empty(repetitions, dtype=float)
    seed_values = values[seed_column].to_numpy()
    subject_values = values[subject_column].to_numpy()
    metric_values = values[value_column].to_numpy(dtype=float)
    for index in range(repetitions):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sampled_subjects = rng.choice(subjects, size=len(subjects), replace=True)
        seed_counts = {value: int(np.count_nonzero(sampled_seeds == value)) for value in seeds}
        subject_counts = {value: int(np.count_nonzero(sampled_subjects == value)) for value in subjects}
        weights = np.asarray(
            [seed_counts[seed] * subject_counts[subject] for seed, subject in zip(seed_values, subject_values, strict=True)],
            dtype=float,
        )
        if weights.sum() == 0:  # possible for a sparse crossed table
            estimates[index] = np.nan
        else:
            estimates[index] = float(np.average(metric_values, weights=weights))
    estimates = estimates[np.isfinite(estimates)]
    if len(estimates) < repetitions * 0.95:
        raise RuntimeError("too many empty two-way bootstrap resamples")
    tail = (1.0 - confidence_level) / 2.0
    return {
        "n_cells": int(len(values)),
        "n_seeds": int(len(seeds)),
        "n_subjects": int(len(subjects)),
        "mean_difference": float(metric_values.mean()),
        "ci_lower": float(np.quantile(estimates, tail)),
        "ci_upper": float(np.quantile(estimates, 1.0 - tail)),
        "bootstrap_repetitions": int(len(estimates)),
    }
