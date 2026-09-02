# Phase 4 Validation Record

**Status:** Complete

**Completed:** 2026-09-02

**Scope:** Compare one-shot and recursive progressive feature removal, choose
subset size using only outer-tuning evidence, freeze every choice, and then use
the outer conformal-calibration and test partitions for final evaluation.

## Selection protocol

The selection objective was fixed in the configuration as Base APS. This avoids
silently optimizing an average of methods with different probability-scaling
objectives. Every frozen subset is nevertheless evaluated under all six
Base/TS/ConfTS x APS/RAPS pipelines at the final stage.

For each classifier and candidate subset:

1. the classifier and standardizer are refit using only outer training data;
2. five repetitions of four-fold cross-fitting are run wholly inside outer tune;
3. paired randomized-APS uniforms are reused for all subset comparisons;
4. cumulative accuracy loss and coverage shortfall are measured against the
   full-feature reference;
5. a subset is eligible only if every resample remains within `0.01` maximum
   accuracy loss and `0.03` maximum coverage shortfall.

One-shot removal fixes the initial single-feature constrained-efficiency order.
Recursive removal refits every remaining single-removal candidate, re-ranks by
positive incremental efficiency gain, and stops when no such safe candidate
remains. Along each valid path, the frozen step maximizes cumulative tuning-set
efficiency gain, with conditional violation and fewer removals as tie-breakers.
The all-feature step is a valid option, so selection cannot be forced to remove a
feature when tuning evidence does not support it.

## Frozen subsets from tuning data

| Model | Method | Features kept | Removed features | Tuning accuracy loss | Tuning APS-size reduction |
|---|---|---:|---|---:|---:|
| Logistic regression | One-shot | 15 | `noise_04`, `redundant_00`, `noise_01`, `redundant_03`, `redundant_01` | -0.0015 | 0.0313 |
| Logistic regression | Recursive | 16 | `noise_04`, `noise_01`, `redundant_01`, `redundant_02` | 0.0000 | 0.0313 |
| Small neural network | One-shot | 19 | `redundant_02` | 0.0025 | 0.0912 |
| Small neural network | Recursive | 19 | `redundant_02` | 0.0025 | 0.0912 |

Negative loss denotes improved accuracy. One-shot and recursive selection found
different logistic-regression subsets with the same estimated efficiency gain.
For the neural network, both methods agreed on the first removal and recursive
selection stopped because no subsequent candidate had positive safe incremental
gain.

## Final untouched-test result: Base APS

| Model | Method | Features | Accuracy | Coverage | Mean size | SSCV | Class max deviation |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | All features | 20 | 0.7615 | 0.8880 | 1.8410 | 0.0479 | 0.080 |
| Logistic regression | One-shot | 15 | 0.7645 | 0.8845 | 1.8370 | 0.0369 | 0.095 |
| Logistic regression | Recursive | 16 | 0.7650 | 0.8840 | 1.8270 | 0.0636 | 0.090 |
| Small neural network | All features | 20 | 0.7460 | 0.8875 | 1.8810 | 0.0364 | 0.100 |
| Small neural network | One-shot | 19 | 0.7460 | 0.8860 | 1.8900 | 0.0206 | 0.090 |
| Small neural network | Recursive | 19 | 0.7460 | 0.8860 | 1.8900 | 0.0206 | 0.090 |

The strongest Phase 4 result is the logistic recursive subset: it improves test
accuracy by `0.0035` and reduces mean Base APS size by `0.0140`. Its marginal
coverage remains close to the full-feature reference, although SSCV worsens from
`0.0479` to `0.0636`. The one-shot logistic subset has a smaller size gain
(`0.0040`) but improves SSCV.

The neural-network tuning gain does **not** transfer to Base APS test efficiency:
mean size increases by `0.0090`. This negative result is retained. Under ConfTS,
the same neural-network subset reduces APS size by `0.0215`, indicating that
feature selection and scaling can interact, but that interaction belongs to the
formal Phase 6 analysis rather than a Phase 4 success claim.

All reported coverages are empirical finite-test-sample outcomes after valid
fresh calibration; exact equality to `0.90` is neither expected nor enforced.

## Validation and decision

The executable Phase 4 protocol record passes every check: both methods are
present, exactly one tuning-eligible subset is frozen per model/method, 36 final
rows cover both models and all six pipelines, thresholds and scientific metrics
are finite, and subset choices are marked frozen before final calibration.

**Phase 4 decision:** the progressive protocol works and yields a modest positive
logistic-regression result, but utility is model-dependent. The neural-network
failure means the paper cannot yet claim general improvement. Phase 5 must test
whether the logistic gain beats ordinary and random matched-size selection; later
multi-seed analysis must determine whether either result is stable.
