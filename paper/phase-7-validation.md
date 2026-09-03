# Phase 7 Validation Record

**Status:** In progress - Dry Bean milestone complete

**Completed:** 2026-09-03

**Scope:** Transfer the complete leakage-safe selection, baseline, and scaling
protocol to the first preregistered real dataset, UCI Dry Bean. Covertype, Human
Activity Recognition, and a carefully framed medical dataset remain before
Phase 7 as a whole is complete.

## Dataset and protocol

The official UCI archive is downloaded only when absent and is accepted only
after its pinned SHA-256 checksum is verified. The loader checks the ARFF schema,
row count, 16 numeric features, absence of missing/non-finite values, and all
seven expected class labels. The final dataset contains 13,611 observations.

The final seed-43 stratified split is:

| Partition | Rows | Permitted use |
|---|---:|---|
| Training | 6,805 | Fit each classifier and its `StandardScaler` |
| Tuning/selection | 2,268 | Rank features, choose subset size, tune TS/ConfTS |
| Conformal calibration | 2,269 | Compute a fresh threshold after choices freeze |
| Test | 2,269 | One final evaluation per frozen pipeline |

Every class is present in every partition. Preprocessing is refit on training
data for each feature subset; no preprocessing statistic is learned from tune,
calibration, or test. The split identifier is `a3664fdf322dbeb4`.

The benchmark includes logistic regression and a small neural network; all
features, 10 matched random subsets, mutual information, permutation
importance, RFE, SHAP, CRFE, and the proposed one-shot/recursive selectors; and
the complete Base/TS/ConfTS x APS/RAPS factorial. Subset selection remains Base
APS with five repetitions of four-fold cross-fitting wholly inside outer tune.

## Numerical-safety reset

An initial seed-42 implementation-validation run selected sharpening
temperatures between 0.4 and 0.8 and triggered the preregistered exact-0/1
probability safeguard on held-out data. That run is rejected and excluded from
all scientific results. Before restarting, the Dry Bean temperature grid was
frozen to `T >= 1.0`, saturation was added explicitly to the ConfTS tuning
rejection rule, and a fresh seed-43 split was created.

This is deliberately conservative. On the accepted run, ConfTS selected
`T=1.0` for every subset and ordinary TS selected `T=1.0` except for the neural
recursive subset (`T=1.1`). Therefore Dry Bean supplies no evidence that
sharpening with ConfTS is useful under the accepted numerical rule.

## Frozen proposed subsets

| Model | Path | Features kept | Removed features | Tuning accuracy loss | Tuning APS-size gain |
|---|---|---:|---|---:|---:|
| Logistic regression | One-shot | 15 | Solidity | 0.0013 | 0.0045 |
| Logistic regression | Recursive | 13 | AspectRatio, Solidity, ShapeFactor3 | 0.0004 | 0.0053 |
| Small neural network | One-shot | 15 | MajorAxisLength | -0.0053 | 0.0132 |
| Small neural network | Recursive | 14 | MajorAxisLength, ShapeFactor1 | -0.0026 | 0.0180 |

Negative accuracy loss means the subset improved tuning accuracy. The neural
choices are distinct from the standard selectors: none of the matched ordinary
15-feature subsets removes `MajorAxisLength`, and none of the matched ordinary
14-feature subsets removes the pair `MajorAxisLength` + `ShapeFactor1`.

## Final untouched-test results: Base APS

| Model | Method | Features | Accuracy | Coverage | Mean size | SSCV | Class max deviation |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | All features | 16 | 0.9207 | 0.8973 | 1.1009 | 0.0494 | 0.0391 |
| Logistic regression | Proposed one-shot | 15 | 0.9207 | 0.8982 | 1.1053 | 0.0503 | 0.0361 |
| Logistic regression | CRFE, matched | 15 | 0.9198 | 0.8969 | 1.0983 | 0.0463 | 0.0420 |
| Logistic regression | Proposed recursive | 13 | 0.9202 | 0.8986 | 1.1062 | 0.0530 | 0.0361 |
| Logistic regression | CRFE, matched | 13 | 0.9211 | 0.8969 | 1.1018 | 0.0504 | 0.0391 |
| Small neural network | All features | 16 | 0.9193 | 0.8942 | 1.0864 | 0.0417 | 0.0539 |
| Small neural network | Proposed one-shot | 15 | **0.9286** | 0.8872 | **1.0577** | **0.0317** | 0.0539 |
| Small neural network | CRFE, matched | 15 | 0.9268 | 0.8925 | 1.0701 | 0.0379 | 0.0539 |
| Small neural network | Proposed recursive | 14 | **0.9290** | 0.8942 | **1.0657** | 0.0467 | **0.0450** |
| Small neural network | CRFE, matched | 14 | 0.9273 | 0.8978 | 1.0727 | 0.0474 | 0.0568 |

The logistic proposed subsets do not improve final efficiency: their mean sets
are 0.0044-0.0053 larger than the all-feature reference, and CRFE is better at
both matched sizes. That negative result is retained.

The neural result is materially stronger. One-shot removal improves accuracy by
0.0093 and reduces mean APS size by 0.0287 versus all features; it also improves
SSCV by 0.0101. At 14 features, recursive selection improves accuracy by 0.0097,
reduces size by 0.0207 at identical empirical coverage, and improves the maximum
class-coverage deviation from 0.0539 to 0.0450. Both proposed neural subsets
also beat matched CRFE and the mean of 10 matched random subsets in the separate
paired Base-APS baseline comparison.

Coverage remains within the preset 0.03 sampling tolerance of the 0.90 target
for every accepted final row. The neural one-shot coverage of 0.8872 is lower
than the all-feature value, so its smaller sets cannot be interpreted without
that qualification. The recursive result is cleaner because its coverage is
identical to the all-feature reference.

## Scaling interaction

Because all-feature TS and ConfTS both select `T=1.0`, they are identical to
Base for Dry Bean. ConfTS is also `T=1.0` for every selected subset. The only
non-neutral scaling choice is TS at `T=1.1` for the neural recursive subset; it
increases APS size by 0.0119 and RAPS size by 0.0123 relative to its Base result.
Consequently, the accepted Dry Bean run does not support H3; it neither negates
the synthetic interaction nor supplies new positive evidence for it.

## Validation and decision

All 16 Phase 7 executable checks pass: source checksum and declared shape,
multiclass suitability, split disjointness and class presence, identical splits
across subprotocols, all required baselines, the complete 156-row deterministic
factorial, finite fresh thresholds and metrics, coverage tolerance, no exact
zero/one probabilities, frozen subsets, one final calibration/test use per
pipeline, and complete interaction/rank outputs. The full suite passes 42 tests.

**Dry Bean decision:** H1 and H2 receive useful single-seed support for the
neural network, while the logistic result is negative. H4 has partial
descriptive support: one-shot improves SSCV and recursive improves maximum
class-coverage deviation. These are not stability claims. Phase 7 should
continue with Covertype, and Phase 8 must establish whether the neural gains and
feature choices survive paired multi-seed analysis.
