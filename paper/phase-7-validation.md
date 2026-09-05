# Phase 7 Validation Record

**Status:** Complete - Dry Bean, Covertype, and Human Activity Recognition frozen

**Completed:** 2026-09-05

**Scope:** Transfer the complete leakage-safe selection, baseline, and scaling
protocol to the preregistered real datasets. UCI Dry Bean, Covertype, and Human
Activity Recognition are complete. The optional medical transfer is deferred;
it is not required for the Phase 8 robustness gate.

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
advance to Covertype, whose result follows below; Phase 8 must establish whether
the neural gains and feature choices survive paired multi-seed analysis.

## Phase 7B - Covertype

### Dataset, split, and compute budget

The official [UCI Covertype archive](https://archive.ics.uci.edu/dataset/31/covertype)
is pinned by SHA-256 digest
`89a975c2457cd48e824238ae43c5a3cb762e42c4b4078d9b44a4514055105f6d`.
The loader verifies the compressed member, 581,012 rows, 54 integer features,
finite values, four wilderness and 40 soil binary indicators, exactly one
active indicator in each group, and all seven labels. The ten quantitative
columns are standardized from training data; the 44 one-hot indicators remain
on their native 0/1 scale.

The seed-44 stratified outer split is:

| Partition | Rows | Permitted use |
|---|---:|---|
| Training | 290,506 | Final classifier and preprocessing fits |
| Tuning/selection | 96,835 | Scaling and frozen-choice data only |
| Conformal calibration | 96,835 | Fresh thresholds after choices freeze |
| Test | 96,836 | One final evaluation per frozen pipeline |

The outer split identifier is `ad4da0a0e1bd9414`. To make the 54-feature
retrain-and-rerank search tractable, every selection method receives the same
fixed stratified subset of 20,000 outer-training and 20,000 outer-tuning rows.
Its identifier is `cef893d2c919063b`. This compute budget affects ranking and
subset-size choice only. Every frozen final model is refit on all 290,506
training rows, while final scaling, calibration, and test use the complete
outer partitions. Recursive removal is capped at five features. The neural
model has a fixed 60-epoch cap; some fits reach that cap, so model convergence
is a declared limitation rather than an unreported exception.

### Numerical-safety reset

The first implementation-validation attempt standardized rare one-hot soil
indicators and reconstructed neural logits from already rounded probabilities.
The resulting full-data neural fit produced exact zero/one probabilities and
all ConfTS temperatures failed the preregistered safety rule. The attempt
stopped before final calibration/test evaluation and is excluded.

Before restarting selection, binary indicators were preserved as 0/1 and the
MLP interface was corrected to return its true final affine-layer logits. A
full-training safety check then produced no exact zeros or ones at any frozen
temperature from 1.0 through 3.0. This corrected preprocessing and logit path
was used consistently for every accepted selector and final pipeline.

### Frozen proposed subsets

| Model | Path | Features kept | Removed features | Tuning accuracy loss | Tuning APS-size gain |
|---|---|---:|---|---:|---:|
| Logistic regression | One-shot | 51 | Soil Types 1, 12, 13 | -0.00055 | 0.00511 |
| Logistic regression | Recursive | 51 | Soil Types 9, 12, 13 | -0.00035 | 0.00568 |
| Small neural network | One-shot | 53 | Soil Type 26 | -0.01145 | 0.03632 |
| Small neural network | Recursive | 53 | Soil Type 26 | -0.01145 | 0.03632 |

Negative accuracy loss means the selected subset improved tuning accuracy.
The neural one-shot and recursive paths converge to the same frozen subset.
Soil Type 26 differs from every matched ordinary selector: mutual information
removes Soil Type 37, RFE and permutation importance remove Soil Type 15, SHAP
removes Soil Type 40, and CRFE removes Hillshade 9am.

### Final untouched-test results: Base APS

| Model | Method | Features | Accuracy | Coverage | Mean size | SSCV | Class max deviation |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | All features | 54 | 0.7246 | 0.9024 | 1.6468 | 0.0987 | 0.4778 |
| Logistic regression | Proposed one-shot | 51 | 0.7248 | 0.9026 | 1.6473 | 0.0968 | 0.4752 |
| Logistic regression | Proposed recursive | 51 | 0.7243 | 0.9028 | 1.6459 | 0.0963 | 0.4778 |
| Logistic regression | CRFE, matched | 51 | 0.7236 | 0.9024 | 1.6621 | 0.1022 | 0.5271 |
| Small neural network | All features | 54 | 0.8578 | 0.9029 | 1.2465 | 0.0772 | 0.1061 |
| Small neural network | Proposed, both paths | 53 | 0.8644 | 0.9000 | **1.2248** | 0.0803 | 0.1402 |
| Small neural network | Mutual information, matched | 53 | **0.8674** | 0.9011 | 1.2258 | **0.0772** | 0.1528 |
| Small neural network | RFE/permutation, matched | 53 | 0.8673 | 0.9006 | 1.2258 | 0.0783 | 0.1459 |
| Small neural network | CRFE, matched | 53 | 0.8656 | 0.9020 | 1.2273 | 0.0802 | 0.1389 |

The logistic effect is negligible. Recursive removal reduces mean size by only
0.0009 with a 0.0002 accuracy loss; one-shot makes sets 0.0005 larger. The
negative logistic conclusion is retained despite both proposed subsets beating
matched CRFE.

The neural subset gives the clearer result. Removing Soil Type 26 improves
accuracy by 0.0066 and reduces mean APS size by 0.0217 versus all features;
coverage moves from 0.9029 to 0.9000. It has the smallest mean set among the
deterministic matched selectors, but its advantage over mutual information and
RFE is only 0.0010, while those selectors have about 0.003 higher accuracy. In
the separate paired Base-APS baseline run, the proposal also beats the mean of
ten matched random subsets in accuracy (0.8644 versus 0.8620) and set size
(1.2289 versus 1.2383). This is a modest Pareto-frontier improvement, not broad
dominance over ordinary feature selection.

Conditional behavior is negative. Neural SSCV worsens from 0.0772 to 0.0803,
and maximum class-coverage deviation worsens from 0.1061 to 0.1402. The highly
imbalanced Covertype labels also produce very large class deviations for the
linear model despite valid marginal coverage. Covertype therefore does not
support H4.

### Scaling interaction and decision

Base, TS, and ConfTS all select `T=1.0` for every deterministic subset under
both APS and RAPS. Their results are identical, every interaction term is zero,
and Covertype supplies no evidence for H3.

All 18 Phase 7 executable checks pass: checksum and schema, multiclass
suitability, disjoint outer splits, class presence, selection rows contained in
their permitted outer partitions, identical selection IDs across subprotocols,
all required methods, the complete 96-row deterministic factorial, finite
fresh thresholds and metrics, coverage within the stricter 0.01 tolerance, no
exact zero/one probabilities, frozen subsets, one final calibration/test use
per pipeline, and complete interaction/rank outputs. The full suite passes 46
tests.

**Covertype decision:** H1 receives useful support because the proposed neural
feature differs from every ordinary selector. H2 receives modest support: the
neural result improves the all-feature accuracy/efficiency frontier and beats
matched CRFE and mean random performance, but MI/RFE nearly match its
efficiency with better accuracy. H3 receives no support and H4 is negative.
These remain single-seed descriptive findings. The subject-disjoint Human
Activity Recognition transfer follows before Phase 8 paired inference.

## Phase 7C - Human Activity Recognition

### Subject-disjoint protocol

The official UCI HAR archive is pinned by SHA-256 digest
`c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`.
The loader verifies 10,299 windows, 561 finite measured features, six activity
classes, and 30 subjects. The seed-45 split is group-disjoint:

| Partition | Subjects | Windows |
|---|---|---:|
| Training | 1, 2, 3, 8, 10, 11, 16, 17, 18, 19, 20, 22, 24, 26, 30 | 5,170 |
| Tuning | 12, 14, 15, 25, 28 | 1,762 |
| Calibration | 4, 5, 6, 7, 21 | 1,660 |
| Test | 9, 13, 23, 27, 29 | 1,707 |

The split identifier is `4b6c2b777e813b84` and the selection-data identifier
is `2f8b707b2b778966`. No subject crosses partitions. Selection uses five
repetitions of four-fold cross-fitting inside outer tune, and frozen final
models are refit on outer train. The grouped coverage estimand is the
unweighted mean of held-out-subject window coverage with a two-sided 95%
Student-t interval across the five test subjects.

### Frozen proposed subsets and final Base APS results

Logistic one-shot and recursive selection both remove
`137:tBodyGyro-energy()-X`. Neural one-shot removes
`014:tBodyAcc-min()-Y`; neural recursive selection removes that feature and
then `360:fBodyAccJerk-sma()`.

| Model | Method | Features | Accuracy | Coverage | Mean size | Group coverage (95% CI) |
|---|---|---:|---:|---:|---:|---:|
| Logistic regression | All features | 561 | 0.9649 | 0.9385 | 1.0047 | 0.9370 [0.9042, 0.9699] |
| Logistic regression | Proposed, both paths | 560 | 0.9684 | 0.9391 | 1.0047 | 0.9376 [0.9045, 0.9707] |
| Small neural network | All features | 561 | 0.9619 | 0.9297 | 1.0293 | 0.9292 [0.9080, 0.9504] |
| Small neural network | Proposed one-shot | 560 | 0.9572 | 0.9279 | 1.0018 | 0.9261 [0.8877, 0.9645] |
| Small neural network | Proposed recursive | 559 | 0.9514 | 0.9291 | 0.9994 | 0.9268 [0.8782, 0.9754] |

The logistic removal improves accuracy by 0.0035 but yields no mean-size gain.
The neural one-shot and recursive subsets reduce mean size by 0.0275 and
0.0299, respectively, while losing 0.0047 and 0.0105 test accuracy. The
recursive accuracy loss is just beyond the nominal 0.01 final descriptive
boundary; subset selection itself used tuning evidence only. Neural maximum
class-coverage deviation worsens from 0.0647 to 0.0686 and 0.0810, so Phase 7C
does not support H4.

All 126 deterministic scaling/score rows pass grouped undercoverage safety: 63
intervals contain the 0.90 target and 63 are overconservative; none has an
upper confidence limit below target. All 19 Phase 7C executable checks pass.
The grouped audit contains 630 subject-level coverage rows.

### Scaling interaction and Phase 7 decision

Ordinary TS selects temperatures 2.0 for the logistic proposed subsets and for
both neural proposed subsets, versus 2.0 and 1.4 for the corresponding
all-feature models. ConfTS selects 1.0 for logistic proposed subsets and 0.8 or
0.9 for neural proposed subsets, versus 0.9 and 0.7 for all features. Under APS,
TS preserves a useful neural feature-removal size gain, while ConfTS partly
absorbs it; the interaction therefore changes sign by scaling method. This is
descriptive single-seed evidence, not support for H3 without Phase 8 pairing.

**HAR decision:** H2 receives qualified single-seed neural support and a
negative logistic result. H3 is mixed and descriptive, and H4 is negative.
Subject-level intervals expose material uncertainty that window-only intervals
would conceal. Phase 7 is frozen here. Phase 8 must determine whether these
effects, feature choices, and scaling interactions are stable across paired
seeds and changing subject assignments.

The archived Phase 7C rows record code version `b6afb43-dirty`. Numerical
cross-audit confirms that the grouped correction left all 57 original aggregate
numeric columns unchanged, and the results remain frozen. Phase 8 treats them
as regression anchors only and requires clean, manifest-bound provenance for
new observations.
