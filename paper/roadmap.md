# Conformal-Harmful Feature Detection

## Implementation and Paper Roadmap

## Status

- ✅ **Phase 0 - Freeze the reproduction foundation:** complete; see [`phase-0-validation.md`](phase-0-validation.md).
- ✅ **Phase 1 - Controlled synthetic data:** complete; see [`phase-1-validation.md`](phase-1-validation.md).
- ✅ **Phase 2 - Single-feature interventions:** complete; see [`phase-2-validation.md`](phase-2-validation.md).
- ✅ **Phase 3 - Define conformal harm:** complete; see [`phase-3-validation.md`](phase-3-validation.md).
- ✅ **Phase 4 - Progressive subset selection:** complete; see [`phase-4-validation.md`](phase-4-validation.md).
- ✅ **Phase 5 - Required baselines:** complete; see [`phase-5-validation.md`](phase-5-validation.md).
- ✅ **Phase 6 - Interaction with scaling:** complete; see [`phase-6-validation.md`](phase-6-validation.md).
- ✅ **Phase 7 - Real datasets:** Dry Bean, Covertype, and subject-disjoint Human Activity Recognition complete. See [`phase-7-validation.md`](phase-7-validation.md).
- 🧊 **Phase 8 - Robustness and statistics:** protocol frozen and implemented; expensive benchmark not started. See [`phase-8-protocol.md`](phase-8-protocol.md).

**Working title:** *Conformal-Harmful Features: Feature Selection for Efficient Adaptive Prediction Sets*

**Scope:** Tabular multiclass classification using APS/RAPS with Base (`T=1`), ordinary temperature scaling (TS), and Conformal Temperature Scaling (ConfTS).

## 1. Research question

Can a feature be useful, or nearly harmless, for classification while disproportionately harming adaptive conformal efficiency or conditional reliability?

A provisional **conformal-harmful feature** is one whose removal or manipulation causes little classification loss but produces smaller APS/RAPS sets and/or better conditional coverage after valid, fresh conformal calibration.

This is not intended as generic feature selection. The paper must show that **classification importance differs from conformal importance**, then demonstrate a better accuracy-coverage-efficiency trade-off than ordinary and conformal feature-selection baselines.

## 2. Main hypotheses

- **H1:** Conformal-harm rankings differ materially from classification-importance rankings.
- **H2:** Targeted intervention reduces APS/RAPS size more than ordinary selection at comparable accuracy and valid marginal coverage.
- **H3:** Feature intervention and ConfTS are partly complementary.
- **H4:** Some interventions improve SSCV or class/group coverage gaps, not merely average size.

H4 is desirable but must not be assumed.

## 3. Valid experimental protocol

Use four disjoint partitions:

| Partition | Permitted use |
|---|---|
| Training | Fit a classifier for a fixed feature subset |
| Tuning/selection | Rank/select features; tune TS, ConfTS, and hyperparameters |
| Conformal calibration | Compute the final APS/RAPS threshold after all choices are frozen |
| Test | Final accuracy, calibration, coverage, size, and conditional metrics |

Never choose a feature subset using final calibration or test performance. For each chosen subset, retrain the classifier unless the experiment is explicitly labeled **inference-time masking**. Retraining and masking answer different questions.

## 4. Success criteria

A smaller set alone is not success. A valid result must:

1. maintain target marginal coverage within sampling uncertainty;
2. constrain or transparently report accuracy loss;
3. improve a Pareto frontier, not one cherry-picked operating point;
4. repeat across seeds, datasets, and at least two classifier families;
5. outperform credible ordinary and conformal baselines;
6. avoid extreme-temperature numerical artifacts.

## 5. Implementation phases

### Phase 0 - Freeze the reproduction foundation

- [x] Preserve existing reproduction scripts and saved results.
- [x] Modularize Base/TS/ConfTS and APS/RAPS only after regression tests pass.
- [x] Verify randomized APS/RAPS and the finite-sample conformal quantile with hand-calculated cases.
- [x] Save split indices and random seeds.
- [x] Log zero-probability counts and probability saturation.

**Exit: PASSED.** Modular code matches the reproduction within the tolerances documented in [`phase-0-validation.md`](phase-0-validation.md).

### Phase 1 - Controlled synthetic data

Generate 5-20 class datasets with about 20 known features:

- strongly informative;
- weakly informative;
- redundant/correlated;
- pure noise;
- optionally misleading interaction features.

Use multinomial logistic regression and a small neural network. Record accuracy, macro-F1, NLL, ECE, APS/RAPS coverage, mean/median/90th-percentile size, empty/full-set rates, SSCV, and class-conditional coverage gaps.

**Exit:** Known feature roles are recoverable and the full-feature baseline is stable.

### Phase 2 - Single-feature interventions

For every feature `j`, separately run:

1. **Retrain ablation:** remove `j`, retrain, retune scaling, recalibrate, evaluate.
2. **Masking sensitivity:** keep the model fixed, mask or permute `j`, then evaluate under a clearly defined calibration protocol.

Store deltas for accuracy, F1, NLL, ECE, APS/RAPS size, coverage, SSCV, and class-coverage gap.

**Exit:** Find repeatable features with small accuracy cost and disproportionate conformal effect, or reject the hypothesis early.

### Phase 3 - Define conformal harm

Compare three formulations:

**A. Constrained efficiency gain**

Rank by set-size reduction only among interventions satisfying preset accuracy-loss and coverage-deviation limits. This should be the first, most interpretable method.

**B. Weighted score**

```text
harm(j) = efficiency_gain
          - beta * accuracy_loss
          - gamma * coverage_violation
          - eta * conditional_violation
```

Weights must be tuned without touching final calibration/test data.

**C. Pareto ranking**

Treat accuracy, set size, and conditional violation as separate objectives. Prefer constrained or Pareto reporting; simple ratios are unstable when accuracy loss is near zero.

**Exit: PASSED.** Constrained efficiency is the primary definition. It uses
tuning-only cross-fitted evidence, requires positive efficiency gain, enforces
accuracy and marginal-coverage safeguards, and passes the selection-resample
stability gate documented in [`phase-3-validation.md`](phase-3-validation.md).

### Phase 4 - Progressive subset selection

1. Rank features on tuning data.
2. Remove the best candidate.
3. Optionally retrain and re-rank recursively.
4. Stop at the preset accuracy constraint.
5. Choose subset size on tuning data.
6. Freeze all choices.
7. Compute a fresh conformal threshold.
8. Evaluate once on test data.

Compare one-shot and recursive rankings. Produce accuracy-versus-size and conditional-violation-versus-size Pareto curves.

### Phase 5 - Required baselines

- All features
- Random removal at matched subset sizes
- Mutual information
- Permutation importance
- Standard RFE
- SHAP ranking where feasible
- Closest reproducible conformal feature-selection method, especially CRFE
- Proposed constrained/Pareto conformal-harm method

All methods must receive the same data budget, subset sizes, models, and evaluation protocol.

### Phase 6 - Interaction with scaling

For all-features, standard-selection, and proposed-selection variants, test:

| Scaling | Score |
|---|---|
| Base / TS / ConfTS | APS / RAPS |

Analyze whether gains are additive or redundant and how intervention changes temperature, threshold, rankings, and probability concentration.

### Phase 7 - Real datasets

Recommended order:

1. Dry Bean
2. Covertype
3. Human Activity Recognition
4. One carefully framed medical dataset
5. Adult or another mixed-type dataset if subgroup analysis becomes central

Do not rely only on tiny binary tasks; APS/RAPS efficiency is more informative when several labels are plausible. Learn preprocessing from training data only.

### Phase 8 - Robustness and statistics

- [x] Freeze 10 initial seeds and a precision-based rule for increasing to 20.
- [x] Enforce identical splits, seeds, uniforms, and ConfTS folds within pairs.
- [x] Preserve subject-disjoint 15/5/5/5 HAR splitting.
- [x] Freeze `alpha` at 0.10 and 0.05 for sensitivity.
- [x] Freeze the RAPS `lambda` and `k_reg` grid.
- [x] Freeze allowed accuracy-loss and 1--5-removal grids.
- [x] Retain logistic regression and the small neural network.
- [x] Implement paired effects, intervals/tests, multiplicity, and rank stability.
- [x] Implement subject-level HAR two-way seed/subject uncertainty.
- [x] Implement atomic, manifest-validated resume checkpoints.
- [ ] Run the approved expensive benchmark.

## 6. Key code interfaces

```python
def fit_classifier(X_train, y_train, config): ...
def tune_temperature(logits_tune, y_tune): ...
def tune_confts(logits_tune, y_tune, alpha, score_name): ...
def calibrate_threshold(prob_cal, y_cal, alpha, score_name, rng): ...
def evaluate_prediction_sets(prob_test, y_test, tau, score_name, rng): ...
def ablate_and_evaluate(feature_id, split, model_config, cp_config): ...
def compute_harm_metrics(reference_result, intervention_result): ...
def select_feature_subset(tuning_results, constraints): ...
```

Every output row should include dataset, model, seed, split ID, selected features, temperature, threshold, score, alpha, RAPS parameters, and code version.

## 7. Recommended repository structure

```text
conformal-harmful-features/
├── README.md
├── paper/                  # base paper, related work, claims/evidence
├── reproduction/           # unchanged verified reproduction
├── configs/
├── src/                    # data, models, scaling, CP, metrics, selection
├── experiments/            # numbered executable studies
├── tests/                  # APS/RAPS, splits, scores, regression tests
└── outputs/                # tables, figures, logs
```

## 8. Essential paper evidence

1. Classification importance versus conformal harm scatterplot.
2. Accuracy-size Pareto curves.
3. Progressive-removal curves showing coverage and SSCV beside size.
4. Base/TS/ConfTS interaction plot.
5. Main benchmark table with accuracy, ECE, coverage, size, SSCV, and class gap.
6. Rank-stability table across seeds and models.

## 9. Main risks and safeguards

| Risk | Safeguard |
|---|---|
| Smaller sets caused by accuracy collapse | Accuracy constraints and full Pareto curves |
| Undercoverage after selection | Fresh calibration after every choice is frozen |
| Method duplicates ordinary importance | Rank correlations and matched-subset comparisons |
| Method duplicates CRFE | Reproduce closest baseline and define objective/protocol differences precisely |
| One-dataset or one-seed effect | Multiple datasets, paired seeds, two model families |
| Extreme ConfTS creates zeros | Stable softmax, zero-count logs, preregistered numerical rule |

## 10. Decision gates

1. **Phenomenon:** Does a repeatable accuracy-conformal mismatch exist?
2. **Distinctness:** Does the ranking differ from ordinary importance and CRFE?
3. **Utility:** Does intervention improve a Pareto frontier?
4. **Publishability:** Does an updated literature comparison support a precise novelty claim?

If Gate 1 fails, stop rather than inventing a complex algorithm. If Gate 2 fails, consider converting the work into a diagnostic study or combining it with the shortcut project.

## 11. First executable milestone

1. Generate one 20-feature, 10-class synthetic dataset.
2. Create train/tune/conformal/test splits.
3. Train the full-feature classifier.
4. Evaluate Base/TS/ConfTS with APS/RAPS.
5. Remove each feature separately and retrain.
6. Save one CSV containing all metric deltas.
7. Plot accuracy loss against APS-size reduction.
8. Look for candidates with tiny accuracy loss and large set-size reduction.

Only after this plot supports the phenomenon should the project name a harm score or build recursive selection.

## 12. Paper outline

1. Introduction and motivating example
2. APS/RAPS and post-hoc scaling background
3. Related feature-selection and conformal-selection work
4. Definition and method
5. Validity-preserving selection protocol
6. Synthetic experiments
7. Real-data experiments and baselines
8. TS/ConfTS interaction
9. Conditional behavior, limitations, and negative results
10. Conclusion

## 13. Claim discipline and base-paper connection

Safe initial wording:

> We investigate whether feature contributions to classification performance can diverge from their effects on adaptive conformal efficiency and reliability.

Avoid claiming to be the first feature-selection method for conformal prediction.

Xi et al. intervene after logits are produced and optimize a post-hoc temperature for conformal efficiency. This project moves upstream to test whether input features create avoidable conformal inefficiency and whether intervention is complementary to ConfTS. Preserve the base paper's held-out tuning, fresh conformal calibration, APS/RAPS metrics, and numerical-stability checks.
