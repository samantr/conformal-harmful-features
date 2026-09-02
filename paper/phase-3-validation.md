# Phase 3 Validation Record

**Status:** Complete

**Completed:** 2026-09-02

**Scope:** Define and compare constrained, weighted, and Pareto formulations of
single-feature conformal harm without using the outer calibration or test
partitions for any ranking decision.

## Selection-only evidence protocol

Every candidate is still a validity-preserving retrain ablation: the classifier
and training-fitted standardizer use only the outer training partition. Harm
evidence comes only from the 2,000-example outer tuning partition.

To avoid reusing the same labels for temperature choice, conformal calibration,
and harm evaluation, Phase 3 uses five seeded repetitions of four-fold cross-fitting
inside the tuning partition. In each fold:

1. one fold is isolated as selection evaluation data;
2. the remaining rows are stratified into scaling-tuning (60%) and selection
   calibration (40%) partitions;
3. Base/TS/ConfTS and APS/RAPS are evaluated using a fresh inner threshold;
4. paired randomized APS/RAPS uniforms are reused across the reference and every
   ablation;
5. every outer-tuning row is evaluated exactly once per repetition.

The outer conformal-calibration and test indices are created to preserve the
original split, but their features and labels are never read by the Phase 3
ranking computation. The run produced 5,040 fold-level evidence rows, 1,200
cross-fitted resample rows, and 240 aggregated feature/pipeline ranking rows.

## Definitions

For reference model `0` and feature-removal intervention `j`:

```text
efficiency_gain(j)    = mean_size(0) - mean_size(j)
accuracy_loss(j)      = accuracy(0) - accuracy(j)
coverage_shortfall(j) = max(0, 1 - alpha - coverage(j))
conditional_violation(j)
                      = max(SSCV(j), maximum class-coverage target deviation(j))
```

Coverage shortfall is one-sided. Overcoverage is not a validity failure and its
efficiency cost is already represented by prediction-set size.

### A. Constrained efficiency — selected primary definition

A feature is eligible in a model/scaling/score pipeline only when every one of
the five cross-fitted resamples satisfies:

- accuracy loss at most `0.01`;
- marginal coverage shortfall at most `0.03`.

Eligible features are ranked by mean efficiency gain. An eligible feature is
marked a **conformal-harm candidate** only if its mean efficiency gain is also
strictly positive. This prevents the method from calling a feature harmful when
its removal enlarges sets.

### B. Weighted score — supporting sensitivity analysis

```text
weighted_harm(j) = efficiency_gain(j)
                   - 4 * accuracy_loss(j)
                   - 10 * coverage_shortfall(j)
                   - 1 * conditional_violation(j)
```

The weights are explicit configuration values fixed before outer calibration or
test evaluation. In label-count units, a one-percentage-point accuracy loss costs
`0.04`, and a one-percentage-point coverage shortfall costs `0.10`. The weighted
score is reported with the same constraint flags rather than being allowed to
silently override validity failures.

### C. Pareto ranking — supporting frontier analysis

Candidates exceeding the coverage-shortfall limit are excluded. Remaining
candidates receive non-dominated fronts over three separately reported objectives:

1. minimize accuracy loss;
2. maximize efficiency gain;
3. minimize conditional violation.

No ratio is used, so near-zero accuracy losses cannot create numerical explosions.

## Main tuning-only result

The consensus across both classifiers, Base/TS/ConfTS, and APS/RAPS is led by
redundant and noise features:

| Feature | Role | Pipelines eligible | Pipelines with positive constrained harm | Mean size reduction | Mean accuracy loss | Constrained consensus rank | Weighted rank | Pareto rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `redundant_02` | Redundant | 12/12 | 10/12 | 0.0237 | 0.0005 | 1 | 1 | 1 |
| `redundant_00` | Redundant | 12/12 | 10/12 | 0.0104 | 0.0010 | 2 | 6 | 7 |
| `redundant_03` | Redundant | 12/12 | 11/12 | 0.0095 | 0.0005 | 3 | 3 | 5 |
| `noise_04` | Noise | 12/12 | 10/12 | 0.0056 | -0.0008 | 4 | 2 | 2 |
| `noise_01` | Noise | 12/12 | 8/12 | 0.0051 | 0.0013 | 5 | 5 | 6 |
| `noise_00` | Noise | 12/12 | 6/12 | 0.0167 | 0.0000 | 6 | 4 | 3 |

Negative accuracy loss means that accuracy improved after removal. At role level,
redundant features have the best mean constrained rank (`6.15`), followed by
noise (`7.29`), strong (`13.56`), and weak (`15.08`). This is directionally
consistent with the controlled construction. Replaceable strong features may
still rank above their role average because the redundant copy preserves their
signal.

For Base APS, the logistic-regression ranking starts with `weak_01`, `noise_01`,
and `noise_04`; the MLP ranking starts with `redundant_02`, `noise_00`, and
`redundant_03`. The difference confirms that conformal harm is a property of a
feature-model-pipeline combination, not a universal intrinsic feature label.

## Formulation stability and agreement

Mean Spearman rank correlation across the five selection-resample seeds was:

| Formulation | Mean across 12 pipelines | Worst pipeline mean |
|---|---:|---:|
| Constrained | 0.719 | 0.534 |
| Weighted | 0.676 | 0.590 |
| Pareto total order | 0.420 | 0.247 |

The constrained primary definition passes the preregistered `0.50` mean-correlation
gate in every pipeline. Constrained and weighted rankings agree strongly (mean
correlation `0.849`, range `0.741`-`0.907`).

Pareto front membership is less stable, with mean pairwise front-one Jaccard
overlap ranging from `0.204` to `0.529` across pipelines. Pareto results are
therefore retained as frontier evidence but not used as a strict primary feature
order. This instability is a result, not suppressed by tie-breaking or a changed
post-hoc threshold.

## Decision and phase boundary

**Phase 3 decision: use constrained efficiency as the primary operational
definition.** It is interpretable, explicitly blocks severe accuracy loss and
marginal undercoverage, preserves a positive-efficiency requirement for the
harmful-feature label, and is stable across the current selection resamples.

Weighted scores remain a configurable sensitivity analysis. Pareto fronts remain
important for reporting trade-offs but are not sufficiently stable to define a
single removal order at this stage.

All executable protocol checks passed, and 30 tests pass. Stability here refers
to repeated partitions of the fixed outer tuning set. Full dataset-generation
seed stability, confidence intervals, and real-data replication remain required
in Phase 8 and must not be inferred from this Phase 3 result.

Phase 4 may now use the constrained tuning-only ranking to perform progressive
feature removal, freeze the subset, calculate one fresh outer-calibration
threshold, and evaluate once on the untouched test partition.
