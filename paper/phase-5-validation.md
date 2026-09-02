# Phase 5 Validation Record

**Status:** Complete

**Completed:** 2026-09-02

**Scope:** Compare the proposed conformal-harm subsets against all required
baselines at the subset sizes frozen in Phase 4, with identical training,
tuning, conformal-calibration, and test budgets.

## Implemented baselines

| Method | Ranking evidence | Implementation |
|---|---|---|
| All features | None | Full 20-feature reference |
| Random | Random only | 10 independent subsets per matched size |
| Mutual information | Train | `mutual_info_classif` |
| Permutation importance | Tune | Accuracy loss over 10 paired permutations of the fitted full model |
| Standard RFE | Train | Coefficient-based logistic-regression RFE |
| SHAP | Train background + tune evaluation | Model-agnostic permutation SHAP with an explicit evaluation cap |
| CRFE | Train + tune | Published multiclass beta elimination rule, `Lambda=0.5` |
| Proposed | Repeated cross-fitting inside tune | Phase 4 one-shot and recursive constrained Base APS paths |

RFE and CRFE use a shared linear selector so their chosen subset is independent
of the downstream classifier family. Permutation importance and SHAP are fitted-
model-specific. CRFE is matched to the proposed subset sizes rather than using
its optional stopping rule; this isolates ranking quality from subset-size
choice.

The CRFE implementation follows the [authors' paper](https://arxiv.org/abs/2405.19429)
and [released implementation](https://github.com/digital-medicine-research-group-UNAV/CRFE): one-vs-
rest linear SVM weights are combined with held-out feature values to compute
beta, the largest-beta feature is removed, and the model is refit recursively.
This is distinct from the proposed method, which directly estimates APS-size
gain subject to accuracy and coverage gates.

## Valid protocol

1. Every selector uses only training and, where required, tuning data.
2. The proposed selections are regenerated through the complete Phase 4
   tuning-only procedure.
3. Comparator subset sizes exactly match the proposed frozen sizes: 15 and 16
   features for logistic regression and 19 for the small neural network.
4. Every frozen subset receives a fresh threshold from the untouched outer
   conformal-calibration split.
5. Each subset is then evaluated once on the same untouched test split using
   paired randomized-APS uniforms.
6. Phase 5 is deliberately restricted to Base APS. The six-way scaling and
   APS/RAPS interaction is the subject of Phase 6.

## Final matched-size results

### Logistic regression

| Method | Features | Accuracy | Coverage | Mean APS size | Size reduction vs all |
|---|---:|---:|---:|---:|---:|
| All features | 20 | 0.7615 | 0.8910 | 1.8410 | 0.0000 |
| CRFE | 15 | 0.7660 | 0.8850 | **1.8030** | **0.0380** |
| SHAP | 15 | 0.7650 | 0.8870 | 1.8220 | 0.0190 |
| Proposed one-shot | 15 | 0.7645 | 0.8905 | 1.8360 | 0.0050 |
| Mutual information | 15 | 0.7640 | 0.8905 | 1.8425 | -0.0015 |
| Permutation importance | 15 | 0.7645 | 0.8905 | 1.8425 | -0.0015 |
| RFE | 15 | 0.7635 | 0.8910 | 1.8430 | -0.0020 |
| Random (mean of 10) | 15 | 0.7204 | 0.8961 | 2.0604 | -0.2194 |
| SHAP | 16 | 0.7640 | 0.8885 | **1.8165** | **0.0245** |
| Mutual information | 16 | 0.7655 | 0.8865 | 1.8185 | 0.0225 |
| Permutation importance | 16 | 0.7650 | 0.8875 | 1.8275 | 0.0135 |
| CRFE | 16 | 0.7635 | 0.8870 | 1.8310 | 0.0100 |
| RFE | 16 | 0.7650 | 0.8920 | 1.8460 | -0.0050 |
| Proposed recursive | 16 | 0.7650 | 0.8910 | 1.8505 | -0.0095 |
| Random (mean of 10) | 16 | 0.6744 | 0.8950 | 2.3408 | -0.4998 |

At 15 features, CRFE is the strongest efficiency baseline but its empirical
coverage is 0.005 below the all-feature result and 0.015 below the nominal
target. The proposed one-shot subset preserves the reference coverage more
closely but produces only a 0.005 size reduction. At 16 features, SHAP and
mutual information both outperform the proposed recursive subset on size.

### Small neural network

| Method | Features | Accuracy | Coverage | Mean APS size | Size reduction vs all |
|---|---:|---:|---:|---:|---:|
| All features | 20 | 0.7460 | 0.8935 | 1.9095 | 0.0000 |
| Proposed one-shot / recursive | 19 | 0.7460 | 0.8885 | **1.8835** | **0.0260** |
| CRFE | 19 | 0.7495 | 0.8970 | 1.9105 | -0.0010 |
| Mutual information | 19 | 0.7480 | 0.8910 | 1.9240 | -0.0145 |
| Permutation importance | 19 | 0.7480 | 0.8910 | 1.9240 | -0.0145 |
| RFE | 19 | 0.7470 | 0.8950 | 1.9305 | -0.0210 |
| SHAP | 19 | 0.7525 | 0.8970 | 1.9375 | -0.0280 |
| Random (mean of 10) | 19 | 0.7384 | 0.8964 | 1.9838 | -0.0743 |

The proposed subset is the only 19-feature method that reduces mean APS size,
and it does so without changing test accuracy. Its empirical coverage is 0.005
below the all-feature reference; all values are finite-sample estimates rather
than enforced equality to 0.90.

## Validation and decision

The executable protocol record passes every check: all nine method labels are
present, 30 random comparator subsets cover every model/target-size pair, all
non-reference subsets match a proposed size, only Base APS is evaluated, every
threshold and reported metric is finite, and all subsets are marked frozen
before final calibration.

**Phase 5 decision:** the required baseline comparison is complete, but the
utility gate remains model-dependent. The proposed method wins clearly for the
small neural network, while CRFE and SHAP outperform it for logistic regression.
We therefore cannot claim general superiority. Phase 6 should test whether the
neural-network advantage and logistic-regression weakness persist under TS,
ConfTS, and RAPS. Multi-seed conclusions remain deferred to Phase 8.
