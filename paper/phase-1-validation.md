# Phase 1 Validation Record

**Status:** Complete

**Completed:** 2026-09-02

**Scope:** Establish a controlled full-feature synthetic baseline before any feature intervention.

## Controlled dataset

The configured dataset has 12,000 balanced examples, 10 classes, and 20 features with fixed, auditable roles:

| Role | Count | Construction |
|---|---:|---|
| Strong | 6 | Class-specific mean signal with scale 1.00 plus independent Gaussian noise |
| Weak | 4 | Class-specific mean signal with scale 0.25 plus independent Gaussian noise |
| Redundant | 4 | Noisy copies of named strong source features |
| Pure noise | 6 | Independent Gaussian noise with no label dependence |

Feature order is fixed and recorded in `feature_manifest.csv`. A training-only recovery check uses the between-class/total-variance ratio plus redundant-source correlations. It does not select or remove features.

Observed median class-separation ratios were:

| Strong | Weak | Redundant | Noise |
|---:|---:|---:|---:|
| 0.4967 | 0.0604 | 0.4871 | 0.0017 |

All redundant-source absolute correlations exceeded 0.90. The role-recovery check passed.

## Valid protocol

- Train/tune/calibration/test sizes: `6000 / 2000 / 2000 / 2000`.
- Split ID: `8b6eb4adb36bfe27`; exact indices and seed 42 are persisted.
- Standardization and classifier fitting use training data only.
- Ordinary TS minimizes tuning-set NLL.
- ConfTS uses disjoint threshold/loss subsets of the tuning partition and is tuned separately for APS and RAPS.
- Once temperature is frozen, a fresh threshold is computed from the calibration partition.
- Test data is used only for the final reported metrics.
- Identical calibration/test uniform draws are reused across methods for paired randomized APS/RAPS comparisons.

## Models and metrics

The two classifier families are multinomial logistic regression and a small MLP with hidden widths 64 and 32.

Each of the 12 result rows records accuracy, macro-F1, NLL, 15-bin ECE, marginal coverage, mean/median/90th-percentile set size, empty/full-set rates, SSCV, class-coverage range and maximum target deviation, temperature, threshold, RAPS parameters, split ID, selected features, numerical diagnostics, fit time, and code version.

SSCV is the maximum absolute deviation from target coverage across size bins `[0,1]`, `[2,3]`, and `[4,10]`, ignoring bins with fewer than 25 examples. This avoids treating a small exact-cardinality group as a stable stratum.

## Full-feature baseline

| Model | Scaling | Score | T | Accuracy | ECE | Coverage | Mean size | SSCV | Class gap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic | Base | APS | 1.0 | 0.7615 | 0.0221 | 0.8850 | 1.8595 | 0.0311 | 0.110 |
| Logistic | TS | APS | 1.0 | 0.7615 | 0.0221 | 0.8850 | 1.8595 | 0.0311 | 0.110 |
| Logistic | ConfTS | APS | 0.4 | 0.7615 | 0.1407 | 0.8980 | 1.6835 | 0.0473 | 0.170 |
| Logistic | Base | RAPS | 1.0 | 0.7615 | 0.0221 | 0.8835 | 1.8460 | 0.0314 | 0.115 |
| Logistic | TS | RAPS | 1.0 | 0.7615 | 0.0221 | 0.8835 | 1.8460 | 0.0314 | 0.115 |
| Logistic | ConfTS | RAPS | 0.4 | 0.7615 | 0.1407 | 0.8945 | 1.6645 | 0.0667 | 0.175 |
| Small MLP | Base | APS | 1.0 | 0.7460 | 0.0268 | 0.8870 | 1.9185 | 0.0275 | 0.105 |
| Small MLP | TS | APS | 1.1 | 0.7460 | 0.0201 | 0.8865 | 1.9360 | 0.0302 | 0.105 |
| Small MLP | ConfTS | APS | 0.4 | 0.7460 | 0.1653 | 0.8915 | 1.6960 | 0.0262 | 0.190 |
| Small MLP | Base | RAPS | 1.0 | 0.7460 | 0.0268 | 0.8870 | 1.9090 | 0.0269 | 0.105 |
| Small MLP | TS | RAPS | 1.1 | 0.7460 | 0.0201 | 0.8865 | 1.9235 | 0.0297 | 0.105 |
| Small MLP | ConfTS | RAPS | 0.4 | 0.7460 | 0.1653 | 0.8915 | 1.6955 | 0.0272 | 0.185 |

These rows are a baseline, not evidence of a harmful feature. ConfTS produced smaller sets here but worse ECE and, for the logistic model, worse conditional metrics. That trade-off is retained rather than summarized as an unconditional improvement.

## Stability gates

The executable experiment fails unless all gates pass:

1. Exactly 12 expected result rows are present.
2. All core metrics are finite.
3. Every marginal coverage is within 0.03 of the 0.90 target.
4. Both classifiers exceed the preregistered 0.50 accuracy floor.
5. Calibration/test probabilities contain no exact zeros or exact-one maxima.

All five gates passed. A second complete run reproduced all scientific result columns exactly; only elapsed fit time was excluded from the equality check.

## Verification result

```text
20 passed
Synthetic feature-role check: PASS
Phase 1 baseline stability checks: PASS
Deterministic repeat check: PASS (12 rows)
```

## Phase boundary

No feature was ranked, removed, masked, or selected in Phase 1. Phase 2 starts with one-feature-at-a-time retraining ablations and a separate fixed-model masking/permutation diagnostic under an explicitly documented calibration protocol.
