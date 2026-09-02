# Phase 2 Validation Record

**Status:** Complete

**Completed:** 2026-09-02

**Scope:** Measure one-feature-at-a-time effects without selecting a subset or defining a conformal-harm score.

## Intervention protocols

Both protocols recreate the Phase 1 dataset and exact split (`8b6eb4adb36bfe27`) and reuse the same calibration/test random uniforms for paired APS/RAPS comparisons.

### Retrain ablation

For each feature and model family:

1. Remove the feature from training, tuning, calibration, and test matrices.
2. Learn preprocessing and refit the classifier on the reduced training matrix.
3. Retune TS and score-specific ConfTS on tuning data.
4. Compute a fresh randomized APS/RAPS threshold on calibration data.
5. Evaluate once on test data.

This produced 252 rows: 12 full-feature references plus `20 features x 2 models x 2 scores x 3 scaling methods`.

### Fixed-model sensitivity

The full-feature classifier and its training-fitted scaler remain frozen. Each feature receives two diagnostics:

- **Mean mask:** replace the feature by its training-partition mean. Under the training-fitted standardizer this is zero in standardized coordinates.
- **Permutation:** shuffle feature values independently within tuning, calibration, and test. No value crosses a partition boundary.

TS/ConfTS are retuned on the intervened tuning data, and a fresh threshold is computed on the correspondingly intervened calibration data. This produced 492 rows: 12 references plus `20 features x 2 interventions x 2 models x 2 scores x 3 scaling methods`.

Masking is an inference-time sensitivity diagnostic. It is not treated as a substitute for retraining.

## Stored deltas

Every intervention row contains its matched full-feature values and signed deltas for accuracy, macro-F1, NLL, ECE, coverage, mean/median/90th-percentile set size, empty/full-set rate, SSCV, class-coverage gap, and maximum class-coverage deviation. Positive `mean_size_reduction` means the intervention produced smaller sets.

For the Phase 2 go/no-go decision only, the validation report marks a descriptive test pattern when, in both model families:

- maximum accuracy loss is at most 0.01;
- the absolute coverage change is at most 0.01 for every Base/TS/ConfTS x APS/RAPS row;
- Base APS mean size decreases.

This mark is not a ranking or feature-selection input, and Phase 3 is prohibited from consuming it. Phase 3 must construct its constrained, weighted, and Pareto evidence from tuning data only.

## Main retrain-ablation signal

The strongest cross-model Base APS results matching the descriptive pattern were:

| Removed feature | Role | Mean accuracy loss | Mean size reduction | Maximum absolute coverage change |
|---|---|---:|---:|---:|
| `redundant_00` | Redundant | -0.0025 | 0.0403 | 0.0050 |
| `noise_00` | Noise | -0.0005 | 0.0390 | 0.0090 |
| `strong_03` | Strong with redundant copy | -0.0045 | 0.0270 | 0.0080 |
| `noise_01` | Noise | -0.0032 | 0.0225 | 0.0085 |
| `redundant_03` | Redundant | -0.0050 | 0.0208 | 0.0050 |
| `redundant_02` | Redundant | 0.0000 | 0.0203 | 0.0065 |
| `noise_02` | Noise | -0.0025 | 0.0090 | 0.0060 |

Negative accuracy loss means accuracy improved after removal. The `strong_00`, `strong_02`, and `strong_03` ablations must not be interpreted as evidence that strong signal is harmful: each has a highly correlated redundant copy, so retraining can preserve its information through the copy.

Role-level Base APS means provide the expected control pattern:

| Role | Mean accuracy loss | Mean size reduction | Mean coverage change |
|---|---:|---:|---:|
| Noise | -0.0019 | 0.0141 | -0.0027 |
| Redundant | -0.0019 | 0.0201 | -0.0022 |
| Weak | 0.0061 | -0.0192 | -0.0013 |
| Strong | 0.0206 | -0.0992 | -0.0031 |

Thus noise/redundant removals tend to improve efficiency with negligible accuracy cost, while removing genuinely unreplaceable signal tends to reduce accuracy and enlarge sets.

## Fixed-model findings

Several noise diagnostics match the same descriptive pattern. The clearest mean-mask effects were `noise_04` (0.0280 mean Base APS size reduction), `noise_03` (0.0170), `noise_00` (0.0158), `noise_02` (0.0135), and `noise_01` (0.0115), all with at most 0.01 accuracy loss in either model. Permutation also matches for `noise_04`, `noise_03`, and `noise_02`.

`noise_00`, `noise_01`, and `noise_02` match under both retrain ablation and mean masking. `noise_02` is the only feature that matches under retraining, mean masking, and permutation across both models.

The contrast between the two protocols is scientifically useful. Removing `redundant_00` and retraining reduces Base APS size by 0.0403 on average, but masking it in the frozen model increases size by 0.0503 and permutation increases size by 0.1490. Retraining can redirect weight to the correlated source/copy; a frozen model cannot. This validates the roadmap's requirement to report the protocols separately.

## Conditional reliability

The current signal is mainly efficiency, not conditional reliability. Across noise-feature retrain ablations, mean Base APS SSCV changes by `+0.0038` and the class-coverage gap by `+0.0025`; positive values are worse. Mean masking and permutation are similarly mixed. H4 is therefore not supported at this phase and remains an empirical question rather than a claim.

## Decision gate

**Gate 1, preliminary result: continue.** The experiment contains a controlled, directionally sensible accuracy-efficiency mismatch, and `noise_02` survives all three intervention views across both classifiers. This is sufficient to implement Phase 3 formulations.

This is not yet evidence of cross-seed stability or publication-level utility. The result uses one configured seed and one synthetic construction. Multi-seed inference, rank stability, real datasets, ordinary feature-selection baselines, and the closest conformal baseline remain later phases.

## Verification result

```text
24 passed
Phase 1 regression after shared-pipeline refactor: PASS (12 rows exact)
Retrain-ablation protocol: PASS (252 rows)
Fixed-model masking protocol: PASS (492 rows)
Deterministic scientific-column rerun: PASS (252 + 492 rows exact)
```

## Phase boundary

Phase 2 does not choose a subset, tune a formal harm definition, or use calibration/test results for a feature decision. Phase 3 starts by implementing and comparing constrained-efficiency, weighted, and Pareto formulations on tuning-derived evidence.
