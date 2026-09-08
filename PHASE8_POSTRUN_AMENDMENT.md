# Phase 8 Post-run Execution Amendment: Numerical-Failure Retention

**Scope:** execution/reporting only. The frozen Phase 8 scientific design is not changed.

**Original scientific commit:** `e4b36459f5030eb49326d129c9cf8b0c761e8d77`

**Original benchmark run:** `34101494390`

## Why this amendment is necessary

The frozen Phase 8 runner used exact 0/1 probability diagnostics as a unit-level
hard-stop. That was appropriate as a guard against silently accepting numerical
pathology, but the benchmark showed that a hard-stop also prevents the pathology
itself from being reported as an experimental result.

Two distinct cases were observed after the benchmark:

1. Covertype can contain an isolated exact-boundary probability already at Base
   (`T=1`). A unit-level absolute-zero rule therefore rejects the entire unit
   even when TS/ConfTS do not introduce the boundary event.
2. In recovered HAR seed 43, ConfTS introduced additional exact-one test
   probabilities relative to the matching Base cells. This was not merely a
   bookkeeping artifact: the affected ConfTS cells also showed material
   undercoverage and nonzero empty-set rates. The event must therefore remain
   visible rather than be waived or repaired using final calibration/test data.

The second case is consistent with the numerical-risk motivation already used
when the protocol enabled saturation rejection during ConfTS tuning. The final
held-out data can still reveal a failure not present on the tuning partition.

## Uniform recovery rule

The recovery applies the following rule to every recovered dataset/seed unit,
without looking at whether the resulting effect favors any method:

- preserve every prespecified grid cell and its original temperature;
- preserve the raw exact-zero/exact-one counts unchanged;
- do **not** retune a temperature from final calibration or test data;
- do **not** replace a flagged scaling with Base or another temperature;
- do **not** drop a flagged cell from the stored results;
- add per-cell flags for any raw boundary event and for boundary counts induced
  above the matching Base (`T=1`) condition;
- continue to enforce all non-numerical structural, split, selection, grid, and
  checkpoint-provenance checks;
- mark the grid protocol `COMPLETE_WITH_NUMERICAL_FLAGS` when the original
  strict numerical rule would have failed.

A dataset/seed completion record may still be `PASS`: here `PASS` means the
prespecified computation is complete and provenance-valid, not that every
scaling setting is numerically safe.

## Interpretation

Numerically flagged scaled cells remain part of the robustness record but must
not be used to claim safe scaling-based efficiency improvement. Their failure is
reported as a robustness result.

The primary Phase 8 estimand remains **Base APS at alpha=0.10**. No primary
feature subset is reselected and no primary paired contrast is changed by this
amendment. Observed marginal coverage, including any HAR undercoverage under
subject-disjoint splitting, must be reported rather than corrected post hoc.

This amendment is intentionally conservative: it changes an execution
hard-stop into transparent result retention. It does not change data, models,
seeds, splits, feature-selection rules, RAPS settings, alpha values, or the
pre-specified 10-to-20-seed precision rule.
