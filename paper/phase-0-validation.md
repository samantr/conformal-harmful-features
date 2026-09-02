# Phase 0 Validation Record

**Status:** Complete  
**Completed:** 2026-09-02  
**Scope:** Freeze and modularize the verified base-paper reproduction foundation.

## Preserved source

The scripts in `reproduction/` were not edited. They remain the historical learning and reproduction record for *Does confidence calibration improve conformal prediction?*

## Implemented interfaces

- `probabilities_from_logits`: Base (`T=1`) and temperature-scaled probabilities using max-shifted softmax.
- `tune_temperature`: ordinary TS selected by minimum tuning-set NLL.
- `tune_confts`: grid ConfTS using disjoint threshold and loss subsets drawn only from the tuning partition.
- `nonrandomized_score_matrix`: non-randomized APS/RAPS scores used for ConfTS optimization.
- `randomized_score_matrix`: standard randomized APS/RAPS scores used for final calibration and prediction.
- `calibrate_threshold`: finite-sample threshold from the dedicated calibration partition.
- `evaluate_prediction_sets`: coverage, mean/median/90th-percentile size, and empty/full-set rates.
- `save_split_artifact` / `load_split_artifact`: exact four-way indices, split seed, and audit metadata.
- `probability_diagnostics`: exact-zero probabilities, exact-one maxima, and mean maximum probability.

## Mathematical conventions frozen by tests

For a candidate label at sorted rank `r`, one uniform value `u` is shared by all candidate labels in the same sample:

```text
randomized APS = probability mass before r + u * probability at r
randomized RAPS = randomized APS + lambda * max(r - k_reg, 0)
```

The split-conformal threshold uses order-statistic rank:

```text
ceil((n + 1) * (1 - alpha))
```

ConfTS uses non-randomized scores during temperature optimization. Final calibration and evaluation use randomized scores with explicitly supplied random-number generators or uniform values.

## Regression tolerance

- Deterministic float64 APS/RAPS score matrices: `rtol=1e-12`, `atol=1e-12` against formulas copied from the frozen reproduction.
- Hand-calculated APS/RAPS and conformal-quantile examples: exact or `pytest.approx` floating-point equality.
- Split artifacts: exact index-array, seed, and metadata equality after save/load.
- Float32 numerical diagnostics: exact zero and exact saturation counts.

Stable class sorting is used to make equal-probability ties deterministic. The frozen scripts used NumPy's default sort; therefore regression arrays deliberately contain no ties.

## Verification result

```text
14 passed
```

The synthetic smoke test also produced four disjoint partitions of sizes `6000 / 2000 / 2000 / 2000`, saved the artifact, and reloaded the same seed (`42`), metadata, and index counts.

## Phase boundary

No feature intervention, harm ranking, or result claim was introduced in Phase 0. Phase 1 begins with a stable full-feature synthetic baseline using two classifier families and all Base/TS/ConfTS x APS/RAPS combinations.
