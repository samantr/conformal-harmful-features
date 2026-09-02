# Conformal Harmful Feature Detection

Research code for testing whether features with little classification value can disproportionately harm the efficiency or conditional reliability of APS/RAPS prediction sets.

## Current milestone

**Phase 2 - Single-feature interventions: COMPLETE.** Every feature now receives a validity-preserving retrain ablation plus separately labeled fixed-model mean-mask and permutation diagnostics. All paths retune scaling on tuning data, compute a fresh threshold on calibration data, evaluate once on test data, and store paired deltas from the untouched full-feature reference.

Next: **Phase 3 - Define conformal harm**. Compare constrained-efficiency, weighted, and Pareto formulations without using calibration or test data for feature decisions.

## Stage tracker

| Stage | Status |
|---|---|
| Phase 0 - Freeze reproduction foundation | ✅ Complete |
| Phase 1 - Controlled synthetic data | ✅ Complete |
| Phase 2 - Single-feature interventions | ✅ Complete |
| Phase 3 - Define conformal harm | ⏭️ Next |
| Phase 4 - Progressive subset selection | Not started |
| Phase 5 - Required baselines | Not started |
| Phase 6 - Interaction with scaling | Not started |
| Phase 7 - Real datasets | Not started |
| Phase 8 - Robustness and statistics | Not started |

## Scientific rule

The four partitions have distinct roles:

- `train`: fit model and preprocessing
- `tune`: feature ranking, subset size, TS/ConfTS and hyperparameters
- `calibration`: compute a fresh final conformal threshold after all choices are frozen
- `test`: final evaluation only

No feature decision may use calibration or test results.

## Layout

```text
paper/          source paper, roadmap, related-work notes
reproduction/   frozen scripts from the completed base-paper reproduction
configs/        experiment settings
src/chf/        reusable research package
experiments/    numbered executable studies
tests/          mathematical, split-integrity and regression tests
outputs/        generated tables, figures and logs (not source data)
```

## Start

```bash
python -m pip install -e ".[dev]"
pytest
python experiments/01_synthetic_baseline.py --config configs/synthetic_debug.yaml
python experiments/02_single_feature_ablation.py --config configs/synthetic_debug.yaml
python experiments/03_masking_sensitivity.py --config configs/synthetic_debug.yaml
```

The first experiment generates the controlled 10-class/20-feature dataset, verifies its known feature roles using training data only, persists the exact four-way split, trains both model families, and writes all 12 Base/TS/ConfTS x APS/RAPS result rows to `baseline_results.csv`.

Phase 0 validation details and tolerances are recorded in [`paper/phase-0-validation.md`](paper/phase-0-validation.md).
Phase 1 protocol, metrics, results, and stability checks are recorded in [`paper/phase-1-validation.md`](paper/phase-1-validation.md).
Phase 2 intervention protocols, exploratory results, and decision gate are recorded in [`paper/phase-2-validation.md`](paper/phase-2-validation.md).

## Planned experiment sequence

1. `01_synthetic_baseline.py`: full-feature Base/TS/ConfTS with APS/RAPS.
2. `02_single_feature_ablation.py`: retrain once per removed feature.
3. `03_masking_sensitivity.py`: fixed-model masking/permutation diagnostic.
4. `04_harm_ranking.py`: constrained and Pareto rankings using tuning data.
5. `05_progressive_selection.py`: freeze a subset, recalibrate, evaluate once.
6. `06_real_data_benchmarks.py`: paired-seed comparison against standard baselines.
