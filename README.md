# Conformal Harmful Feature Detection

Research code for testing whether features with little classification value can disproportionately harm the efficiency or conditional reliability of APS/RAPS prediction sets.

## Current milestone

**Phase 0 - Freeze the reproduction foundation: COMPLETE.** The original learning/reproduction scripts remain unchanged in `reproduction/`. Reusable and tested implementations of Base/TS/ConfTS, randomized APS/RAPS, the finite-sample conformal quantile, numerical diagnostics, and split artifacts now live in `src/chf/`.

Next: **Phase 1 - Controlled synthetic data**. Complete the full-feature Base/TS/ConfTS x APS/RAPS baseline with logistic regression and a small neural network. Do not begin feature ranking or ablation until that baseline is stable.

## Stage tracker

| Stage | Status |
|---|---|
| Phase 0 - Freeze reproduction foundation | ✅ Complete |
| Phase 1 - Controlled synthetic data | ⏭️ Next |
| Phase 2 - Single-feature interventions | Not started |
| Phase 3 - Define conformal harm | Not started |
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
```

The first experiment currently verifies data generation, split isolation, and persistence of exact split indices and their seed. The reusable conformal/scaling core is ready for the Phase 1 full-feature baseline.

Phase 0 validation details and tolerances are recorded in [`paper/phase-0-validation.md`](paper/phase-0-validation.md).

## Planned experiment sequence

1. `01_synthetic_baseline.py`: full-feature Base/TS/ConfTS with APS/RAPS.
2. `02_single_feature_ablation.py`: retrain once per removed feature.
3. `03_masking_sensitivity.py`: fixed-model masking/permutation diagnostic.
4. `04_harm_ranking.py`: constrained and Pareto rankings using tuning data.
5. `05_progressive_selection.py`: freeze a subset, recalibrate, evaluate once.
6. `06_real_data_benchmarks.py`: paired-seed comparison against standard baselines.
