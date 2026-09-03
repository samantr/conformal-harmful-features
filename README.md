# Conformal Harmful Feature Detection

Research code for testing whether features with little classification value can disproportionately harm the efficiency or conditional reliability of APS/RAPS prediction sets.

## Current milestone

**Phase 7 - Real datasets: IN PROGRESS.** The complete leakage-safe protocol has
been transferred to UCI Dry Bean, including verified source provenance,
train-only preprocessing, both model families, matched baselines, and the full
Base/TS/ConfTS x APS/RAPS factorial.

Dry Bean is complete at the single-seed descriptive level. Next: transfer the
same frozen protocol to **Covertype**; multi-seed inference remains Phase 8.

## Stage tracker

| Stage | Status |
|---|---|
| Phase 0 - Freeze reproduction foundation | ✅ Complete |
| Phase 1 - Controlled synthetic data | ✅ Complete |
| Phase 2 - Single-feature interventions | ✅ Complete |
| Phase 3 - Define conformal harm | ✅ Complete |
| Phase 4 - Progressive subset selection | ✅ Complete |
| Phase 5 - Required baselines | ✅ Complete |
| Phase 6 - Interaction with scaling | ✅ Complete |
| Phase 7 - Real datasets | 🚧 In progress - Dry Bean complete |
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
python -m pip install -e ".[dev,baselines]"
pytest
python experiments/01_synthetic_baseline.py --config configs/synthetic_debug.yaml
python experiments/02_single_feature_ablation.py --config configs/synthetic_debug.yaml
python experiments/03_masking_sensitivity.py --config configs/synthetic_debug.yaml
python experiments/04_harm_ranking.py --config configs/synthetic_debug.yaml
python experiments/05_progressive_selection.py --config configs/synthetic_debug.yaml
python experiments/06_required_baselines.py --config configs/synthetic_debug.yaml
python experiments/07_scaling_interaction.py --config configs/synthetic_debug.yaml
python experiments/08_real_dry_bean.py --config configs/dry_bean.yaml
```

The first experiment generates the controlled 10-class/20-feature dataset, verifies its known feature roles using training data only, persists the exact four-way split, trains both model families, and writes all 12 Base/TS/ConfTS x APS/RAPS result rows to `baseline_results.csv`.

Phase 0 validation details and tolerances are recorded in [`paper/phase-0-validation.md`](paper/phase-0-validation.md).
Phase 1 protocol, metrics, results, and stability checks are recorded in [`paper/phase-1-validation.md`](paper/phase-1-validation.md).
Phase 2 intervention protocols, exploratory results, and decision gate are recorded in [`paper/phase-2-validation.md`](paper/phase-2-validation.md).
Phase 3 definitions, tuning-only cross-fitting protocol, stability results, and primary-method decision are recorded in [`paper/phase-3-validation.md`](paper/phase-3-validation.md).
Phase 4 one-shot/recursive paths, frozen-subset results, and untouched-test findings are recorded in [`paper/phase-4-validation.md`](paper/phase-4-validation.md).
Phase 5 matched-size baseline protocols and results are recorded in [`paper/phase-5-validation.md`](paper/phase-5-validation.md).
Phase 6 scaling-factorial protocol, interaction decomposition, and results are recorded in [`paper/phase-6-validation.md`](paper/phase-6-validation.md).
Phase 7 Dry Bean provenance, protocol, accepted results, numerical reset, and decision are recorded in [`paper/phase-7-validation.md`](paper/phase-7-validation.md).

## Planned experiment sequence

1. `01_synthetic_baseline.py`: full-feature Base/TS/ConfTS with APS/RAPS.
2. `02_single_feature_ablation.py`: retrain once per removed feature.
3. `03_masking_sensitivity.py`: fixed-model masking/permutation diagnostic.
4. `04_harm_ranking.py`: constrained and Pareto rankings using tuning data.
5. `05_progressive_selection.py`: freeze a subset, recalibrate, evaluate once.
6. `06_required_baselines.py`: matched-size comparison against required baselines.
7. `07_scaling_interaction.py`: Base/TS/ConfTS x APS/RAPS interaction analysis.
8. `08_real_dry_bean.py`: first real-data transfer with provenance and protocol audits.
