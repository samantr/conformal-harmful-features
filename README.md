# Conformal Harmful Feature Detection

Research code and manuscript evidence for testing whether features with little classification value can disproportionately harm the efficiency or conditional reliability of APS/RAPS prediction sets.

## Project status

Phases 0–8 are complete and scientifically frozen. Phase 9 verified the final result archives, generated the paper tables/figures, drafted the general manuscript, and incorporated the first instructor-review presentation edits. Phase 10 is repository cleanup, final audit, and merge preparation; it must not change the frozen scientific results.

The frozen scientific core is commit `e4b36459f5030eb49326d129c9cf8b0c761e8d77`. Final evidence comprises 50 dataset/seed units, 75,800 aggregate result rows, and 146,500 HAR subject-level rows. See [`paper/phase-9-verification.md`](paper/phase-9-verification.md) for the verification boundaries and qualified findings.

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
| Phase 7 - Real datasets | ✅ Complete - Dry Bean + Covertype + subject-disjoint HAR |
| Phase 8 - Robustness and statistics | ✅ Complete - final precision extension verified |
| Phase 9 - Paper preparation and evidence verification | ✅ Complete for general instructor-review manuscript |
| Phase 10 - Repository cleanup/final audit | 🔎 In progress on `phase-10-final-audit` |

A target-journal-specific submission package is intentionally deferred until a journal is selected.

## Scientific rule

The four partitions have distinct roles:

- `train`: fit model and preprocessing;
- `tune`: feature ranking, subset size, TS/ConfTS and hyperparameters;
- `calibration`: compute a fresh final conformal threshold after all choices are frozen;
- `test`: final evaluation only.

No feature decision may use calibration or test results.

## Repository layout

```text
paper/          roadmap, validation records, verified evidence, manuscript and figures
reproduction/   frozen scripts from the completed base-paper reproduction
configs/        experiment settings
src/chf/        reusable scientific package
experiments/    numbered executable studies and frozen recovery/aggregation utilities
tests/          mathematical, split-integrity and regression tests
outputs/        generated local experiment artifacts (ignored except documentation)
```

The committed `paper/phase9_outputs/` package is intentional manuscript evidence and is separate from the ignored runtime `outputs/` tree.

## Installation and tests

```bash
python -m pip install -e ".[dev,baselines]"
pytest
```

The project requires Python 3.10 or newer. The GitHub CI job also validates the frozen Phase 8 plan without launching model fitting.

## Main experiment entry points

```bash
python experiments/01_synthetic_baseline.py --config configs/synthetic_debug.yaml
python experiments/02_single_feature_ablation.py --config configs/synthetic_debug.yaml
python experiments/03_masking_sensitivity.py --config configs/synthetic_debug.yaml
python experiments/04_harm_ranking.py --config configs/synthetic_debug.yaml
python experiments/05_progressive_selection.py --config configs/synthetic_debug.yaml
python experiments/06_required_baselines.py --config configs/synthetic_debug.yaml
python experiments/07_scaling_interaction.py --config configs/synthetic_debug.yaml
python experiments/08_real_dry_bean.py --config configs/dry_bean.yaml
python experiments/09_real_covertype.py --config configs/covertype.yaml
python experiments/10_har_provenance.py --config configs/human_activity_recognition.yaml
python experiments/11_real_har.py --config configs/human_activity_recognition.yaml
python experiments/12_robustness_statistics.py --config configs/phase8_robustness.yaml --plan-only
```

The later `13`–`19` scripts document frozen Phase 8 recovery, audit, numerical-diagnostic and precision-extension work; they are retained for provenance rather than presented as new experimental phases.

## Evidence and manuscript records

Phase-specific validation is recorded in `paper/phase-0-validation.md` through `paper/phase-7-validation.md`. The Phase 8 design is in [`paper/phase-8-protocol.md`](paper/phase-8-protocol.md). Phase 9 evidence is documented in:

- [`paper/phase-9-plan.md`](paper/phase-9-plan.md)
- [`paper/phase-9-claims-ledger.md`](paper/phase-9-claims-ledger.md)
- [`paper/phase-9-evidence-map.md`](paper/phase-9-evidence-map.md)
- [`paper/phase-9-verification.md`](paper/phase-9-verification.md)
- [`paper/phase9_outputs/`](paper/phase9_outputs/)
- [`paper/manuscript/`](paper/manuscript/)

Phase 10 records the final repository/merge audit in [`paper/phase-10-final-audit.md`](paper/phase-10-final-audit.md).
