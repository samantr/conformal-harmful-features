# Phase 9 tables and figures

Generated from frozen Phase 8 outputs by `paper/generate_phase9_tables_figures.py`. The source hashes and reporting boundaries are recorded in `manifest.json`. CSV files are the canonical tabular sources; TeX files are generated renderings. Each figure is available as a vector PDF and a 220-dpi PNG.

## Main tables

- T1 (`table_t1_design_audit`): dataset/seed population, row counts, split regime, precision status and scientific provenance.
- T2 (`table_t2_primary_results`): Base APS at alpha 0.10, including absolute metrics, paired size/accuracy effects and descriptive coverage differences.
- T3a (`table_t3a_matched_baselines`): proposed paths versus every frozen size-matched standard selector.
- T3b (`table_t3b_rank_stability`): across-seed identity-stability summaries, kept separate from within-seed cross-method overlap.
- T4 (`table_t4_scaling_interaction`): APS Base/TS/ConfTS absolute metrics and supplied H3 component contrasts.
- T5a (`table_t5a_har_subject_effects`): exactly reproduced HAR two-way seed/subject bootstrap summaries.
- T5b (`table_t5b_numerical_diagnostics`): numerical boundary, induced-boundary, coverage and empty-set diagnostics.

## Main figures

- F1 (`figure_f1_primary_effects`): primary set-size and accuracy forest plots with descriptive coverage differences. Coverage has no newly computed interval.
- F2 (`figure_f2_overlap_stability`): within-seed removed-set overlap beside across-seed top-1 stability. The common color scale makes their distinction explicit.
- F3 (`figure_f3_sensitivity_small_nn`): complete frozen accuracy-allowance and removal-count curves for the small NN. The LR counterpart is `figure_s3_sensitivity_lr` in the supplement.
- F4 (`figure_f4_scaling_interaction`): mean set size and coverage for Base APS, TS and ConfTS across all dataset/model strata. The red dotted line is nominal 0.90 coverage.
- F5 (`figure_f5_har_subject_effects`): frozen subject-weighted bootstrap effects and the distribution of the 100 primary seed/subject cells per model/path. Windows and grid rows are not independent subjects.

## Claim boundaries retained

- H1 remains supported with qualification: selected removals are distinct, but exact identities are unstable.
- H2 remains partially supported overall and strongest for Dry Bean small NN. HAR efficiency must be read beside undercoverage and incomplete matched-baseline superiority.
- H3 remains descriptively sub-additive complementarity, not general synergy. Its supplied interval generator remains unavailable.
- H4 remains not generally supported.
- HAR subject-disjoint undercoverage and numerical saturation stay visible. No figure implies that nominal coverage was maintained where it was not.

No model fitting, new seed, recalibration, retuning, feature reselection, grid modification or new hypothesis test is performed by the generator.
