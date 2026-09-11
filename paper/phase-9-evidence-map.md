# Phase 9 evidence map and upload request

Status: final artifact identified but not downloaded. Source run: https://github.com/samantr/conformal-harmful-features/actions/runs/34251770903 . Download `phase8-final-precision-extension` and upload a smaller ZIP containing the following final CSV/JSON files only (no checkpoints/models/datasets). Preserve names and values.

## Exact required files

```text
phase8_protocol.json
phase8_precision_extension_aggregate.json
phase8_precision_decision_initial10.json
phase8_precision_decision.json
phase8_all_results.csv
phase8_all_subject_results.csv
phase8_paired_size_effects.csv
phase8_paired_accuracy_effects.csv
phase8_matched_standard_effects.csv
phase8_accuracy_loss_sensitivity_effects.csv
phase8_subset_size_sensitivity_effects.csv
phase8_rank_stability_pairs.csv
phase8_rank_stability_summary.csv
phase8_har_subject_effects.csv
phase8_numerical_diagnostics.csv
phase8_numerical_diagnostics.json
```

These filenames are emitted by the committed aggregate/finalization scripts and listed by the final workflow. File contents and schemas remain unverified.

## Tables and figures

| ID | Exact proposed content | Frozen inputs |
|---|---|---|
| T1 | Dataset/model design; seeds, units, row counts, split regime, final precision status; scientific/control provenance footnote | protocol, precision-extension aggregate, both precision decision JSONs; frozen configs |
| T2 | Base APS alpha .10: all/proposed absolute size, accuracy, coverage, ECE, SSCV and class gap where recorded; one-shot/recursive paired size and accuracy effects, 95% intervals, exact p, Holm p, paired effect size; all six dataset/model strata | all_results; paired_size_effects; paired_accuracy_effects |
| T3 | Proposed versus each matched MI/permutation/RFE/SHAP/CRFE comparator, explicit size match and paired n; separate panel for top-1/3/5 across-seed Jaccard/Kuncheva and standard full-rank Spearman | matched_standard_effects; all_results; rank_stability_summary; rank_stability_pairs |
| T4 | Base/TS/ConfTS absolute sizes and coverage; component and joint gains, residual gains and additive excess; both paths and models, no best-cell selection | all_results; numerical_diagnostics CSV/JSON; any derived contrasts clearly labeled |
| T5 | Primary conditional reliability and HAR paired subject size/accuracy/coverage effects with two-way bootstrap intervals; numerical boundary frequencies | all_results; all_subject_results; har_subject_effects; numerical_diagnostics |
| F1 | Faceted paired-size-effect forest plot with adjacent accuracy and coverage panels, all six dataset/model strata and both paths | paired_size_effects; paired_accuracy_effects; all_results |
| F2 | Separate within-seed cross-method overlap and across-seed identity-stability panels; no fabricated partial-ranking correlations | rank_stability_pairs/summary for across-seed panel; per-unit files below for cross-method panel |
| F3 | Frozen removal-count and accuracy-allowance curves with size, accuracy and coverage panels; alpha/RAPS sensitivity faceted separately in supplement | all_results; accuracy_loss_sensitivity_effects; subset_size_sensitivity_effects; per-unit frozen choices if needed to map allowances |
| F4 | Scaling interaction plot with size/coverage panels and numerical flags, all primary selected subsets; APS in main text and full RAPS grid supplement | all_results; numerical_diagnostics |
| F5 | HAR subject-weighted effect intervals and subject coverage distribution, nominal .90/.95 references as applicable; windows never treated as independent subjects | all_subject_results; har_subject_effects |

Supplement: full grid metric tables, full matched comparator/effect-size outputs, sign-flip/Wilcoxon comparison, complete sensitivity and rank tables, precision decisions, saturation counts and provenance. No table pools datasets or score settings as independent replicates. Main text uses Base APS alpha .10 as primary; alpha .05 and RAPS are sensitivity results.

## Additional sources needed for direct H1 verification

The aggregate rank tables measure **within-method stability across seeds**, not necessarily **cross-method disagreement within a seed**. The runner reads these exact per-unit sources:

```text
<dataset>/seed_<seed>/baseline_selections.csv
<dataset>/seed_<seed>/proposed_selection/progressive_consensus_paths.csv
```

Datasets/seeds: `dry_bean` 43–62, `covertype` 43–52, `human_activity_recognition` 43–62. Upload these small CSVs with their directory structure if absent from the smaller final bundle. They permit path/selector overlap; a classification-importance versus conformal-harm score scatterplot additionally requires saved numerical tuning evidence and must not be promised from these aggregate summaries alone. Do not rerun selection to fill a missing evidence source.

## Read-only audit acceptance

- Check final metadata against user-reported 50/75,800/146,500 counts and 20/10/20 seeds, code `e4b3645`, recovery and extension audit statuses.
- Inspect schemas before defining joins; retain subset identifiers and grid coordinates, distinguish repeated random removals, and validate one reference per paired contrast.
- Use primary-selection markers plus Base APS alpha .10; never select a row by test performance.
- Reconcile paired n, signs, intervals and correction families; preserve initial and final precision decisions separately.
- Retain every numerical flag and every coverage failure. Report missing cells and summaries rather than silently dropping them.
- Hash input files once available and record which rows feed every output. Preserve Phase 8 files byte-for-byte.
