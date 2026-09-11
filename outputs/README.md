# Generated outputs

Store machine-generated tables, figures, and logs here. Every output row must record dataset, model, seed, split ID, feature subset, temperature, threshold, score, alpha, RAPS parameters, and code version.

Phase 2 writes the following files below the configured experiment directory:

- `single_feature_ablation.csv`: full retrain-ablation metrics, matched references, and deltas.
- `ablation_summary.csv`: per-feature descriptive pattern across both model families.
- `ablation_accuracy_vs_aps_size.png`: Base APS accuracy/efficiency scatterplot.
- `masking_sensitivity.csv`: fixed-model mean-mask and permutation results.
- `masking_summary.csv`: per-feature and intervention descriptive pattern.
- `masking_accuracy_vs_aps_size.png`: fixed-model Base APS scatterplots.
- `ablation_protocol.json` and `masking_protocol.json`: executable protocol checks.

Generated outputs are intentionally ignored by Git. The validation records under `paper/` preserve the phase-level evidence and conclusions.

Phase 3 additionally writes:

- `harm_tuning_evidence.csv`: fold-level tuning-only reference and ablation evidence.
- `harm_resample_evidence.csv`: cross-fitted evidence aggregated within each selection seed.
- `harm_rankings.csv`: constrained, weighted, and Pareto results per feature/pipeline.
- `harm_resample_rankings.csv` and `harm_rank_stability.csv`: rank reproducibility evidence.
- `harm_formulation_agreement.csv`: pairwise agreement among the three definitions.
- `harm_consensus.csv`: cross-model/scaling/score descriptive consensus.
- `harm_formulations.png`: Base APS accuracy-efficiency view with constrained candidates.
- `harm_protocol.json`: leakage, row-count, reference-pairing, and stability checks.

Phase 4 additionally writes:

- `progressive_pipeline_paths.csv`: tuning-only progressive paths for the preregistered selection pipeline.
- `progressive_consensus_paths.csv`: one row per subset step with the frozen-step flag.
- `progressive_final_results.csv`: all Base/TS/ConfTS x APS/RAPS results after fresh outer calibration.
- `progressive_pareto_curves.png`: accuracy-size and conditional-violation-size paths.
- `progressive_protocol.json`: partition-use, subset-freeze, row-count, and numerical checks.

Phase 5 additionally writes:

- `baseline_selections.csv`: frozen subsets, ranking sources, seeds, and feature scores.
- `baseline_final_results.csv`: one final Base APS result for every frozen subset.
- `baseline_summary.csv`: repeated-random aggregates and deterministic-method results.
- `baseline_protocol.json`: method presence, matched-size, partition-use, and numerical checks.
- `proposed_selection/`: auditable tuning-only Phase 4 rerun used to freeze the proposed subsets.

Phase 6 additionally writes:

- `scaling_interaction_results.csv`: all deterministic frozen subsets under Base/TS/ConfTS x APS/RAPS.
- `scaling_interaction_decomposition.csv`: feature, scaling, expected-additive, observed-joint, and interaction size gains.
- `scaling_rank_stability.csv`: matched-size selector-rank correlations and best-method changes between scalings.
- `scaling_interaction_summary.csv`: descriptive interaction counts and aggregate gains by subset type and pipeline.
- `scaling_interaction_proposed.png`: the primary all-features versus proposed-selection interaction plot.
- `scaling_interaction_protocol.json`: executable factorial, leakage, numerical, and decomposition checks.

Phase 7 real-data benchmarks additionally write:

- `dataset_provenance.json`: pinned source checksum, declared schema, feature manifest, and split identifiers.
- `phase7_split_indices.npz` and `phase7_split_distribution.csv`: the full outer split and class audit.
- `phase7_selection_indices.npz`: the exact train/tune rows available to selection under any declared compute budget.
- `phase7_main_table.csv`: deterministic final Base/TS/ConfTS x APS/RAPS results.
- `phase7_protocol.json`: source, split, selection-budget, leakage, coverage, and numerical-safety checks.

Phase 8 robustness runs additionally write:

- `phase8_run_plan.csv` and `phase8_plan_manifest.json`: the complete zero-fit 30-unit plan and frozen grid digest.
- Per-unit `phase8_split_indices.npz`, `phase8_selection_indices.npz`, and `phase8_accuracy_loss_choices.csv`: exact paired partitions and tuning-only sensitivity choices.
- Per-unit `checkpoints/`: atomic candidate and unique-subset shards bound to split, selection, configuration, grid, and code manifests.
- Per-unit `phase8_results.csv` and `phase8_subject_results.csv`: final conformal-grid results and HAR held-out-subject metrics.
- `phase8_paired_*_effects.csv` and `phase8_matched_standard_effects.csv`: paired intervals, effect sizes, tests, and Holm corrections.
- `phase8_accuracy_loss_sensitivity_effects.csv` and `phase8_subset_size_sensitivity_effects.csv`: frozen tolerance and 1--5-removal analyses.
- `phase8_rank_stability_*.csv`: all seed-pair rank correlations and top-k stability summaries.
- `phase8_har_subject_effects.csv`: two-way seed/subject bootstrap intervals.
- `phase8_precision_decision.json`: the preregistered 10-to-20-seed precision decision.
