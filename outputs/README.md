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
