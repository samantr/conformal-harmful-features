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
