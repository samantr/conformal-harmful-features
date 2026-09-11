# Phase 8 precision extension

## Why this extension is being run

The completed 10-seed Phase 8 aggregate triggered the **pre-registered precision-escalation rule** for two datasets and not for the third. The initial decision was computed from `phase8_precision_decision.json` produced by recovery-v2 run `34239393188`:

- `dry_bean`: 2 of 4 co-primary contrasts exceeded the precision target -> extend to 20 seeds.
- `human_activity_recognition`: 2 of 4 co-primary contrasts exceeded the precision target -> extend to 20 seeds.
- `covertype`: 0 of 4 contrasts exceeded the precision target -> stop at 10 seeds.

The frozen rule is: extend when at least two co-primary contrasts have a 95% CI half-width above `max(0.01, 1% of the all-feature mean prediction-set size)`, with a maximum of 20 paired seeds.

This extension is therefore **not significance chasing** and does not change any scientific choice made after observing test outcomes. It executes the escalation already specified in `configs/phase8_robustness.yaml`.

## Frozen scientific design

The extension preserves the Phase 8 scientific code and grid from commit:

`e4b36459f5030eb49326d129c9cf8b0c761e8d77` (`e4b3645`)

The following remain unchanged:

- train/tune/fresh calibration/test partitioning;
- subject-disjoint splitting for HAR;
- Logistic Regression and the small neural network;
- alpha values 0.10 and 0.05;
- Base, TS, and ConfTS;
- APS and the full RAPS lambda/k_reg grid;
- allowed accuracy-loss sensitivity values;
- subset-size sensitivity values;
- paired seeds within each dataset comparison;
- exact two-sided sign-flip inference, Wilcoxon sensitivity checks, effect sizes, Holm correction, rank stability, and HAR two-way seed/subject bootstrap;
- the post-run numerical-boundary policy: retain and flag observed exact 0/1 probability events without calibration/test-driven retuning, replacement, or deletion.

## Extension seeds

The original 10 seeds are 43-52. The extension batch uses the next 10 deterministic seeds:

`53, 54, 55, 56, 57, 58, 59, 60, 61, 62`

Only `dry_bean` and `human_activity_recognition` are run for the extension batch. Covertype remains frozen at its completed 10-seed result because the pre-registered rule did not trigger there.

The extension seed file intentionally contains exactly 10 seeds so that it is accepted by the original Phase 8 scientific validator without modifying the scientific code. Final inference then combines the initial and extension batches, yielding 20 paired seeds for Dry Bean and HAR and 10 paired seeds for Covertype.

## Numerical boundary observations

The completed 10-seed benchmark established that exact probability boundaries on HAR are a real held-out robustness observation, especially under ConfTS. The extension must therefore use the same transparent retention rule introduced in `PHASE8_POSTRUN_AMENDMENT.md`:

1. retain every prespecified grid cell and its originally tuned temperature;
2. keep raw exact-zero/exact-one counts unchanged;
3. flag raw and scaling-induced boundary events;
4. keep structural/provenance validation strict;
5. never retune a temperature using final calibration/test outcomes;
6. never replace a flagged scaled cell with Base and never drop it.

A unit marked complete means the computation and provenance checks completed; it does not mean every scaling cell is numerically safe.

## Final stopping rule

After the extension, Dry Bean and HAR are at the pre-registered maximum of 20 seeds. No additional seeds will be added to rescue significance, coverage, or numerical behavior. The final report will show whether the precision target was met at 20 seeds, but the experiment stops at 20 either way.
