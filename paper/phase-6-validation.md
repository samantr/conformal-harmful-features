# Phase 6 Validation Record

**Status:** Complete

**Completed:** 2026-09-03

**Scope:** Evaluate all frozen deterministic Phase 5 feature subsets under the
full Base/TS/ConfTS x APS/RAPS factorial and determine whether feature
intervention and scaling gains are additive, synergistic, or redundant.

## Factorial protocol

Phase 6 consumes `baseline_selections.csv`, not the Phase 5 test-result table.
No comparator is chosen using final performance. The evaluated variants are:

- all features;
- mutual information, permutation importance, RFE, SHAP, and CRFE;
- proposed conformal-harm one-shot and recursive selection.

Random subsets are excluded because this phase studies deterministic
feature-selection/scaling interactions rather than the matched-size null
distribution already recorded in Phase 5. Across the two model families there
are 21 frozen selections and 126 final rows:

```text
21 selections x 3 scalings x 2 scores = 126 rows
```

Each unique feature subset is refit on outer train. Base, TS, and score-specific
ConfTS temperatures are fixed using outer tune only. ConfTS retains the base
paper's disjoint threshold/loss split within tune. Every frozen scaling-score
pipeline receives a fresh threshold from outer calibration and one evaluation
on outer test. Randomized APS/RAPS uniforms are paired across every comparison.

The configuration remains seed 42, alpha 0.10, RAPS `lambda=0.001`, and
`k_reg=1`. Multi-seed inference remains Phase 8.

## Interaction definition

For selected subset `F` and non-Base scaling `S`, define gains as reductions in
mean prediction-set size relative to the all-feature Base result:

```text
feature gain at Base       = size(all, Base) - size(F, Base)
scaling gain on all        = size(all, Base) - size(all, S)
expected additive gain     = feature gain at Base + scaling gain on all
observed joint gain        = size(all, Base) - size(F, S)
interaction size gain      = observed joint gain - expected additive gain
```

Equivalently, the interaction is the feature gain under `S` minus the feature
gain under Base. Positive values mean descriptive synergy; negative values mean
overlap or antagonism. Values within `+/-0.01` are labeled approximately
additive. These labels are effect descriptions, not significance claims.

## All-feature scaling results

| Model | Score | Base size | TS temperature / size | ConfTS temperature / size | ConfTS coverage |
|---|---|---:|---:|---:|---:|
| Logistic regression | APS | 1.8665 | 1.0 / 1.8665 | 0.4 / 1.6755 | 0.8950 |
| Logistic regression | RAPS | 1.8550 | 1.0 / 1.8550 | 0.4 / 1.6540 | 0.8930 |
| Small neural network | APS | 1.9410 | 1.1 / 1.9650 | 0.4 / 1.7210 | 0.8960 |
| Small neural network | RAPS | 1.9355 | 1.1 / 1.9560 | 0.4 / 1.7010 | 0.8930 |

Ordinary TS is identical to Base for logistic regression and enlarges the
neural-network sets by 0.0240 for APS and 0.0205 for RAPS. ConfTS reduces the
all-feature sizes by 0.1910-0.2345 while keeping empirical coverage within
0.007 of the 0.90 target. As expected from its efficiency objective, ConfTS
raises ECE substantially; it is not a confidence-calibration improvement.

## Proposed-selection interaction

| Model / path | Score | Base feature gain | All-feature ConfTS gain | Joint ConfTS gain | Interaction | Description |
|---|---|---:|---:|---:|---:|---|
| Logistic / one-shot | APS | -0.0055 | 0.1910 | 0.1900 | 0.0045 | Approximately additive |
| Logistic / one-shot | RAPS | -0.0035 | 0.2010 | 0.1970 | -0.0005 | Approximately additive |
| Logistic / recursive | APS | -0.0080 | 0.1910 | 0.2030 | **0.0200** | Synergistic |
| Logistic / recursive | RAPS | -0.0115 | 0.2010 | 0.2055 | **0.0160** | Synergistic |
| Neural / one-shot = recursive | APS | 0.0005 | 0.2200 | 0.2450 | **0.0245** | Synergistic |
| Neural / one-shot = recursive | RAPS | 0.0025 | 0.2345 | 0.2440 | 0.0070 | Approximately additive |

The logistic recursive subset is slightly worse than all features under Base,
but becomes better under ConfTS: its mean size is smaller than all-feature
ConfTS by 0.0120 for APS and 0.0045 for RAPS. The neural subset is smaller than
all-feature ConfTS by 0.0250 for APS and 0.0095 for RAPS, with unchanged test
accuracy. The neural one-shot and recursive paths selected the same 19-feature
subset, so they are one unique curve despite retaining separate method labels.

The proposed method is therefore complementary to ConfTS at some operating
points, supporting H3 descriptively. It is not uniformly complementary: the
logistic one-shot interactions and neural RAPS interaction are approximately
additive, and Phase 8 must establish stability across seeds.

## Standard-selector comparison and ranking changes

At 15 logistic features, CRFE remains the most efficient selector under every
scaling-score pipeline. At 16 logistic features, SHAP is best under Base/TS,
but CRFE becomes best under ConfTS. At 19 neural features, CRFE is best under
Base/TS, while the proposed subset becomes best under ConfTS for both APS and
RAPS.

This change is visible in the Base-to-ConfTS matched-size rank correlations:

| Model / size | APS correlation | RAPS correlation | Best method changed? |
|---|---:|---:|---|
| Logistic / 15 | 0.3143 | 0.9276 | No; CRFE remains best |
| Logistic / 16 | -0.6029 | -0.4638 | Yes; SHAP to CRFE |
| Neural / 19 | 0.5926 | 0.7778 | Yes; CRFE to proposed |

Thus scaling is not merely a constant offset applied to every feature subset;
it can reverse the relative ordering of selectors.

## Temperature, threshold, class ranking, and concentration

All proposed subsets choose the same grid temperatures as their all-feature
references: logistic Base/TS use 1.0, neural TS uses 1.1, and ConfTS uses 0.4.
The only deterministic selector that changes a selected temperature is neural
CRFE, whose TS temperature moves from 1.1 to 1.0. Proposed-subset threshold
deltas are small (`-0.00103` to `+0.00058`). Their mean maximum probabilities
decrease by only `0.00136-0.00208`, while mean entropy increases by
`0.00195-0.00754`.

Positive scalar temperature cannot change class ordering, and the executable
checks confirm exact rank and accuracy invariance across Base/TS/ConfTS for a
fixed fitted subset. Retraining after feature removal does change some model
rankings: top-1 disagreement with the full model is 2.45-2.50% for the logistic
subsets and 9.15% for the neural subset. Mean true-label rank changes remain
small (`-0.0045` to `+0.0015`).

## Validation and decision

The executable protocol passes every check: all three feature-set types and all
six pipelines are present; no random subset is evaluated; all choices are
frozen before calibration; thresholds and scientific metrics are finite;
empirical coverage remains within 0.03 of target; no zero or exactly-one
probabilities occur; positive scaling preserves accuracy and class order; and
all 76 difference-in-differences identities hold numerically.

**Phase 6 decision:** interaction with scaling is complete. The results provide
descriptive evidence that the proposed intervention and ConfTS can be
complementary, especially for neural APS and logistic recursive selection.
They also show that the proposed method is not generally superior: CRFE remains
best for 15-feature logistic models and becomes best at 16 features under
ConfTS. Phase 7 should now test the complete protocol on Dry Bean before adding
other real datasets; Phase 8 must determine whether these interactions survive
paired multi-seed analysis.
