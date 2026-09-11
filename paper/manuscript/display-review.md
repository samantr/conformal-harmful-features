# Phase 9 display review

## Scope and outcome

All T1–T5 tables and F1–F5 figures at remote commit `5046da5` were reviewed against the Phase 9 claims ledger, verification report, evidence map, canonical CSVs, and the deterministic figure-generation code. The scientific interpretation does not change: H1 remains supported with qualification; H2 remains partially supported overall; H3 remains partially supported descriptively as sub-additive complementarity; H4 remains not generally supported. No numerical result, experimental row, model, temperature, subset, seed, calibration threshold, or hypothesis test was changed.

The review found no verified numerical sign or interval error in the generated evidence. It did identify presentation issues: the full T2, T3a/T3b, and T4 tables are too wide for a main manuscript; H1 within-seed overlap and across-seed stability require a clearer separation; H3 interval provenance needs to be explicit; numerical-boundary frequencies need a denominator warning; and F4 itself displays size and coverage but not numerical flags, so the earlier evidence-map wording that implied flags were plotted in F4 is superseded. Numerical flags remain visible through Table T5b and Results §5.5.

## Main-paper versus supplement decisions

| Evidence | Placement | Decision |
|---|---|---|
| T1 design/audit | Main | Keep as a small provenance table. Add a caption warning that “precision met” is not efficacy or coverage. |
| T2 primary results | Main compact; full table supplement | Main version retains absolute size/accuracy/coverage plus paired size and accuracy effects. Full ECE/SSCV/class-gap/effect-size columns remain in the generated full table. |
| T3 H1 overlap/stability | Main compact; full stability table supplement | Main version separates within-seed cross-method removed-set overlap from across-seed top-1 identity stability. No complete-rank statistic is assigned to partial progressive paths. |
| T3a matched baselines | Main compact summary; full comparator table supplement | Main version reports the number/names of five matched selectors beaten on set size and the full range of point effects. Full CIs, effect sizes, exact/Wilcoxon p values and corrections stay in T3a. |
| T4 scaling interaction | Main compact; full table supplement | Main version retains feature-only, scaling-only, combined, interaction, selected size and coverage. H3 intervals are explicitly labeled as supplied Phase 8 evidence. |
| T5a HAR subject effects | Main compact; full table supplement | Main version pairs subject-weighted size and coverage effects and preserves clustered denominators. |
| T5b numerical diagnostics | Main supporting limitation table | Keep visible because HAR saturation materially qualifies H3/HAR efficiency claims. Pooled coverage is diagnostic, not a single nominal-coverage estimate. |
| F1 primary effects | Main | Keep. Size/accuracy intervals are paired t intervals; coverage panel is descriptive and has no newly computed CI. |
| F2 overlap/stability | Main | Keep. Caption must distinguish the two estimands and conditional overlap denominator. |
| F3 small-NN sensitivities | Supplement | Secondary frozen-grid robustness; full grid retained without test-selected operating points. |
| S3 LR sensitivities | Supplement | Same rationale as F3. |
| F4 scaling interaction | Main | Keep size and coverage panels with nominal 0.90 line. Numerical flags are not plotted in F4 and are referenced to T5b. |
| F5 HAR subject effects | Main | Keep. Bootstrap is clustered by seed and subject; subject/grid/window rows are not independent subjects. |

## Publication-quality captions

**T1 — Audit and evaluation population.** Frozen Phase 8 result population by dataset. Aggregate rows are experiment-result rows under scientific code `e4b3645`; HAR subject rows are subject-condition records used for clustered subject analysis, not independent subjects. `Final precision = Met` denotes satisfaction of the frozen precision stopping rule only and does not establish efficacy, statistical significance, numerical safety, or nominal conformal coverage. HAR splitting is subject-disjoint.

**T2 — Primary Base-APS effects.** Primary results at alpha = 0.10 with tuning accuracy-loss allowance 0.01. Set-size effect is all-feature mean set size minus selected mean set size, so positive values favor removal. Accuracy effect is selected minus all-feature accuracy, so positive values favor removal. Effect intervals are supplied paired Student-t 95% CIs and p values are Holm-adjusted exact sign-flip tests. Coverage is the absolute descriptive mean and has no newly constructed interval; nonsignificant accuracy differences are not noninferiority evidence.

**T3-H1 — Selection distinctness and identity stability.** Within-seed Jaccard compares proposed removed sets with five size-matched conventional selectors at the same seed; zero-removal cases are excluded only from this overlap estimand and the conditional n is reported. Across-seed top-1 Jaccard summarizes pairwise seed agreement of the proposed progressive path; the 45/190 seed pairs overlap and are not independent replications. These are different estimands, and partial proposed paths are not assigned fabricated complete-rank correlations.

**T3-B — Size-matched comparator summary.** Proposed-versus-standard Base-APS set-size effects under matching by removed-feature count within seed. Positive effects mean the proposed subset yields a smaller mean prediction set. `Wins` counts the five prespecified selectors (MI, permutation importance, RFE, SHAP, CRFE) for which the supplied Holm-adjusted exact sign-flip p value is below 0.05 and the effect favors the proposed method; it is a descriptive summary of the five corrected contrasts, not a new hypothesis test. Full effect sizes, intervals and p values are retained in Supplementary Table T3a.

**T4 — APS feature-removal × scaling interaction.** Frozen alpha = 0.10 APS contrasts. Feature gain is all-feature minus selected set size within ConfTS; scaling gain is all-Base minus all-ConfTS; combined gain is all-Base minus selected-ConfTS. Positive values indicate smaller sets. Additive excess is B + C - A - D for all-Base A, selected-Base B, all-ConfTS C and selected-ConfTS D; negative values indicate sub-additivity. H3 means were independently reconciled, but the original interval-generation script is unavailable, so displayed H3 intervals remain supplied Phase 8 evidence. Absolute coverage is shown beside size and must be interpreted with the HAR undercoverage and numerical diagnostics in T5b.

**T5 — HAR subject-disjoint effects.** Subject-weighted proposed-versus-all-feature effects under the frozen primary filter. Positive size reduction favors removal; positive coverage difference means higher selected-subset coverage. Intervals are the exactly reproduced 95% percentile intervals from the frozen 10,000-repetition two-way seed/subject bootstrap. Each contrast contains 100 observed seed × held-out-subject cells from 20 seeds and 26 unique subject identifiers; windows and the 146,500 subject-condition grid rows are not independent subjects. Coverage intervals crossing zero do not establish equivalence.

**T5b — Numerical-boundary diagnostics.** Frozen diagnostic counts over the full dependent grid. Raw and scaling-induced flag frequencies are grid-row counts and must not be interpreted as independent model-, seed-, subject-, or window-level failure probabilities. The pooled mean coverage column spans multiple frozen operating points and is not a single nominal-coverage estimate. No flagged rows were excluded.

**F1 — Primary efficiency, accuracy and coverage.** Paired primary effects for Base APS at alpha = 0.10. Positive set-size reduction and positive accuracy difference favor feature removal. Horizontal intervals in the size and accuracy panels are paired Student-t 95% CIs; the coverage panel shows only the descriptive mean selected-minus-all difference and intentionally has no new interval. Statistical decisions use the separately reported Holm-adjusted exact sign-flip p values.

**F2 — Cross-method distinctness versus cross-seed identity stability.** Left: mean within-seed removed-set Jaccard between each proposed path and size-matched MI, permutation importance, RFE, SHAP and CRFE, conditional on at least one removal. Right: across-seed top-1 Jaccard for the proposed progressive paths. The shared scale is for visual comparison only; the panels answer different questions and pairwise seed comparisons are dependent.

**F3 — Frozen feature-removal sensitivities (supplement).** Small-NN effects across every prespecified tuning accuracy-loss allowance and fixed removal-count cell, with set-size, accuracy-loss and coverage-difference panels. Lines summarize dependent grid cells and are not independent replications; no operating point is selected from test outcomes. The corresponding LR display is Supplementary Figure S3.

**F4 — Interaction with probability scaling.** Mean APS set size and empirical coverage at alpha = 0.10 for all features, one-shot removal and recursive removal under Base, ordinary temperature scaling and ConfTS. The dotted reference is nominal 0.90 coverage. Smaller sets are not interpreted as improvements when accompanied by undercoverage. Numerical-boundary flags are not encoded in this figure; they are reported in T5b.

**F5 — Reliability under subject shift.** HAR subject-disjoint subject-weighted size and coverage effects with exactly reproduced two-way seed/subject bootstrap intervals, plus descriptive coverage distributions for the 100 primary seed × subject cells per proposed contrast. The dotted line marks nominal 0.90 coverage. The display does not treat windows, subject-condition grid rows, or overlapping subject appearances across seeds as independent subjects.

## Corrections and validation

The compact manuscript tables are deterministic reductions of the generated canonical tables; their builder performs no inference and asserts expected row/comparator counts. Signs were standardized so that positive set-size values favor removal and positive accuracy/coverage differences favor the selected subset. The full generated tables remain unchanged as the complete supplement source. All primary and subject-effect intervals, Holm p values and H3 supplied intervals were checked against the verified evidence ledger. Figure-generating code was checked against the canonical tables: F1 uses paired-t intervals only for size/accuracy; F4 draws the 0.90 nominal line on coverage; F5 draws the same nominal reference and uses frozen subject bootstrap effects. No figure data were changed.

## Remaining limitations

The original H3 interval-generation script is unavailable, so those intervals remain supplied evidence even though their means are independently reconciled. Original split-index files and individual unit completion manifests are unavailable. Per-window prediction sets are unavailable, preventing reconstruction of pooled `size_p90` from subject quantiles. HAR subject-disjoint splitting prevents direct subject leakage but does not itself establish window exchangeability or nominal coverage. Numerical saturation co-occurs with HAR undercoverage but is not demonstrated to be its sole cause. Finally, grid cells, matched windows, rank-pair comparisons and repeated subject appearances are dependent and must not be interpreted as additional independent replicates.
