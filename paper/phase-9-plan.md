# Phase 9 manuscript plan

Working title: *Conformal-Harmful Features: Feature Selection for Efficient Adaptive Prediction Sets*

Status (2026-09-11, updated after complete HAR verification): primary result rows, key summary arithmetic and all 146,500 HAR subject rows are verified. The eight frozen subject size/coverage bootstrap summaries reproduce exactly. The historical access-gap and provisional claim text below records the initial plan; the current claims in phase-9-claims-ledger.md and phase-9-verification.md supersede those provisional entries. Planning/support files only; no major manuscript edits.

## Evidence and freeze

Inspected branch head: `13556b98be8aa4873643bca0acd7fd1b8589a206` on `phase-8-robustness`. Scientific code: `e4b36459f5030eb49326d129c9cf8b0c761e8d77`. Read the Phase 8 protocol, recovery and post-run amendment, precision-extension documentation and runner/aggregate code. The original protocol's “benchmark not started” status is historical; preserve it and explain chronology here rather than rewriting the frozen record.

The documented final design is Dry Bean 20 seeds (43–62), Covertype 10 (43–52), HAR 20 (43–62). The user reports 50 audited units, 75,800 results and 146,500 subject rows; these counts still require verification against the final files. Successful workflow run `34251770903` contains artifact `10116096484`, `phase8-final-precision-extension`, 771,864,217 bytes. The connected download tool rejects artifacts above 536,870,912 bytes. This is an access limitation, not evidence of experimental failure. The later run `34252010574` is cancelled and must not replace the successful final artifact.

Phases 0–8, configurations, scientific code and all experimental outputs remain frozen. No experiments, seeds, temperatures, subsets or result values may be changed. No merge to master. Phase 7 observations are descriptive anchors, not extra Phase 8 replicates.

## Verified claim boundaries

| Hypothesis | Final interpretation for planning |
|---|---|
| H1 | Supported with qualification: direct removed-set disagreement is verified, but exact feature identities are unstable and complete-rank divergence is not established by these overlaps. |
| H2 | Partially supported overall: strongest combined evidence is Dry Bean small NN. HAR size gains do not establish coverage-preserving superiority; LR/Covertype utility is not established. |
| H3 | Partially supported descriptively: sub-additive complementarity especially HAR small-NN one-shot, with weaker recursive residual evidence and explicit saturation/coverage limits. |
| H4 | Not generally supported: mixed conditional metrics and subject-coverage intervals crossing zero; no equivalence claim. |

See phase-9-verification.md for numbers and audit boundaries. No general synergy, universal selector superiority or blanket nominal-coverage claim.

## Proposed paper structure

1. Introduction: classification utility may diverge from adaptive-set efficiency; state bounded contributions and all four hypotheses.
2. Background and related work: APS/RAPS, TS/ConfTS, ordinary and conformal feature selection. Compare the actual CRFE implementation and budget; avoid unsupported first-ever novelty claims. Verify literature separately before drafting.
3. Method: conformal-harm definition, tuning-only constrained selection, one-shot and recursive paths, stopping and zero-removal fallback.
4. Experimental protocol: train/tune/calibration/test separation, paired randomization, subject-disjoint HAR, baselines, frozen grid, initial precision rule and extension, exact inference and correction families, recovery/amendment chronology.
5. Results, ordered below.
6. Discussion, limitations and practical implications.
7. Conclusion: qualified evidence for efficiency-oriented selection, with reliability and numerical boundaries explicit.
8. Supplement: full grid and comparator tables, precision status, rank diagnostics, numerical diagnostics, provenance and reporting derivations.

## Results sequence

5.1 Audit and evaluation population (T1): final counts, seed allocation, split regime and precision status. Completion means provenance-valid computation, not nominal coverage or numerical safety.

5.2 Classification importance versus conformal harm (T3/F2): within-seed agreement and between-seed stability, reported separately.

5.3 Primary efficiency, accuracy and coverage (T2/F1): Base APS, alpha 0.10, tuning accuracy-loss allowance 0.01; both paths and both models on all three datasets. Report absolute metrics and paired effects. Do not treat an insignificant accuracy delta as a noninferiority test.

5.4 Matched baselines and frozen sensitivities (T3/F3): all prespecified deterministic comparators, matched random reference, alpha/RAPS grid, accuracy allowances and removal counts. No test-selected best operating point. The 60-cell grid applies only to the documented subset families; do not imply a fully crossed design for random or diagnostic subsets.

5.5 Interaction with scaling (T4/F4): paired Base/TS/ConfTS patterns; distinguish TS calibration from conformal set efficiency, residual feature benefit and sub-additivity. Show coverage and saturation next to size.

5.6 Reliability under subject shift (T5/F5): SSCV/class gaps and HAR subject-level uncertainty, nominal reference lines and observed deficits. Include negative findings in the main text.

## Discussion design

- Model dependence: compare actual effects and intervals, not significance labels across models. Neural sensitivity is an observation; a representational mechanism is a hypothesis unless measured. Weak LR/Covertype findings limit generality.
- Feature identity: correlated/redundant features may permit different effective subsets, but instability does not establish this explanation. Distinguish stable aggregate gains from stable feature attribution.
- HAR subject shift: subject-disjoint splitting prevents subject leakage but does not itself establish window exchangeability or nominal coverage. Separate subject-weighted bootstrap estimands from window-weighted marginal metrics. Do not present 146,500 rows as independent subjects.
- Numerical saturation: preserve raw and scaling-induced exact boundary flags and empty-set/coverage observations. Co-occurrence does not establish that saturation caused all undercoverage. Never remove flagged cells or retune from held-out data. TS may enlarge sets; ConfTS reductions require coverage and numerical qualifications.
- Statistical scope: 20/10/20 seeds, three datasets and two models; report unmet precision targets at the maximum seed budget. Seed intervals quantify repeat-run variability conditional on these datasets, not universal population guarantees. Grid cells and overlapping seed pairs are dependent.
- Baseline fairness and selection scope: disclose implemented approximations, feature-removal cap, tuning budgets and zero-removal fallback. Do not imply inference-time masking and retrained ablation are the same method.
- Practical implication: recommend reporting accuracy, efficiency, coverage and numerical diagnostics together; avoid deployment recommendations based solely on smaller sets.

## Frozen-output-only additions proposed, not executed

1. Descriptive paired coverage deltas and per-seed size/accuracy/coverage panels, preserving all primary comparisons.
2. Relative size reduction per seed alongside absolute reduction; specify mean of paired percentages, not an interchangeable ratio of means.
3. Scaling interaction algebra on matched frozen subsets: let A/B/C/D be sizes for all-Base, selected-Base, all-ConfTS, selected-ConfTS. Feature-only gain A−B; scaling-only A−C; combined A−D; excess over additivity B+C−A−D. Negative excess is sub-additive, but complementarity also requires residual benefits C−D and B−D. Retain numerical flags. Treat new inferential analyses as post hoc and define a complete family before calculation; do not reuse primary significance claims.
4. Cross-method top-k removal overlap from saved rankings/paths, within each seed. Full-rank Spearman is inappropriate for partial recursive paths. For unequal/short paths report available lengths and a predeclared overlap convention; never fabricate a full ranking.
5. Descriptive coverage-deficit and numerical-flag frequencies over the full fixed grid with explicit denominators. Flag-stratified displays are diagnostics, not a filtered replacement efficacy analysis.

No new model fitting, prediction generation, calibration, threshold estimation, temperature tuning, subset selection or experimental grid expansion. Derived displays must have separate Phase 9 paths, source hashes, deterministic transformations and no writes to Phase 8 outputs.

## Step-by-step next work

1. Obtain the exact final aggregate files listed in `phase-9-evidence-map.md`.
2. Read-only audit counts, keys, seed pairing, provenance, missingness, primary selections and flag retention; compare stored summaries with frozen row arithmetic without launching the runner.
3. Populate `phase-9-claims-ledger.md` with numerical evidence and corrected hypothesis labels. Resolve missing H1 source data explicitly.
4. Review this plan and paper structure before major manuscript edits.
5. Produce tables/figures from frozen outputs, then draft Results and Discussion, then other sections and abstract. Keep unsupported and adverse findings visible.

## Update after frozen-output verification

Main evidence gap resolved by user uploads. Actual rows confirm 75,800 results, 50 dataset/seed units, 20/10/20 seeds and e4b3645 throughout. All final precision targets are met. The separate HAR subject CSV is verified at 146,500 rows, and its eight frozen two-way-bootstrap summaries reproduce exactly.

Final labels: H1 supported with qualification for selection distinctness; H2 partially supported overall, strongest on Dry Bean small NN, with HAR efficiency evidence limited by undercoverage and incomplete matched-comparator superiority; H3 partially supported descriptively as sub-additive complementarity, especially HAR small-NN one-shot; H4 not generally supported. See the populated ledger for boundaries.

Steps 1–3 are complete within the audit scope described in phase-9-verification.md, including the complete HAR subject-row verification. No experiments have been run. The next step is to generate the planned tables and figures from the audited compact sources, review them and the existing paper structure, and only then begin major manuscript drafting. H1 can use selected_indices in the main CSV; H3 RAPS summaries must be labeled as within-seed averages over nine fixed cells and not plotted as individual settings.
