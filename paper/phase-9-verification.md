# Phase 9 frozen-results verification

Updated 2026-09-11. This is manuscript-support analysis, not manuscript prose or new experimentation.

## Audit scope

The archive contains one 991,392,656-byte CSV with 75,800 rows. Actual rows confirm 50 dataset/seed units, Dry Bean/HAR seeds 43–62 and Covertype 43–52, scientific version e4b3645 throughout, one recorded split_id/selection_data_id per unit, frozen subset flags, and no recorded calibration/test-driven selection. Original split-index files and individual completion manifests were not supplied.

Final JSON records 50 audited units, 12 recovered initial units and 20 extension units. Every dataset meets the final precision criterion, with zero contrasts above its half-width threshold. Precision is not efficacy or coverage.

The read-only `audit_phase9_outputs.py` reconciles 24 primary size/accuracy mean effects, all 60 H1 overlap summaries, all 108 H3 mean-contrast summaries, rank-summary means, numerical flags and four subject-coverage effect means. Primary/H3 arithmetic differences are below 1e-15. Source hashes and detailed checks are in `phase9_audit/audit.json`. No experimental modules are imported, and no models, temperatures, calibration thresholds, subsets or seeds are generated. Stored p-values and confidence intervals are reported as supplied; their entire statistical implementation was not independently repeated.

The follow-up `verify_phase9_har.py` audit reads the 4,424,509,870-byte subject CSV in 5,000-row chunks and retains only necessary columns. It independently verifies all 146,500 subject rows against all 29,300 HAR aggregate cells, then executes the exact frozen `_primary_proposed`, `_paired_subject_bootstrap_table` and `two_way_cluster_bootstrap` functions loaded from scientific commit `e4b3645`. All 66 checks pass. Detailed hashes, pairing keys, subject lists and maximum errors are in `phase9_audit/har_verification.json`; the reproduced table is `phase9_audit/har_subject_effects_reproduced.csv`.

## Primary numerical evidence

Base APS at alpha .10, tuning accuracy-loss allowance .01. Positive size reduction is all-feature minus selected mean size. CI is the supplied paired Student-t interval; p is the supplied Holm-adjusted exact sign-flip p-value.

| Dataset | Model | Path | Size reduction | 95% CI | Holm p |
|---|---|---|---:|---|---:|
| Dry Bean | Small NN | One-shot | .016968 | [.007914, .026021] | .00000763 |
| Dry Bean | Small NN | Recursive | .017761 | [.008777, .026745] | .00000763 |
| HAR | Small NN | One-shot | .017774 | [.009855, .025693] | .000717 |
| HAR | Small NN | Recursive | .013224 | [.004643, .021805] | .004959 |
| Dry Bean | LR | One-shot | −.000992 | [−.002455, .000471] | .184364 |
| Dry Bean | LR | Recursive | −.001432 | [−.002856, −.000008] | .106918 |
| HAR | LR | One-shot | .000824 | [−.001997, .003645] | .699783 |
| HAR | LR | Recursive | .001418 | [−.001704, .004541] | .699783 |
| Covertype | LR | One-shot | .000377 | [−.001443, .002197] | .666016 |
| Covertype | LR | Recursive | .001269 | [−.000578, .003117] | .312500 |
| Covertype | Small NN | One-shot | .005704 | [−.004183, .015592] | .457031 |
| Covertype | Small NN | Recursive | .004272 | [−.006037, .014581] | .457031 |

An unadjusted t interval excluding zero need not agree with a corrected exact test. Do not modify outputs to remove this difference.

Dry Bean small-NN accuracy gains are .003283 and .003680 (0.3283 and 0.3680 percentage points), with Holm p=.001648 and .000519. HAR small-NN accuracy deltas are positive in mean but intervals cross zero. Nonsignificance is not a noninferiority result.

Primary coverage means across seeds:

| Dataset/model | All features | One-shot | Recursive |
|---|---:|---:|---:|
| Dry Bean small NN | .899074 | .899361 | .898744 |
| Dry Bean LR | .901432 | .901388 | .901498 |
| Covertype small NN | .900102 | .900379 | .900312 |
| Covertype LR | .899567 | .899594 | .899547 |
| HAR small NN | .888214 | .888247 | .892221 |
| HAR LR | .888246 | .889145 | .887972 |

Dry Bean is close to nominal in aggregate, not literally at least .90 for every condition/seed. HAR undercoverage remains explicit. Subject-disjoint splitting prevents leakage but does not establish window exchangeability across subjects. Absolute metrics including accuracy, ECE, SSCV and class gap are in `phase9_audit/primary_absolute.csv`.

## H1 — supported with qualification

All 60 supplied overlap summaries reproduce from selected_indices at the same removed-feature count. No-removal cases are excluded from this descriptive overlap calculation: Dry Bean uses n=19 per model/path and HAR small-NN one-shot uses n=19. Primary inference retains all 20 seeds. Report the conditional denominator; do not describe this as missing primary data.

HAR small-NN one-shot has zero removed-set Jaccard against all five standards. Recursive has zero except permutation importance (.011111). Dry Bean small-NN Jaccard ranges .005848–.048956 for one-shot and .017544–.101086 for recursive. This verifies distinct selected removals, not full-rank correlation divergence. Low overlap in a high-dimensional universe alone does not establish usefulness.

Across-seed small-NN top-1 Jaccard is .047368 for Dry Bean and zero for HAR/Covertype. Other model/dataset combinations differ. Stable aggregate gains do not mean stable feature identity; a correlated-feature explanation is a hypothesis, not a demonstrated mechanism. Partial recursive paths must not receive fabricated full-rank correlations.

## H2 — partially supported overall

Both Dry Bean small-NN paths beat every matched MI, permutation, RFE, SHAP and CRFE comparator on size after the supplied Holm correction; largest adjusted p=.000763. This is the strongest utility evidence, alongside improved accuracy and near-nominal observed coverage.

HAR small-NN one-shot beats four of five on size; RFE has Holm p=.070171. Recursive beats only SHAP at .05 (p=.015541); the other four have p=.070171. HAR therefore cannot be claimed superior to every standard, and undercoverage prevents support for the original coverage-qualified H2. All LR and Covertype primary size contrasts fail to establish improvement after correction. Different significance labels across models are not a formal interaction test.

## H3 — partial, descriptive sub-additive complementarity

Let A/B/C/D be matched sizes for all-Base, selected-Base, all-ConfTS and selected-ConfTS. Additive excess is B+C−A−D. APS supplied means reproduce at alpha .10. RAPS means reproduce after averaging the nine fixed lambda/k settings within each seed and alpha; those settings are not independent replications.

HAR small-NN one-shot: feature-only gain .017774; scaling-only .015289; combined .023335; additive excess −.009728, supplied CI [−.015758, −.003699]. Residual feature gain under ConfTS is .008045, supplied CI [.002255, .013836]. The combined mean gain also exceeds feature-only gain. Recursive residual gain .003819 has supplied CI [−.002195, .009833], so evidence is weaker.

Dry Bean APS ConfTS stays at T=1 for these primary subsets and adds no benefit. No general synergy claim is supported. H3 interval-generation source code was not supplied: the mean algebra is independently checked, but the intervals remain supplied descriptive post-hoc summaries. HAR saturation and undercoverage prohibit a claim of safe scaling-based complementarity.

## H4 and HAR uncertainty

All four subject-coverage intervals cross zero: LR one-shot .000988 [−.001609, .003545], LR recursive −.000176 [−.002894, .002636], NN one-shot −.000012 [−.005227, .004836], NN recursive .003886 [−.003692, .010944]. Their means reproduce from embedded subject coverage dictionaries. SSCV/class-gap changes are mixed. The supplied NN subject-size intervals favor removal, but this does not imply reliability improvement.

The separate subject CSV is now verified: 146,500 unique result keys, 20 seeds (43–62), five held-out subjects in every aggregate condition and 26 unique held-out subject IDs across seeds. There are no missing expected records, duplicate pairing keys, absent all-feature references or missing required values. Null RAPS parameters occur only as structural APS nulls. Every subject row matches an aggregate row on the frozen condition/subset key and recorded experiment, split ID, selection-data ID, code version, selected indices, primary marker and selection seed. Recorded flags confirm classifier refitting, frozen subsets and no final-calibration/test-driven selection. Subject IDs are integers in the expected HAR domain and window counts are positive integers, constant within each seed/subject.

All subject accuracy, coverage and mean-size values reconcile to the aggregate results after weighting by `window_count`; their stored all-feature references and paired differences also reconcile. Subject coverage additionally agrees with all 146,500 embedded `group_coverages` entries. Aggregate `size_p90` is not reconstructible by averaging subject quantiles, so only its stored references and within-subject differences were checked.

All eight supplied size/coverage summaries reproduce using the frozen primary filter, 10,000-repetition two-way seed/subject bootstrap, seed 810001 and 95% percentile interval. The largest absolute reproduced-table difference is below 1e-16. Each proposed model/path contrast has 100 observed seed/subject cells, 20 seeds and 26 unique subjects. The table contains no subject-accuracy interval; subject accuracy is verified descriptively without adding new inference.

Input identity is recorded at both archive and member level. The subject RAR SHA-256 is `bbe2b687fb29c1d9b285a70498519506107a93d406672ca3f700e97599bd600a`; its CSV member is 4,424,509,870 bytes with SHA-256 `32b5daa57173a6cbd1dce12d3b4c5c3682b82e10d8ddac55666a4147a317d964`. The independently supplied aggregate RAR member matches the aggregate CSV previously audited from ZIP byte-for-byte.

## Numerical diagnostics and sensitivities

There are 10,006 raw-boundary flagged rows and 5,115 induced-boundary flagged rows. All induced flags are HAR ConfTS: 5,115/8,700 cells (58.79%). This is a dependent-grid row frequency, not an independent subject/model failure probability. Base has 4,343 HAR flagged rows and three Covertype flagged rows; do not attribute all undercoverage to ConfTS.

HAR small-NN all-feature APS size is 1.021782 at Base, 1.058356 at TS and 1.006493 at ConfTS. ConfTS coverage is .887779. One-shot ConfTS size is .998447 with coverage .887573; size below one can coexist with empty sets and is not automatic success. Pooled diagnostics over alpha .05/.10 do not establish nominal coverage at either operating point.

Supplied allowance-sensitivity summaries retain positive, Holm-significant small-NN size gains for Dry Bean/HAR at all four allowances. Covertype does not. Fixed removal-count results are not uniformly favorable or significant; do not select a better removal count from test outcomes or claim monotonic improvement.

## Paper decisions and remaining limitations

Retain the planned structure: Introduction; Background/related work; Method; Experimental protocol; Results; Discussion; Conclusion, with full-grid supplement. Results must distinguish direct overlap from stability, all-feature gains from matched-baseline gains, and efficiency from reliability. Show actual coverage and flags alongside scaling interaction. Main text retains adverse and null results.

The HAR subject evidence gap is closed; no claim decision changes. The remaining source limitation is the unavailable H3 interval-generation script: H3 mean algebra is verified, but its supplied intervals are not independently reproduced. Original split-index files and all individual unit manifests are also unavailable, so the audit verifies recorded provenance and the exact expected subject membership rather than reconstructing the complete split history. Per-window predictions and sets are unavailable, preventing reconstruction of pooled `size_p90` from subject quantiles. These limitations do not block the qualified interpretation or the planned tables/figures. No new experiment is needed.
