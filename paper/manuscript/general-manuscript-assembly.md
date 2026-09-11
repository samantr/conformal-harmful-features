# General manuscript assembly

**Working title:** *Conformal-Harmful Features: Feature Selection for Efficient Adaptive Prediction Sets*  
**Assembly status:** General journal-style draft, assembled 2026-09-11.  
**Scientific status:** Editorial assembly only. Phases 0–8, scientific code, experimental outputs, hypothesis decisions, tables and figures remain frozen.

## Front matter

- Author: Saman Tarighpeima
- Affiliation(s): to be finalized before submission
- Corresponding author: to be finalized before submission
- Keywords: conformal prediction; feature selection; adaptive prediction sets; uncertainty quantification; prediction-set efficiency; temperature scaling; robustness

The general assembly uses the finalized manuscript components in this order:

1. `abstract.md`
2. `introduction.md`
3. `related-work.md`
4. `method.md`
5. `experimental-protocol.md`
6. `results.md`
7. `discussion.md`
8. `conclusion.md`
9. Declarations placeholders
10. `references.bib`
11. Main tables and figure captions

## Main-display numbering in the assembled draft

The manuscript uses ordinary journal-style numbering rather than the Phase 9 production labels:

| Manuscript display | Frozen source |
|---|---|
| Table 1 | `tables/table_t1_design_audit_main.tex` |
| Table 2 | `tables/table_t2_primary_compact.tex` |
| Table 3 | `tables/table_t3_h1_compact.tex` |
| Table 4 | `tables/table_t3_matched_baselines_compact.tex` |
| Table 5 | `tables/table_t4_scaling_compact.tex` |
| Table 6 | `tables/table_t5_har_subject_compact.tex` |
| Table 7 | `tables/table_t5b_numerical_diagnostics_main.tex` |
| Figure 1 | `../phase9_outputs/figures/figure_f1_primary_effects.pdf` |
| Figure 2 | `../phase9_outputs/figures/figure_f2_overlap_stability.pdf` |
| Figure 3 | `../phase9_outputs/figures/figure_f3_sensitivity_small_nn.pdf` |
| Figure 4 | `../phase9_outputs/figures/figure_f4_scaling_interaction.pdf` |
| Figure 5 | `../phase9_outputs/figures/figure_f5_har_subject_effects.pdf` |

The logistic-regression sensitivity display remains supplementary as `../phase9_outputs/figures/figure_s3_sensitivity_lr.pdf`, and full matched-baseline effects remain Supplementary Table T3a in the frozen Phase 9 outputs.

## Declarations placeholders

Until a target journal and final author list are known, the assembled draft retains placeholders for Funding, Competing interests, Author contributions, Data and code availability, and Acknowledgements. The final data/code statement should cite the selected repository/archive and preserve the scientific-code provenance `e4b3645`.

## Claim boundaries preserved during assembly

- H1 remains supported with qualification: selected removals are distinct, but exact identities are not universally stable.
- H2 remains partially supported overall, strongest for Dry Bean small NN; HAR efficiency gains remain qualified by subnominal coverage.
- H3 remains partially supported descriptively as sub-additive complementarity, not general synergy.
- H4 remains not generally supported.
- HAR subject-disjoint splitting is described as leakage control, not proof of exchangeability or nominal coverage.
- Numerical-boundary flags remain visible and are not treated as independent failure probabilities or a causal explanation for all HAR undercoverage.

The assembled general draft does not add experiments, inference, seeds, tuning, subset selection or scientific claims. Journal-specific styling, declarations, final figure placement and submission metadata can be applied after a target journal is selected.
