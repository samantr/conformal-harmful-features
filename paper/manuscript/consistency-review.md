# Phase 9 manuscript consistency review

**Scope:** Sections 1--7, bibliography, frozen claims ledger, and verified Results/Discussion evidence.  
**Scientific status:** wording and manuscript-structure review only; no Phase 8 data, inference, subsets, grids, temperatures, or hypothesis decisions were changed.

## Review outcome

The manuscript is internally consistent with the frozen H1--H4 interpretation:

- **H1:** supported with qualification for selection distinctness; no intrinsic or seed-invariant harmful-feature claim.
- **H2:** partially supported overall; strongest evidence is Dry Bean small neural network; HAR efficiency gains remain explicitly limited by subnominal coverage.
- **H3:** partially supported descriptively as sub-additive complementarity, not general synergy; supplied H3 intervals remain identified as not independently regenerated.
- **H4:** not generally supported; subject-weighted efficiency is separated from coverage/reliability and from any claim of grouped conformal validity.

Section order is now complete as Introduction, Background and Related Work, Method, Experimental Protocol, Results, Discussion, and Conclusion. The Abstract is maintained separately and should be treated as a summary of these locked sections rather than a source of new claims.

## Consistency corrections made

1. Normalized Method citations to the same Pandoc/BibTeX citation-key style used in the Introduction and Related Work for TS, ConfTS, APS, and RAPS.
2. Replaced the ambiguous Introduction phrase “two error levels” with “two alpha levels” so the robustness grid is described consistently with the Experimental Protocol.
3. Preserved “Base APS at alpha = 0.10” as the primary operating point throughout.
4. Preserved the statistical unit as the paired **seed** for primary inference; dependent grid rows, random-subset repetitions, and overlapping seed-pair stability summaries are not described as additional independent replications.
5. Preserved HAR subject-disjoint splitting as a leakage-control design rather than a guarantee of exchangeability or nominal coverage. The subject bootstrap remains an effect-robustness analysis over the observed seed/subject structure.
6. Preserved numerical-boundary events as retained diagnostics. No manuscript section attributes all HAR undercoverage causally to ConfTS or numerical saturation.
7. Preserved the distinction between retraining-based feature removal and inference-time masking, and between feature-selection effects and probability-scaling effects.
8. Preserved the literature-positioning boundary: conformal feature selection, recursive conformal elimination, conformal-efficiency optimization, and tune/calibrate separation all have prior art. The contribution is the specific constrained retrained-ablation formulation and its robustness analysis, not a “first” claim.

## Cross-section numerical checks

The central values repeated outside Results are consistent with the verified evidence: Dry Bean small-NN primary size reductions are approximately 0.017 for both paths with positive accuracy changes; HAR small-NN primary size reductions are approximately 0.018 and 0.013 with observed coverage below 0.90; the HAR one-shot residual ConfTS feature benefit is approximately 0.008 with negative additive excess; and all four subject-level HAR coverage-effect intervals cross zero. No new inferential result was introduced in the Conclusion or Abstract.

## Remaining manuscript-level caveats

The original Phase 8 H3 interval-generation script remains unavailable, so those intervals must continue to be described as supplied frozen evidence. Original split-index files and individual unit completion manifests were not available to the Phase 9 audit. Per-window prediction sets were also unavailable for independent reconstruction of pooled quantities such as `size_p90`. These evidence limitations remain documented in the Discussion and should not be silently removed during later formatting or journal adaptation.
