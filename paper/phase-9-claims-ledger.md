# Phase 9 claims ledger

## Verified decisions after uploads

Input names and SHA-256 hashes: `phase9_audit/audit.json`. Filters: Base APS, alpha .10, primary selected markers, both paths and models; pairing by dataset/model/seed and matched subset size for standards. All Phase 8 files remain unchanged.

| Claim | Current decision | Evidence and boundary |
|---|---|---|
| Final counts/code | Verified within supplied scope | 75,800 rows; 50 units; exact 20/10/20 seeds; e4b3645 throughout. Audit JSON reports 50 audited units, 12 recovered and 20 extensions. Individual manifests not independently re-audited. |
| 146,500 HAR subject rows | Metadata and embedded coverage count verified | Separate subject CSV absent; do not claim its size rows were inspected. |
| H1 | Supported with qualification | All 60 within-seed removed-set overlap summaries reproduce from raw selected_indices. No-removal cases excluded only from overlap summaries. Across-seed identity stability is separately weak, particularly small NN. No complete-rank divergence or intrinsic harmful-feature claim. |
| H2 | Partially supported overall | Dry Bean small NN improves size and accuracy, and both paths beat all five matched selectors on size after Holm. HAR has size gains but undercoverage and does not beat every matched comparator. LR/Covertype do not establish broad utility. |
| H3 | Partially supported descriptively | HAR small-NN one-shot residual gain under ConfTS .008045; combined .023335; additive excess −.009728. Recursive residual interval crosses zero. No general synergy or safe-scaling claim. |
| H4 | Not generally supported | All four HAR subject-coverage intervals cross zero; SSCV/class-gap directions are mixed. Not evidence of equivalence. |
| TS/ConfTS | Observed pattern confirmed | HAR small-NN all-feature APS size: Base 1.021782, TS 1.058356, ConfTS 1.006493; ConfTS coverage .887779. |
| Numerical limitation | Confirmed | 10,006 raw-flag rows, 5,115 induced-flag rows, all induced flags in HAR ConfTS. No flagged rows removed. |
| Precision | Final target met | All datasets meet frozen final precision rule; this says nothing by itself about significance or coverage. |

Stored inference is reported as supplied, not independently reimplemented. H3 means are independently reconciled; interval-generation code is absent. Rank pairs/grid rows are dependent and are not additional independent replicates.
