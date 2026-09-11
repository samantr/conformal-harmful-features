# Phase 10 — Repository Cleanup, Final Audit, and Merge Preparation

**Date:** 2026-09-12  
**Source branch:** `phase-9-paper-preparation` at `831d064709f92a8e8f955584b8b7c57aa226f2af`  
**Cleanup branch:** `phase-10-final-audit`  
**Target branch:** `master`  
**Merge candidate:** draft PR #1

Phase 10 is repository maintenance and merge preparation only. It does not authorize model fitting, new seeds, tuning, recalibration, subset reselection, grid expansion, result replacement, or new scientific hypothesis tests.

## 1. Frozen scientific state

The scientific core remains frozen at commit `e4b36459f5030eb49326d129c9cf8b0c761e8d77`. A repository comparison from that commit to the Phase 9 head shows no changes under `src/chf/` after the freeze. Later additions are recovery/aggregation utilities, precision-extension configuration/workflows/tests, and manuscript/evidence files. This preserves the frozen core implementation used by the recorded Phase 8 results.

The final verified evidence remains unchanged:

- 50 dataset/seed units;
- Dry Bean: 20 paired seeds (43–62);
- Covertype: 10 paired seeds (43–52);
- HAR: 20 paired seeds (43–62);
- 75,800 aggregate result rows;
- 146,500 HAR subject-level rows;
- scientific version `e4b3645` recorded throughout the aggregate evidence.

The Phase 9 audit reconciles the primary effects, overlap summaries, H3 mean algebra, numerical diagnostics and HAR grouped uncertainty within the documented verification boundaries. All eight frozen HAR subject size/coverage bootstrap summaries reproduce exactly. See `phase-9-verification.md` for the evidence boundaries and remaining limitations.

## 2. Merge-lineage audit

At Phase 10 start, `master` was at `87f61e11d5b6619a110eafe3561801d351ed0ccf` and `phase-9-paper-preparation` was 73 commits ahead and 0 commits behind. The merge base was exactly the current `master` head. Therefore the research branch forms a linear continuation of `master` at the audit point; there is no divergent master-side history to reconcile before merge.

`phase-9-paper-preparation` is also a linear continuation of `phase-8-robustness` (35 commits ahead, 0 behind), so Phase 9 contains the frozen Phase 8 history rather than an independent fork.

## 3. Repository-hygiene changes

Phase 10 applies only non-scientific cleanup:

- remove tracked `.idea/` / PyCharm project metadata;
- retain `.idea/` and `*.iml` ignore rules and extend ignores for editor/OS state, local archives, build products, logs and manuscript build output;
- keep runtime `outputs/*/` ignored;
- keep `paper/phase9_outputs/` committed intentionally because it is curated manuscript evidence, not a runtime result cache;
- refresh the root README so the project status no longer says that Phase 8 is unrun;
- retain Phase 8 recovery scripts, workflows, amendments and audit utilities as provenance rather than deleting historical evidence.

The inspected Phase 9 tree contained no tracked virtual environment, Python cache directory, local ZIP/RAR bundle, or generated runtime output directory requiring removal. The only identified tracked local-environment clutter was `.idea/`, which Phase 10 removes.

## 4. Documentation consistency

The root README is now the current project-status entry point. Historical Phase 8 planning files remain historical records and are not rewritten to pretend that the benchmark had already run when they were authored. The current state is instead documented by the Phase 9 verification and this Phase 10 audit.

The general instructor-review manuscript is retained under `paper/manuscript/`. The first instructor feedback round changed presentation only: shorter one-sentence captions, panel lettering, explanation after displays, no bold emphasis in running prose, metrics in the conclusion, and a continuous Discussion. No scientific result or frozen source figure was altered by that editorial round.

A target-journal-specific submission-ready package remains intentionally deferred until a journal is selected.

## 5. Files intentionally retained

The following may look historical or generated but are intentionally kept:

- `paper/base-paper.pdf`: provenance/reference for the reproduced foundation;
- `reproduction/`: frozen learning/reproduction scripts;
- `PHASE8_*.md`: recovery/precision chronology;
- `experiments/13_*` through `19_*`: frozen recovery, audit, numerical-diagnostic and precision-extension utilities;
- `.github/workflows/phase8-*`: benchmark/recovery/report provenance and reproducible manual entry points;
- `paper/phase9_audit/`: compact verification artifacts;
- `paper/phase9_outputs/`: curated tables/figures and manifest used by the manuscript.

Deleting these would make the repository cosmetically smaller but materially weaken the audit trail.

## 6. Scientific claims at merge time

The merge must preserve the Phase 9 claim boundaries:

- **H1:** supported with qualification for selection distinctness, not complete-rank divergence or stable feature identity;
- **H2:** partially supported overall, strongest for Dry Bean small NN; HAR efficiency evidence is limited by undercoverage and incomplete matched-comparator superiority;
- **H3:** partially supported descriptively as sub-additive complementarity, strongest for HAR small-NN one-shot; no general synergy claim;
- **H4:** not generally supported; subject-coverage intervals cross zero and conditional metrics are mixed.

No merge-time cleanup may revise evidence to make these conclusions stronger.

## 7. Final validation before merge

Required before merging `phase-10-final-audit` into `master`:

- [x] confirm branch lineage is ahead-only relative to `master`;
- [x] confirm no post-freeze `src/chf/` drift after scientific commit `e4b3645`;
- [x] confirm final evidence counts and scientific version from the Phase 9 audit;
- [x] remove tracked IDE metadata and harden ignore rules;
- [x] refresh stale root project status;
- [x] review the Phase 10-only diff for accidental scientific changes;
- [ ] run the repository test suite on the merge candidate;
- [ ] run the frozen Phase 8 `--plan-only` validation on the merge candidate;
- [ ] merge only after the checks above pass.

The Phase 10-only comparison against `phase-9-paper-preparation` contains only `.gitignore`, root `README.md`, removal of tracked `.idea/` files, and this audit document. It contains no `src/`, `experiments/`, `configs/`, `tests/`, frozen evidence, manuscript-result, table, or figure modification.

Draft PR #1 targets `master` from `phase-10-final-audit` and GitHub currently reports it as mergeable. No workflow run was created for the Phase 10 head. The existing Phase 8 CI workflow is configured for pull requests, but because that workflow is introduced by the research branch rather than the current `master`, it did not provide a fresh PR-head validation here. The latest inspected Phase 8 CI on `phase-8-robustness` (run `34252010591`, head `13556b98`) completed successfully, but that historical pass is not treated as a substitute for validating the present merge candidate.

The recommended merge is a squash merge only after the two remaining executable validations pass. Historical Phase 7/8/9 branches should not be deleted as part of the merge itself; branch pruning can be considered separately after the merged master has been verified.

## 8. Audit limitations

Phase 10 does not repeat the Phase 8 experiments. It relies on the already completed Phase 9 read-only evidence audit for result integrity. As previously documented, the original split-index files and every individual unit manifest are not available, the H3 interval-generation script was not independently reproduced, and per-window prediction sets are unavailable for reconstruction of pooled `size_p90`. These are manuscript limitations, not repository-merge blockers.

The remaining merge blocker is a fresh execution of the test suite and frozen Phase 8 `--plan-only` validation on the Phase 10 merge candidate. Once those pass, the repository is ready for merge preparation completion; the actual merge remains a separate explicit action.
