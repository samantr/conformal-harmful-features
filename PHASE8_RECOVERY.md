# Phase 8 Recovery Addendum

This addendum records an operational recovery of the preregistered Phase 8 robustness run. It does **not** change the scientific grid, seeds, splits, feature-selection constraints, models, alpha values, RAPS parameters, or final calibration/test isolation.

## Original run

- Workflow run: `34101494390`
- Scientific commit: `e4b36459f5030eb49326d129c9cf8b0c761e8d77` (`e4b3645`)
- Completed without recovery: all 10 Dry Bean units and 8/10 Covertype units.
- Incomplete: Covertype seeds 43 and 50; all 10 HAR units.

Every matrix job uploaded its output directory even on failure/cancellation, so expensive intermediate artifacts are available for validated resume.

## Observed operational failures

1. **HAR subject-level filtering under pandas 3**: the legacy subject dataframe stored APS `raps_lambda`/`raps_k_reg` as floating NaN values, while the allowed-grid dataframe could infer object dtype from `None`. Pandas 3 rejects the float/object merge. The recovery controller normalizes only these two merge keys in memory before that exact subject-filter merge.
2. **Over-strict exact-boundary validation**: Covertype seeds 43 and 50 reached the end of the scientific computation but failed because isolated exact-1 probabilities were already present in Base (`T=1`). Inspection of the uploaded final-grid shards showed no additional exact-zero/exact-one counts induced by TS or ConfTS. Recovery therefore retains the original boundary counts in the result CSV and validates whether TS/ConfTS increases exact-boundary counts above the matching Base cell.
3. **HAR wall-time**: several HAR jobs reached the 330-minute job timeout after completing the expensive progressive-selection stage. Recovery restores the uploaded artifacts and resumes only unfinished work.

## Provenance policy

Recovery is intentionally controlled from a separate checkout. The scientific checkout is pinned to the original clean commit `e4b3645`, so existing checkpoint manifests remain code-version compatible. The recovery controller verifies the frozen seed, split ID, selection-data ID, and PASS selection protocols before reuse. If a completed `baseline_selections.csv` is present it is reused; otherwise a completed, PASS progressive-selection CSV is reused and only the deterministic standard-baseline ranking stage is rebuilt.

Each recovered unit writes `phase8_recovery_manifest.json`, and the recovered aggregate writes `phase8_recovery_aggregate.json`. These records identify the source workflow run, scientific commit, recovery-control commit, pandas version, reuse path, and operational repairs.

## Scientific interpretation

The recovered outputs must be treated as continuation/post-processing of the original Phase 8 run, not as a new robustness design. The intended paired 10-seed inference is produced only after all 30 original dataset/seed units have a PASS completion record.
