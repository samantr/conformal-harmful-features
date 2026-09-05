# Phase 8 Robustness and Statistics Protocol

**Status:** Protocol approved and implemented; benchmark not started

**Frozen parent:** `phase-7c-har` at
`b6afb43f23e075f2380a9c81d910d09732425a8c`

**Execution branch:** `phase-8-robustness`

This document freezes the Phase 8 design before any Phase 8 classifier is fit.
The Phase 7A--7C results are immutable descriptive anchors. They are not pooled
with or relabeled as Phase 8 replicates.

## Questions and estimands

The primary operating point is Base APS at `alpha=0.10`, with an allowed
tuning accuracy loss of 0.01. For each dataset, classifier family, seed, and
proposed path, the primary paired estimands are:

1. all-feature mean prediction-set size minus proposed-subset mean size;
2. proposed-subset accuracy minus all-feature accuracy;
3. proposed-subset coverage minus all-feature coverage; and
4. the analogous conditional-reliability deltas.

Positive size and accuracy effects therefore favor the proposed subset. The
one-shot and recursive paths are separate, prespecified comparisons. Logistic
regression and the small neural network are both retained; a neural-only result
cannot establish model-family robustness.

At the exact proposed subset size in each seed, the proposal is also compared
with every deterministic standard selector: mutual information, permutation
importance, RFE, SHAP, and CRFE. This is safer than choosing an oracle
"strongest" method from test performance. A claim of superiority to the
strongest matched standard requires superiority to all prespecified standard
comparators after multiplicity correction. Ten matched random subsets remain
a descriptive repeated baseline at each primary proposed size.

## Paired experimental units

The initial seeds are exactly `43, 44, ..., 52`. Each of the three datasets has
all ten seeds, producing 30 dataset/seed units. Within a unit, every method,
feature subset, alpha, scaling, and conformal score uses:

- the identical outer split;
- the identical selection train/tune indices;
- the identical classifier seed for a given model family;
- the identical randomized APS/RAPS calibration uniforms;
- the identical randomized APS/RAPS test uniforms; and
- the identical internal ConfTS split.

Dry Bean and Covertype retain stratified row-disjoint splits and their frozen
Phase 7 selection budgets. HAR retains the 15/5/5/5 group allocation and the
group-split search. A subject can occur in exactly one of train, tune,
calibration, or test in any seed.

## Frozen sensitivity grids

The full conformal grid has 60 cells per primary subset:

| Dimension | Values |
|---|---|
| Alpha | 0.10, 0.05 |
| Scaling | Base, TS, ConfTS |
| APS | One unregularized setting |
| RAPS lambda | 0.001, 0.01, 0.1 |
| RAPS `k_reg` | 1, 3, 5 |

Thus each alpha/scaling combination contains one APS cell and nine RAPS cells.
Full-grid evaluation applies to all features, the tuning-selected proposed
subsets, and deterministic standard subsets at the primary matched sizes.
Other deterministic 1--5-removal subsets receive Base APS at both alphas.
Random subsets receive only primary Base APS at alpha 0.10.

Progressive paths continue diagnostically through exactly 1, 2, 3, 4, and 5
removed features even if the primary stopping rule fires. These diagnostic
steps do not replace the primary selected subset. The allowed tuning accuracy
loss is varied over `0, 0.005, 0.01, 0.02`. For each allowance, subset choice is
recomputed from cross-fitted outer-tune evidence only, subject to the frozen
coverage-shortfall constraint and nonnegative tuning efficiency gain. Step zero
is retained as the safe choice when no removal qualifies.

## Inference

Each reported method contrast uses seed-paired differences. Reports include
the mean and median difference, standard deviation and standard error, a 95%
Student-t confidence interval, Cohen's paired `d_z`, matched rank-biserial
correlation, the exact two-sided sign-flip test, and the Wilcoxon signed-rank
sensitivity test. P-values are Holm-adjusted within dataset/model/metric
families. Exact p-values, intervals, and effect sizes are reported together;
no conclusion depends on an isolated threshold crossing.

Rank robustness is computed over all 45 seed pairs per dataset/model/method.
Complete standard-selector harmfulness rankings report Spearman correlation,
top-1/3/5 Jaccard overlap, and Kuncheva stability. Proposed one-shot and
recursive removal paths report top-1/3/5 Jaccard and Kuncheva stability;
Spearman is deliberately omitted for a five-step partial recursive ranking.

HAR additionally reports window-level metrics by held-out subject. Paired
subject effects use a 10,000-repetition two-way seed/subject pigeonhole
bootstrap so neither repeated windows nor repeated seed assignments are
treated as independent. The primary grouped estimand remains the unweighted
mean across held-out subjects.

After ten seeds, a dataset is flagged for extension to 20 seeds when at least
two co-primary size contrasts have confidence-interval half-width above
`max(0.01, 1% of the all-feature mean size)`. Extension is a precision decision,
not a result-dependent significance rescue.

## Checkpoints, resume, and provenance

Every long candidate search and final unique subset is stored as an atomic
checkpoint shard. A checkpoint directory is bound to a manifest containing
the dataset/seed unit, split ID, selection-data ID, model configuration,
scientific configuration hash, feature-selection digest, full grid, and Git
code version. `--resume` accepts only an identical manifest. A missing,
mismatched, or orphaned manifest aborts rather than silently mixing runs.

Dataset/seed units can be sharded across workers, but a single unit must have
only one writer. Partial seed runs write completion state but cannot produce
inferential summaries. Aggregation requires all ten planned seeds for every
included dataset.

Frozen UCI archives may be reused only when their SHA-256 checksums match.
Phase 7 tables may be reused for regression auditing and narrative context, not
as Phase 8 observations. Model outputs and selection evidence may be reused
only from manifest-compatible Phase 8 checkpoints.

Repository hygiene is part of the provenance correction. `.gitignore` now
excludes PyCharm `.idea/` and `*.iml` state, common local virtual-environment
directories (`.venv/`, `venv/`, `env/`, and `ENV/`), and locally packaged
`*.zip` bundles. Generated outputs were already ignored. Scientific source,
configuration, and documentation changes remain visible to `code_version`, so
a genuinely changed experiment still receives the `-dirty` marker.

## Computational budget

The approved initial ceiling is:

| Item | Budget |
|---|---:|
| Dataset/seed units | 30 |
| Primary grid cells per subset | 60 |
| Approximate classifier fit-equivalents | 65,560 |
| RFE/CRFE selector iterations, additional | 12,600 |
| Serial CPU time | 45--60 hours |
| Four-worker wall time | 12--20 hours |
| Required accelerator | None |

The fit count is an engineering estimate because recursive paths can share
cached subsets and early model stopping varies. The hard scientific budget is
the frozen grid and seed count, not a promise to exhaust every estimated fit.
The benchmark is CPU-oriented; four dataset/seed workers are the recommended
maximum to avoid memory pressure, especially on Covertype and HAR.

## Execution controls

The canonical specification is `configs/phase8_robustness.yaml`. The command
below validates and writes the complete plan without loading a dataset or
fitting a model:

```bash
python experiments/12_robustness_statistics.py \
  --config configs/phase8_robustness.yaml \
  --plan-only
```

The expensive `--run` action is intentionally separate. This protocol commit
must be clean and reviewed before that action is launched.
