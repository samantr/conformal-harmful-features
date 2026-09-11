"""Read frozen user-supplied outputs; write separate Phase 9 audit evidence.

No experimental imports, fitting, tuning, calibration, new tests of significance,
or modification of source outputs. Usage: python paper/audit_phase9_outputs.py
--inputs ../upload --out paper/phase9_audit
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--inputs', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

def source(stem, suffix='.csv'):
    matches = list(args.inputs.glob(stem + '*' + suffix))
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one source for {stem}: {matches}')
    return matches[0]

def read(stem):
    return pd.read_csv(source(stem))

manifest = []
for f in sorted(args.inputs.iterdir()):
    if f.is_file():
        h = hashlib.sha256()
        with f.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
        manifest.append(dict(file=f.name, bytes=f.stat().st_size, sha256=h.hexdigest()))

with zipfile.ZipFile(source('phase8_all_results', '.zip')) as archive:
    with archive.open('phase8_all_results.csv') as stream:
        # Large serialized ranking vectors are unnecessary for this row audit.
        d = pd.read_csv(stream, usecols=lambda c: c not in {
            'feature_scores', 'feature_ranking', 'selected_features', 'rejected_temperatures'
        }, low_memory=False)

checks = {}
def check(name, condition):
    checks[name] = bool(condition)
    if not condition:
        raise ValueError(name)

expected = {'covertype': 10, 'dry_bean': 20, 'human_activity_recognition': 20}
check('75800_rows', len(d) == 75800)
check('50_units', len(d[['dataset', 'seed']].drop_duplicates()) == 50)
check('seed_counts', d.groupby('dataset').seed.nunique().to_dict() == expected)
check('seed_values', all(set(g.seed) == set(range(43, 43 + expected[k])) for k,g in d.groupby('dataset')))
check('scientific_code', set(d.code_version) == {'e4b3645'})
check('single_split_per_unit', d.groupby(['dataset','seed'])[['split_id','selection_data_id']].nunique().eq(1).all().all())
check('recorded_no_heldout_selection', not d.final_calibration_used_for_selection.any() and not d.final_test_used_for_selection.any())
check('recorded_subsets_frozen', d.subset_frozen_before_final_calibration.all())
check('raw_flags', d.numerical_boundary_flag.sum() == 10006)
check('induced_flags', d.scaling_induced_boundary_flag.sum() == 5115)

methods = ['all_features','conformal_harm_one_shot','conformal_harm_recursive']
selected = d[d.method.isin(methods) & ((d.method == 'all_features') | d.phase8_primary_selected)]
aps = selected[(selected.score == 'aps') & (selected.alpha == .1)]
primary = aps[aps.scaling == 'base']
check('primary_300_unique_rows', len(primary) == 300 and not primary.duplicated(['dataset','model','method','seed']).any())
metrics = ['mean_size','accuracy','coverage','ece','sscv','class_coverage_gap']
primary.groupby(['dataset','model','method'])[metrics].mean().reset_index().to_csv(args.out/'primary_absolute.csv',index=False)

errors = []
for file, metric, direction in [('phase8_paired_size_effects','mean_size',-1),('phase8_paired_accuracy_effects','accuracy',1)]:
    for r in read(file).itertuples():
        g = primary[(primary.dataset==r.dataset)&(primary.model==r.model)]
        a = g[g.method==r.method].set_index('seed')[metric]
        b = g[g.method=='all_features'].set_index('seed')[metric]
        delta = direction*(a-b)
        check(f'{file}:{r.dataset}:{r.model}:{r.method}', len(delta)==r.n_pairs and np.isclose(delta.mean(),r.mean_difference,atol=1e-12))
        errors.append(abs(delta.mean()-r.mean_difference))

# Direct within-seed removal overlap, excluding no-removal pairs explicitly.
overlaps = []
for (dataset,model,seed), g in d[(d.score=='aps')&(d.alpha==.1)&(d.scaling=='base')].groupby(['dataset','model','seed']):
    universe = set(json.loads(g[g.method=='all_features'].iloc[0].selected_indices))
    for method in methods[1:]:
        p = g[(g.method==method)&g.phase8_primary_selected].iloc[0]
        removed = universe-set(json.loads(p.selected_indices))
        if not removed:
            continue
        for standard in ['crfe','mutual_information','permutation_importance','rfe','shap']:
            candidates = g[(g.method==standard)&(g.n_features==p.n_features)]
            check(f'match:{dataset}:{model}:{seed}:{method}:{standard}', len(candidates)==1)
            other = universe-set(json.loads(candidates.iloc[0].selected_indices))
            overlaps.append(dict(dataset=dataset,model=model,proposed=method,standard=standard,seed=seed,
                removed=len(removed),jaccard=len(removed&other)/len(removed|other),
                shared_fraction=len(removed&other)/len(removed),exact_match=float(removed==other)))
o = pd.DataFrame(overlaps)
summary = o.groupby(['dataset','model','proposed','standard']).agg(n=('seed','size'),mean_removed=('removed','mean'),jaccard_mean=('jaccard','mean'),jaccard_median=('jaccard','median'),shared_fraction_mean=('shared_fraction','mean'),exact_match_fraction=('exact_match','mean')).reset_index()
keys=['dataset','model','proposed','standard']
actual=summary.set_index(keys).sort_index()
given=read('phase8_h1_removed_overlap_summary').set_index(keys).sort_index()
check('H1_all_summary_fields_match', actual.index.equals(given.index) and np.allclose(actual[given.columns],given,atol=1e-12))
summary.to_csv(args.out/'h1_overlap_verified.csv',index=False)

# Verify supplied H3 mean contrasts only; no new inferential analysis.
h3errors=[]
for score,filename in [('aps','phase8_h3_aps_summary'),('raps','phase8_h3_raps_summary')]:
    grid=selected[selected.score==score]
    for r in read(filename).itertuples():
        alpha=.1 if score=='aps' else r.alpha
        g=grid[(grid.dataset==r.dataset)&(grid.model==r.model)&(grid.alpha==alpha)]
        # RAPS summary is the equal-weight mean of the nine frozen cells per seed.
        table=g.groupby(['seed','method','scaling']).mean_size.mean().unstack(['method','scaling'])
        A=table[('all_features','base')]; B=table[(r.method,'base')]
        C=table[('all_features',r.scaling)]; D=table[(r.method,r.scaling)]
        values=[(C-D).mean(),(A-C).mean(),(A-D).mean(),(B+C-A-D).mean()]
        expected_values=([r.feature_size_gain,r.scaling_size_gain_all,r.combined_size_gain_vs_base_all,r.interaction]
            if score=='aps' else [r.feature_gain_mean,r.scaling_gain_all_mean,r.combined_gain_mean,r.interaction_mean])
        h3errors.extend(np.abs(np.array(values)-expected_values))
check('H3_APS_RAPS_means_match', max(h3errors)<1e-12)

# HAR aggregate embeds per-subject coverage, but not per-subject set sizes.
har=d[d.dataset=='human_activity_recognition']
embedded_count=sum(len(json.loads(s)) for s in har.group_coverages)
check('146500_embedded_subject_coverage_entries',embedded_count==146500)
subject=read('phase8_har_subject_effects')
for r in subject[subject.metric=='coverage_difference_vs_all'].itertuples():
    g=primary[(primary.dataset==r.dataset)&(primary.model==r.model)]
    a=g[g.method==r.method].set_index('seed'); b=g[g.method=='all_features'].set_index('seed')
    deltas=[]
    for seed in a.index:
        x=json.loads(a.loc[seed,'group_coverages']); y=json.loads(b.loc[seed,'group_coverages'])
        check(f'paired_subjects:{r.model}:{r.method}:{seed}',set(x)==set(y))
        deltas.extend(x[s]-y[s] for s in x)
    check(f'subject_mean:{r.model}:{r.method}',np.isclose(np.mean(deltas),r.mean_difference,atol=1e-12))

rank=read('phase8_rank_stability_pairs')
rs=read('phase8_rank_stability_summary')
rankkeys=['dataset','model','method','stability_type','top_k']
for _,r in rs.iterrows():
    g=rank
    for k in rankkeys: g=g[g[k]==r[k]]
    check('rank:'+':'.join(str(r[k]) for k in rankkeys),len(g)==r.seed_pairs and all(np.isclose(g[c].mean(),r[c+'_mean'],equal_nan=True) for c in ['spearman','jaccard','kuncheva']))

report=dict(status='PASS_WITH_SCOPE_LIMITATIONS',checks=checks,inputs=manifest,
    primary_mean_max_error=max(errors),h3_mean_max_error=float(max(h3errors)),
    subject_coverage_entries=embedded_count,
    limitations=['Raw phase8_all_subject_results.csv absent; embedded coverage entries and subject coverage means verified, but subject size rows and bootstrap intervals not independently reproduced.',
    'Stored exact p-values, Holm corrections and intervals are reported as supplied; this audit verifies primary effect means, not an independent full statistical reimplementation.',
    'Recorded split/selection provenance checked; original split-index files and 50 individual completion manifests not supplied.',
    'H3 means verified; supplied H3 interval-generation code/provenance not supplied. Treat as descriptive post-hoc summaries, not new confirmatory evidence.'])
(args.out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['checks','inputs']},indent=2))
print('Passed checks:',len(checks))
