"""Read-only frozen HAR verification. No fitting or new inference.

Usage: python paper/verify_phase9_har.py --subjects CSV --aggregate ZIP_OR_CSV
--effects CSV --subject-archive RAR --out DIRECTORY
Only selected columns are retained; CSVs are read in 5,000-row chunks.
Frozen functions are loaded verbatim from git e4b3645 by AST, avoiding runner imports.
"""
import argparse, ast, hashlib, json, subprocess, zipfile
import ctypes as ct
import ctypes.util
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd
import yaml

p=argparse.ArgumentParser()
for name in ['subjects','aggregate','effects','subject-archive','out']:
    p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--aggregate-archive',type=Path)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
report={'checks':{},'max_errors':{},'inputs':[]}
def check(name, ok):
    report['checks'][name]=bool(ok)
def close(name,x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    report['max_errors'][name]=float(np.max(np.abs(x-y)))
    check(name,np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.allclose(x,y,atol=1e-12,rtol=0))
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()}
for path in [a.subjects,a.aggregate,a.effects,a.subject_archive]:report['inputs'].append(digest(path))

def archive_members(path):
    lib=ct.CDLL(ct.util.find_library('archive'))
    lib.archive_read_new.restype=ct.c_void_p
    for name,types,result in [
        ('archive_read_support_format_all',[ct.c_void_p],ct.c_int),
        ('archive_read_support_filter_all',[ct.c_void_p],ct.c_int),
        ('archive_read_open_filename',[ct.c_void_p,ct.c_char_p,ct.c_size_t],ct.c_int),
        ('archive_read_next_header',[ct.c_void_p,ct.POINTER(ct.c_void_p)],ct.c_int),
        ('archive_entry_pathname',[ct.c_void_p],ct.c_char_p),
        ('archive_read_data',[ct.c_void_p,ct.c_void_p,ct.c_size_t],ct.c_ssize_t),
        ('archive_read_free',[ct.c_void_p],ct.c_int)]:
        fun=getattr(lib,name);fun.argtypes=types;fun.restype=result
    handle=lib.archive_read_new()
    lib.archive_read_support_format_all(handle);lib.archive_read_support_filter_all(handle)
    rows=[]
    try:
        if lib.archive_read_open_filename(handle,str(path).encode(),10240)!=0:
            raise RuntimeError('Cannot open archive '+str(path))
        entry=ct.c_void_p();buffer=ct.create_string_buffer(8388608)
        while True:
            status=lib.archive_read_next_header(handle,ct.byref(entry))
            if status==1:break
            if status!=0:raise RuntimeError('Invalid archive header')
            name=lib.archive_entry_pathname(entry).decode();sha=hashlib.sha256();size=0
            while True:
                n=lib.archive_read_data(handle,buffer,len(buffer))
                if n==0:break
                if n<0:raise RuntimeError('Archive read failed')
                sha.update(buffer.raw[:n]);size+=n
            rows.append(dict(archive=str(path),member=name,bytes=size,sha256=sha.hexdigest()))
    finally:lib.archive_read_free(handle)
    return rows
members=archive_members(a.subject_archive)
check('uploaded_RAR_subject_matches_extracted_CSV',len(members)==1 and members[0]['sha256']==report['inputs'][0]['sha256'])
if a.aggregate_archive:
    report['inputs'].append(digest(a.aggregate_archive))
    other=archive_members(a.aggregate_archive);members+=other
    sha=hashlib.sha256()
    if a.aggregate.suffix=='.zip':
        archive=zipfile.ZipFile(a.aggregate);stream=archive.open('phase8_all_results.csv')
    else:stream=a.aggregate.open('rb')
    with stream:
        for b in iter(lambda:stream.read(8388608),b''):sha.update(b)
    check('uploaded_RAR_aggregate_matches_verified_ZIP_member',len(other)==1 and sha.hexdigest()==other[0]['sha256'])
report['archive_members']=members
(a.out/'archive_members.json').write_text(json.dumps(members,indent=2)+'\n')

print('Input hashes recorded',flush=True)
ns={'np':np,'pd':pd,'Mapping':Mapping,'Any':Any}
report['frozen_sources']=[]
for file,names in [('src/chf/experiments/statistics.py',['two_way_cluster_bootstrap']),('src/chf/experiments/phase8_runner.py',['_primary_proposed','_paired_subject_bootstrap_table'])]:
    raw=subprocess.check_output(['git','show','e4b3645:'+file])
    current=Path(file).read_bytes()
    check('frozen_source_unchanged:'+file,raw==current)
    report['frozen_sources'].append({'path':file,'commit':'e4b3645','sha256':hashlib.sha256(raw).hexdigest(),'functions':names})
    tree=ast.parse(raw)
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node],type_ignores=[]),file,'exec'),ns)
specraw=subprocess.check_output(['git','show','e4b3645:configs/phase8_robustness.yaml'])
spec=yaml.safe_load(specraw)
report['settings']={'statistics':spec['statistics'],'selection':spec['selection'],'config_sha256':hashlib.sha256(specraw).hexdigest(),'primary':'frozen _primary_proposed: selected marker, Base APS alpha .10; both proposed paths; original CSV row order preserved'}
skip={'selected_features','feature_scores','feature_ranking'}
s=pd.concat(pd.read_csv(a.subjects,usecols=lambda c:c not in skip,chunksize=5000),ignore_index=True)
if a.aggregate.suffix=='.zip':
    z=zipfile.ZipFile(a.aggregate);f=z.open('phase8_all_results.csv')
else:f=a.aggregate.open('rb')
with f:
    chunks=list(pd.read_csv(f,usecols=lambda c:c not in skip|{'rejected_temperatures'},chunksize=5000))
d=pd.concat(chunks,ignore_index=True);h=d[d.dataset=='human_activity_recognition'].copy()
print('Read',len(s),'subject rows and',len(d),'aggregate rows',flush=True)
check('75800_aggregate_rows',len(d)==75800)
check('50_units',len(d[['dataset','seed']].drop_duplicates())==50)
check('20_10_20_seeds',d.groupby('dataset').seed.nunique().to_dict()=={'covertype':10,'dry_bean':20,'human_activity_recognition':20})
check('146500_subject_rows',len(s)==146500)
check('HAR_only_20_seeds',set(s.dataset)=={'human_activity_recognition'} and set(s.seed)==set(range(43,63)))
check('scientific_code',set(s.code_version)==set(d.code_version)=={'e4b3645'})
for flag,expected in [('classifier_refit',True),('subset_frozen_before_final_calibration',True),('final_calibration_used_for_selection',False),('final_test_used_for_selection',False)]:check(flag,s[flag].eq(expected).all())
cond=['dataset','seed','model','alpha','scaling','score','raps_lambda','raps_k_reg']
key=cond+['method','target_size','n_features','repetition']
skey=key+['subject_id']
report['pairing_keys']={'aggregate':key,'subject':skey,'all_feature_reference':cond+['subject_id'],'nulls':'APS lambda/k are structurally absent; pandas joins match null keys, groupby(dropna=False) retains them.'}
check('unique_aggregate_keys',not h.duplicated(key).any())
check('unique_subject_keys',not s.duplicated(skey).any())
metrics=['accuracy','coverage','mean_size','size_p90']
required=[c for c in skey if c not in ['raps_lambda','raps_k_reg']]+['split_id','selection_data_id','code_version','selected_indices','window_count']+metrics+[c for c in s if c.startswith('all_features_') or c.endswith('_vs_all')]
check('no_missing_required',not s[required].isna().any().any())
report['missing_by_column']=s.isna().sum().to_dict()
check('structural_APS_nulls',s.loc[s.score=='aps',['raps_lambda','raps_k_reg']].isna().all().all() and s.loc[s.score=='raps',['raps_lambda','raps_k_reg']].notna().all().all())
check('subject_ID_domain',s.subject_id.between(1,30).all() and s.subject_id.mod(1).eq(0).all() and s.subject_id.nunique()==26)
check('positive_integer_window_counts',s.window_count.gt(0).all() and s.window_count.mod(1).eq(0).all())
check('consistent_window_counts',s.groupby(['seed','subject_id']).window_count.nunique().eq(1).all())
check('five_subjects_per_condition',s.groupby(key,dropna=False).subject_id.nunique().eq(5).all())
provenance=['experiment','split_id','selection_data_id','code_version','selected_indices','phase8_primary_selected','selection_seed']
j=s.merge(h[key+provenance+['group_coverages']],on=key,how='left',validate='many_to_one',suffixes=('','_aggregate'),indicator=True)
check('every_subject_has_aggregate',j._merge.eq('both').all())
for c in provenance:check('aggregate_metadata:'+c,j[c].eq(j[c+'_aggregate']).all())
expected=h[key+['group_coverages']].copy();expected['subject_id']=expected.group_coverages.map(lambda x:[int(k) for k in json.loads(x)]);expected=expected.explode('subject_id')
check('exact_expected_subject_keys',len(expected)==len(s) and expected.merge(s[skey],on=skey,how='outer',indicator=True)._merge.eq('both').all())
close('embedded_subject_coverage',j.coverage,[json.loads(x)[str(v)] for x,v in zip(j.group_coverages,j.subject_id)])
ref=s[s.method=='all_features'][cond+['subject_id']+metrics+['window_count']]
check('unique_all_feature_references',not ref.duplicated(cond+['subject_id']).any())
j=s.merge(ref,on=cond+['subject_id'],how='left',validate='many_to_one',suffixes=('','_reference'),indicator=True)
check('all_references_present',j._merge.eq('both').all())
check('paired_window_counts',j.window_count.eq(j.window_count_reference).all())
for m in metrics:
    close('stored_reference:'+m,j['all_features_'+m],j[m+'_reference'])
    close('subject_difference:'+m,j[m+'_difference_vs_all'],j[m]-j[m+'_reference'])
close('subject_size_reduction',j.mean_size_reduction_vs_all,j.mean_size_reference-j.mean_size)
# Weighted means reconstruct window metrics; quantiles cannot be averaged.
for m in ['accuracy','coverage','mean_size','accuracy_difference_vs_all','coverage_difference_vs_all','mean_size_reduction_vs_all']:
    weighted=s[key].copy();weighted['total']=s[m]*s.window_count;weighted['n']=s.window_count
    w=weighted.groupby(key,dropna=False)[['total','n']].sum().reset_index();w['value']=w.total/w.n
    merged=w.merge(h,on=key,validate='one_to_one',how='outer',indicator=True)
    check('all_aggregate_cells:'+m,merged._merge.eq('both').all())
    col={'accuracy_difference_vs_all':'accuracy_loss_vs_all','coverage_difference_vs_all':'coverage_delta_vs_all'}.get(m,m)
    sign=-1 if m=='accuracy_difference_vs_all' else 1
    close('window_weighted:'+m,merged.value,sign*merged[col])
report['population']={'aggregate_HAR_rows':len(h),'subject_rows':len(s),'unique_subjects':sorted(s.subject_id.unique().tolist()),'subjects_by_seed':s.groupby('seed').subject_id.unique().map(lambda x:sorted(x.tolist())).to_dict(),'rows_by_seed':s.groupby('seed').size().to_dict()}
primary=ns['_primary_proposed'](s,spec)
check('600_primary_subject_rows',len(primary)==600)
check('100_cells_per_primary_group',primary.groupby(['model','method']).size().eq(100).all())
primary.drop(columns=['selected_indices']).to_csv(a.out/'har_primary_subjects.csv',index=False)
print('Row/reference reconciliation complete; reproducing eight frozen bootstraps',flush=True)
actual=ns['_paired_subject_bootstrap_table'](s,spec)
given=pd.read_csv(a.effects);bkeys=['dataset','model','method','metric']
x=actual.set_index(bkeys).sort_index();y=given.set_index(bkeys).sort_index()
check('eight_summary_keys',len(x)==8 and x.index.equals(y.index))
for c in y:close('bootstrap:'+c,x[c],y[c])
actual.to_csv(a.out/'har_subject_effects_reproduced.csv',index=False)
report['environment']={'numpy':np.__version__,'pandas':pd.__version__}
report['limitations']=['Original train/tune/calibration/test split-index files and individual unit manifests unavailable: recorded provenance and expected test subjects reconciled, but full split disjointness not independently re-established.','Per-window predictions/set sizes unavailable: accuracy, coverage and mean size reconcile by window weights; pooled size_p90 cannot be reconstructed from subject quantiles. Subject size_p90 references and differences are checked.','Existing H3 interval-generation source remains unavailable; this audit does not reproduce H3 intervals or other seed-level tests.','Large feature_scores, feature_ranking and selected_features serialization excluded; selected_indices is retained and checked against aggregate.','Subject accuracy is verified descriptively; frozen subject-effects table has size and coverage only. No new accuracy interval is computed.']
report['status']='PASS_WITH_SCOPE_LIMITATIONS' if all(report['checks'].values()) else 'DISCREPANCIES_FOUND'
(a.out/'har_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(report['status']);print('Failed checks:',[k for k,v in report['checks'].items() if not v]);print('Maximum error',max(report['max_errors'].values()))
