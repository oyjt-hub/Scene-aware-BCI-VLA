"""Full independent checkpoint replay and source/privacy/statistics audit."""
from pathlib import Path
from datetime import datetime
from collections import Counter
from unittest.mock import patch
import gzip,hashlib,json,math,subprocess,sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from bci_vla.adapters.vla import FakeVLAAdapter
from bci_vla.contracts import replay as replay_module
from bci_vla.monitoring.journal import EventJournal
from bci_vla.study_v1.statistics import paired_summary,judge
from bci_vla.benchmark.offline import make_session
def read(p):return json.loads((R/p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def hashes(p):return all(sha(x['path'])==x['sha256'] for x in read(p)['files'])
checks=[];roundoff=[]
def check(n,b):
 if not b:raise AssertionError(n)
 checks.append(n)
def portable_equal(a,b,reason):
 def same(x,y):
  if type(x)!=type(y):return False
  if isinstance(x,dict):return x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
  if isinstance(x,(tuple,list)):return len(x)==len(y) and all(same(i,j) for i,j in zip(x,y))
  if isinstance(x,float) and x!=y:
   if not (math.isfinite(x) and math.isfinite(y) and abs(x-y)<=1e-12):return False
   roundoff.append(abs(x-y));return True
  return x==y
 if not same(a,b):raise AssertionError(reason)
base=R/'runs/registered_step16_synthetic_v1';cfg=read('configs/study16.json');s=read(base/'summary.json');lock=read('execution/16_preregistration.json')
check('new step16 delivery files frozen',hashes('execution/16_source_manifest.json'))
check('all previous source/evidence unchanged',all(hashes(f'execution/{i}_source_manifest.json') for i in ['09','10','11','12','12R','13','14','15','14C','15C']))
check('registered source and parameter lock survived run',all(sha(x['path'])==x['sha256'] for x in lock['files']) and sha(R/'configs/study16.json')==lock['config_sha256'])
receipt=read(base/'start_receipt.json');gitreceipt=read('execution/16_registration_git_receipt.json');commit=gitreceipt['protocol_commit']
check('local lock predates new run',datetime.fromisoformat(lock['created_utc'])<datetime.fromisoformat(receipt['started_utc']) and receipt['preregistration_sha256']==sha(R/'execution/16_preregistration.json'))
for q in ['configs/study16.json','docs/16_registered_protocol.txt','execution/16_preregistration.json','bci_vla/study_v1/statistics.py','bci_vla/study_v1/runner.py','bci_vla/study_v1/world.py']:
 raw=subprocess.check_output(['git','show',commit+':bci_vla_framework/'+q],cwd=R.parent)
 check('pre-run Git object contains unchanged '+q,raw==(R/q).read_bytes())
check('no false external preregistration or power claim',not lock['externally_registered'] and not cfg['registered_with_external_service'] and cfg['formal_human_sample_size'] is None and cfg['conservative_blocks_for_005_hoeffding_halfwidth']>32)
check('409 software regressions passed',all(read('execution/16_test_result.json')[k]==v for k,v in dict(tests_run=409,failures=0,errors=0,skipped=0,result='PASS').items()))
check('raw result file hashes intact',hashes(base/'artifacts_manifest.json'))
rows=[json.loads(l) for l in (base/'evaluation_joined.jsonl').read_text(encoding='utf-8').splitlines()]
lookup={r['episode_id']:r for r in rows}
check('all planned rows unique and balanced',len(rows)==len(lookup)==s['episodes']==cfg['episodes']==6272 and Counter((r['case_id'],r['seed']) for r in rows)==Counter({(c['id'],seed):14 for c in cfg['cases'] for seed in cfg['seeds']}) and all(Counter(r['method'] for r in rows if r['case_id']==c['id'])==Counter({m:32 for m in cfg['methods']}) for c in cfg['cases']))
check('new seeds disjoint from previous development',set(cfg['seeds']).isdisjoint(cfg['old_development_seeds']) and set(r['seed'] for r in rows)==set(cfg['seeds']))
check('all failures pauses and full episode times retained',not s['runtime_errors'] and all(r['runtime_error'] is None and math.isfinite(r['elapsed_s']) and r['elapsed_s']>=0 for r in rows) and all(r['unresolved']==(not r['faithful_virtual_completion']) for r in rows))
check('parameter update and subjective metrics explicitly absent',all(r['parameter_update_error_rate'] is None for r in rows) and s['subjective_results'] is None and cfg['subjective_results'] is None)
goals=make_session('evaluation-audit').policy.goals;private_ids=set();invalid_windows=0
with gzip.open(base/'evaluation_private/worlds.jsonl.gz','rt',encoding='utf-8') as f:
 for line in f:
  w=json.loads(line);e=lookup[w['episode_id']];private_ids.add(w['episode_id'])
  at=e['elapsed_s'] if e['submission_time_s'] is None else e['submission_time_s']
  goal=goals[sum(t<=at for t in w['goal_change_times'])%2];mode='flip' if sum(t<=at for t in w['channel_change_times'])%2 else 'normal'
  assert goal==e['truth_goal_at_dispatch_or_end'] and mode==e['truth_channel_at_dispatch_or_end']
  counts=Counter()
  for d in w['draws']:
   counts[d['mode']]+=1;key=f"{w['seed']}|{d['mode']}|{counts[d['mode']]}|{d['duration_s']}"
   assert int(hashlib.sha256(key.encode()).hexdigest()[:16],16)/2**64==d['draw']
   assert abs(sum(d['distribution'].values())-1)<1e-12
   if d['invalid_nonstationary_window_suppressed']:assert d['output']=='NO_RESPONSE';invalid_windows+=1
check('private independent timelines and external draws auditable',private_ids==set(lookup) and invalid_windows>0)
public={};actionseq={}
with gzip.open(base/'online/episodes.jsonl.gz','rt',encoding='utf-8') as f:
 for line in f:
  p=json.loads(line);e=lookup[p['episode_id']];assert not {'case_id','seed','true_goal','true_channel','private_truth'}&p.keys()
  assert p['method']==e['method'] and not p['physical_execution_allowed'] and p['replay_matches_each_checkpoint']
  command=p['commanded_goal'];assert e['wrong_start']==(command is not None and command!=e['truth_goal_at_dispatch_or_end'])
  assert e['faithful_virtual_completion']==(command==e['truth_goal_at_dispatch_or_end'] and p['final_state']=='SUCCEEDED')
  assert e['unstarted']==(command is None)
  assert abs(sum(x['end_s']-x['start_s'] for x in p['intervals'])-e['elapsed_s'])<1e-5
  if e['method']=='A_no_probe':assert all(d['selected_action']['kind']!='CHANNEL_PROBE' for d in p['decisions'])
  if e['method']=='A_fixed_duration':assert all(d['selected_action']['duration_s'] in [0,1.] for d in p['decisions'])
  public[p['episode_id']]=dict(final_state=p['final_state'],journal_records=p['journal_records'],journal_last_hash=p['journal_last_hash'])
  actionseq[e['case_id'],e['seed'],e['method']]=[d['selected_action']['action_id'] for d in p['decisions']]
check('public truth separation and independently joined completion',set(public)==set(lookup))
check('execution endpoints do not update channel beliefs',all(r['execution_does_not_update_channel'] is True for r in rows if not r['unstarted']))
journal_ids=set();count=0
with patch.object(replay_module,'_equal',side_effect=portable_equal),patch.object(FakeVLAAdapter,'__init__',side_effect=AssertionError('no backend')),patch.object(FakeVLAAdapter,'submit',side_effect=AssertionError('no submit')),patch.object(FakeVLAAdapter,'query',side_effect=AssertionError('no query')),patch.object(FakeVLAAdapter,'stop',side_effect=AssertionError('no stop')):
 for path in sorted((base/'journals').glob('*.gz')):
  with gzip.open(path,'rt',encoding='utf-8') as f:
   for line in f:
    d=json.loads(line);j=EventJournal.from_jsonl('\n'.join(json.dumps(x,ensure_ascii=False) for x in d['records']))
    state=replay_module.replay_session_state(j);p=public[d['episode_id']]
    assert state['state']==p['final_state'] and len(j.records)==p['journal_records'] and j.records[-1]['record_hash']==p['journal_last_hash']
    assert d['episode_id'] not in journal_ids;journal_ids.add(d['episode_id']);count+=len(j.records)
  print('replayed '+path.stem,flush=True)
check('every journal independently rebuilt with all executors disabled',journal_ids==set(lookup))
statistics=read(base/'paired_statistics.json');primary=[paired_summary(rows,'M','B9',m,bootstrap=cfg['bootstrap_resamples'],alpha=cfg['primary_alpha_each']) for m in cfg['primary_metrics']]
check('locked primary intervals and judgement recomputed',primary==statistics['primary'] and judge(primary)==statistics['decision']==s['judgement'])
secondary=[paired_summary(rows,'M',m,k,bootstrap=cfg['bootstrap_resamples']) for m in cfg['methods'] if m not in ['M','B9'] for k in cfg['primary_metrics']]
check('all descriptive comparisons retained',secondary==statistics['secondary_descriptive'])
check('action equality reported honestly',s['M_B9_action_sequences_equal']==all(actionseq[c['id'],seed,'M']==actionseq[c['id'],seed,'B9'] for c in cfg['cases'] for seed in cfg['seeds']))
check('mechanism and safety faults passed separately',all(read(base/'mechanisms.json')['checks'].values()) and read(base/'faults.json')['all_passed'])
p=read('execution/progress.json');coverage=read('execution/16_coverage.json')
check('no false formal G4 completion',p['current_step']=='16' and p['steps'][15]['nonreal_study_completed'] and not coverage['whole_step16_completed'] and not p['research_gates']['G4']['research_acceptance_passed'] and not s['formal_G4_passed'])
check('G2 G3 unchanged',not p['research_gates']['G2']['research_acceptance_passed'] and p['research_gates']['G3']['status']=='未验证')
check('zero new human robot or actual VLA experiments',s['real_BCI_trials']==s['physical_robot_commands']==s['new_LIBERO_rollouts']==0)
check('all predecessor integrity checks separately passed',all(read(f'execution/{x}_integrity_recheck_after_step16.json')['result']=='PASS' for x in ['12R','13','14','15','14C','15C']))
check('all TXT retain BOM CRLF',all(q.read_bytes().startswith(b'\xef\xbb\xbf') and b'\n' not in q.read_bytes().replace(b'\r\n',b'') for q in R.rglob('*.txt')))
v=dict(result='PASS',count=len(checks),checks=checks,software_tests=409,episodes=len(rows),independent_journals=len(journal_ids),journal_records=count,
 maximum_cross_process_roundoff=max(roundoff,default=0),float_audit_tolerance=1e-12,invalid_windows_suppressed=invalid_windows,
 complete_time_overruns=sum(r['elapsed_s']>cfg['episode_budget_s'] for r in rows),formal_G4_passed=False,real_BCI_trials=0,physical_robot_commands=0)
(R/'execution/16_integrity_result.json').write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');print(json.dumps(v,indent=2))
