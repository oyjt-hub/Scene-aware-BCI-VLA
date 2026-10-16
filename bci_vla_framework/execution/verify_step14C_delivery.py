"""Read-only evidence audit; writes a separate result without replaying actions."""
from pathlib import Path
import ast,hashlib,json,math,sys,zipfile
from collections import Counter
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from bci_vla.simulation_v3.backend import SimulationVLAAdapter
from bci_vla.simulation_v3.factory import empirical_outcomes,GOALS,OBJECTS
from bci_vla.simulation_v3.replay import replay_session_state
from bci_vla.simulation_v3 import replay as replay_module
from bci_vla.monitoring.journal import EventJournal
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def hashes(p):return all(sha(r['path'])==r['sha256'] for r in read(p)['files'])
checks=[]
def check(name,ok):
 if not ok:raise AssertionError(name)
 checks.append(name)
base=ROOT/'runs/pilot_robot_14C_full_pipeline';run=base/'v3/results'
s=read(run/'summary.json');eps=s['episodes'];by={e['id']:e for e in eps}
check('new delivery files frozen',hashes('execution/14C_source_manifest.json'))
check('old09 through15 source and evidence unchanged',all(hashes(f'execution/{i}_source_manifest.json') for i in ['09','10','11','12','12R','13','14','15']))
check('downloaded artifacts intact',hashes(base/'collected_manifest.json'))
job=read('execution/14C_remote_job_v3.json')
check('executed model bundle equals local frozen upload',sha(base/'v3/sources.zip')==job['archive_sha256'] and all(sha(r['path'])==r['sha256'] for r in job['files']))
with zipfile.ZipFile(base/'v3/sources.zip') as z:
 check('deployed namespace matches local implementation',all(z.read(str(q.relative_to(ROOT)).replace('\\','/'))==q.read_bytes() for q in (ROOT/'bci_vla/simulation_v3').glob('*.py')))
check('398 regressions passed',all(read('execution/14C_test_result.json')[k]==v for k,v in {'tests_run':398,'result':'PASS','failures':0,'errors':0,'skipped':0}.items()))
methods=[]
for f in ['bci_vla/contracts/session.py','bci_vla/simulation_v3/session.py']:
 c=next(n for n in ast.parse((ROOT/f).read_text()).body if isinstance(n,ast.ClassDef) and n.name=='InteractionSession')
 methods.append({n.name:ast.dump(n,include_attributes=False).replace('offline_fake_only','simulation_model_only') for n in c.body if isinstance(n,ast.FunctionDef)})
check('core ingestion planning confirmation authorization dispatch unchanged',all(methods[0][k]==methods[1][k] for k in ['begin_interaction','ingest','authorize','dispatch','request_policy','revise_context']))
expected={'nominal_cream','nominal_wine','changed_goal','changed_channel','joint_alias_conflict','late_confirmation','STOP_during_execution','goal_change_recovery','execution_failure_not_channel','inference_failure_not_channel'}
check('all ten planned cases reached and nonvacuous',set(by)==expected and len(eps)==10 and all(e['reason'] is None and e['checks'] and all(e['checks'].values()) for e in eps))
check('five development pick cases independently succeeded',all(by[k]['final_state']=='SUCCEEDED' and by[k]['faithful_task_success'] for k in ['nominal_cream','nominal_wine','changed_goal','changed_channel','joint_alias_conflict']))
check('late confirmation blocked without simulator steps',by['late_confirmation']['actual_sim_steps']==0 and by['late_confirmation']['checks']['late_confirmation_blocked'])
stop=by['STOP_during_execution'];rec=by['goal_change_recovery']
check('stop cancels and blocks subsequent action',stop['final_state']=='CANCELLED' and stop['actual_sim_steps']==8 and stop['checks']['STOP_queue_empty'] and stop['checks']['STOP_no_further_steps'])
check('active context recovery control passes but grasp failure retained',rec['final_state']=='FAILED' and not rec['task_success'] and rec['actual_sim_steps']==168 and len(rec['dispatches'])==2 and all(rec['checks'][k] for k in ['change_stops_and_drains_old','old_authorization_rejected','recovery_new_confirmation']))
check('injected execution and inference failures retained',by['execution_failure_not_channel']['final_state']=='FAILED' and by['execution_failure_not_channel']['actual_sim_steps']==1 and by['inference_failure_not_channel']['final_state']=='CANCELLED' and by['inference_failure_not_channel']['actual_sim_steps']==0 and len(by['inference_failure_not_channel']['dispatches'])==1)
check('private truth used to evaluate each dispatch separately',all(not e['wrong_target_start'] and all(d['goal']==d['intended_goal_on_dispatch'] for d in e['dispatches']) for e in eps))
proofs=[p for e in eps for p in e['mapping_repair_evidence']]
check('repair uses four public known probes and preserves goal prior',len(proofs)==4 and all([p['prompt'] for p in x['observations']]==['0','1','0','1'] and x['prior_from_joint_posterior_not_private_goal'] and x['goal_marginals_before']==x['initial_repaired_goal_marginals'] and 'repaired' in x['new_context']['mapping_id'] for x in proofs))
for k in ['changed_goal','changed_channel','joint_alias_conflict']:
 check('fresh confirmation after repair '+k,by[k]['checks']['mapping_repair_fresh_confirmation'])
journals=list(run.glob('*_journal.jsonl'));count=0
# Cross-process set iteration can change floating-point summation by one ULP.
# IDs, hashes, types, states and all nonnumeric evidence remain exact. This audit
# tolerance never changes the production filter, authorization or replay source.
roundoff=[]
def portable_equal(a,b,reason):
 def same(x,y):
  if type(x) is not type(y):return False
  if isinstance(x,dict):return x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
  if isinstance(x,(list,tuple)):return len(x)==len(y) and all(same(i,j) for i,j in zip(x,y))
  if isinstance(x,float) and x!=y:
   if not (math.isfinite(x) and math.isfinite(y) and abs(x-y)<=1e-12):return False
   roundoff.append(abs(x-y));return True
  return x==y
 if not same(a,b):raise AssertionError(reason)
with patch.object(replay_module,'_equal',side_effect=portable_equal),patch.object(SimulationVLAAdapter,'__init__',side_effect=AssertionError('no backend')),patch.object(SimulationVLAAdapter,'submit',side_effect=AssertionError('no submit')),patch.object(SimulationVLAAdapter,'query',side_effect=AssertionError('no query')),patch.object(SimulationVLAAdapter,'stop',side_effect=AssertionError('no stop')):
 for q in journals:
  journal=EventJournal.from_jsonl(q.read_text(encoding='utf-8'));state=replay_session_state(journal);assert state;count+=len(journal.records)
check('every session independently replayed with executor disabled',len(journals)==sum(e['sessions'] for e in eps) and all(e['journals_replayed'] for e in eps))
events=[json.loads(x) for x in (run/'simulator_events.jsonl').read_text().splitlines()]
steps=[e for e in events if e['kind']=='actual_simulation_step'];counts=Counter(e['label'] for e in steps)
check('actual simulator telemetry matches all episode step counts',all(counts[e['id']]==e['actual_sim_steps'] for e in eps) and len(steps)==549)
check('all actions finite seven dimensional',all(len(e['action'])==7 and all(math.isfinite(x) for x in e['action']) for e in steps))
check('actual predicate contact plus lift semantics retained',all(p['held']==(p['grasp_contact'] and p['lift_m']>.03) for e in steps for p in e['predicates'].values()))
last={e['label']:e for e in steps}
check('success endpoints agree with independent simulator predicates',all(last[e['id']]['predicates'][OBJECTS[GOALS.index(e['dispatched_goal'])][0]]['held'] for e in eps if e['task_success']))
records=read(run/'independent_trajectories.json');cal=[r for r in records if r['split']=='calibration'];held=[r for r in records if r['split']=='calibration_holdout']
check('independent calibration and holdout disjoint from test init3',len(cal)==4 and len(held)==2 and {r['init_state_index'] for r in cal}=={0,1} and {r['init_state_index'] for r in held}=={2} and s['calibration_holdout_test_init_indices']==[[0,1],[2],[3]])
check('both heldout failures retained',all(not any(r['held'].values()) for r in held))
check('v1 calibration reused exactly without favorable reselection',sha(run/'independent_trajectories.json')==sha(base/'v1/results/independent_trajectories.json')==read('execution/14C_run_plan_v3.json')['reused_calibration_sha256'])
check('outcome table derived from calibration only',empirical_outcomes(records,s['model_checkpoint'])==read(run/'calibrated_outcomes.json')==empirical_outcomes(cal,s['model_checkpoint']))
check('model checkpoint and nonphysical scope explicit',s['model_checkpoint']=='26b36799a702ca920467c987096a8ccb5f476c6965af6c7f2fd419ecb6893bf2' and s['real_BCI_trials']==s['physical_robot_commands']==0 and all(not e['physical_execution_allowed'] for e in eps))
check('temporary model service stopped',read(base/'v3/orchestration.json')['worker_exit_code']==0 and read(base/'v3/orchestration.json')['owned_server_stopped'])
check('failed prototypes retained',read(base/'v1/orchestration.json')['worker_exit_code']!=0 and not read(base/'v2/results/summary.json')['all_checks_passed'])
check('videos and final frames retained',all((run/(k+'.mp4')).stat().st_size>1000 and (run/(k+'_last.png')).read_bytes().startswith(b'\x89PNG') for k in ['nominal_cream','nominal_wine','changed_goal','changed_channel','joint_alias_conflict','STOP_during_execution','goal_change_recovery']))
p=read('execution/progress.json');c=read('execution/14C_coverage.json')
check('nonreal completed while real branch and research remain open',p['current_step'] in {'14','15'} and p['steps'][13]['nonreal_closed_loop_completed'] and c['full_framework_simulation_completed'] and not c['real_step14_completed'] and not p['research_gates']['G2']['research_acceptance_passed'] and p['research_gates']['G3']['status']=='未验证')
check('no formal superiority or measured human cost claim',not s['statistical_strategy_superiority_claim'] and not s['real_user_time_measured'] and not c['formal_success_rate_claim'])
check('old delivery verifiers separately passed',all(read(f'execution/{v}_integrity_recheck_after_step14C.json')['result']=='PASS' for v in ['12R','13','14','15']))
check('all TXT BOM CRLF',all(q.read_bytes().startswith(b'\xef\xbb\xbf') and b'\n' not in q.read_bytes().replace(b'\r\n',b'') for q in ROOT.rglob('*.txt')))
value=dict(date='2026-10-16',result='PASS',scope='step14_nonreal_full_framework_engineering_only',count=len(checks),checks=checks,software_tests=398,
 independently_replayed_sessions=len(journals),journal_records=count,actual_functional_sim_steps=len(steps),completed_pick_cases=5,
 cross_process_float_audit_tolerance=1e-12,cross_process_max_float_difference=max(roundoff,default=0),
 active_goal_change_grasp_success=False,heldout_grasp_successes=0,real_BCI_trials=0,physical_robot_commands=0,G2_passed=False,G3_passed=False)
destination='execution/14C_integrity_recheck_after_step15C.json' if p['steps'][14].get('nonreal_migration_completed') else 'execution/14C_integrity_result.json'
(ROOT/destination).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(value,ensure_ascii=True,indent=2))
