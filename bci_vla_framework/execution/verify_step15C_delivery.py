"""Verify frozen full simulator migration; no model or physical executor calls."""
from pathlib import Path
from collections import Counter
from unittest.mock import patch
import hashlib,json,math,sys,zipfile
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from bci_vla.simulation_v3.backend import SimulationVLAAdapter
from bci_vla.simulation_v3 import replay
from bci_vla.monitoring.journal import EventJournal
from bci_vla.simulation_transfer_v4 import factory
def read(p):return json.loads((R/p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def hashes(p):return all(sha(x['path'])==x['sha256'] for x in read(p)['files'])
checks=[];differences=[]
def check(n,b):
 if not b:raise AssertionError(n)
 checks.append(n)
def equal(a,b,reason):
 def same(x,y):
  if type(x)!=type(y):return False
  if isinstance(x,dict):return x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
  if isinstance(x,(list,tuple)):return len(x)==len(y) and all(same(i,j) for i,j in zip(x,y))
  if isinstance(x,float) and x!=y:
   if not (math.isfinite(x) and math.isfinite(y) and abs(x-y)<=1e-12):return False
   differences.append(abs(x-y));return True
  return x==y
 if not same(a,b):raise AssertionError(reason)
check('new15C delivery intact',hashes('execution/15C_source_manifest.json'))
check('09-15 and14C frozen delivery unchanged',all(hashes(f'execution/{x}_source_manifest.json') for x in ['09','10','11','12','12R','13','14','15','14C']))
check('all current software tests pass',all(read('execution/15C_test_result.json')[k]==v for k,v in dict(tests_run=401,failures=0,errors=0,skipped=0,result='PASS').items()))
check('collected raw files match hashes',hashes('runs/transfer_step15C_full/collected_manifest.json'))
assets=read('execution/15C_current_assets.json')['files'];binding=read('configs/transfer/step15_binding.json')['backend_B'];oldplan=read('execution/14C_run_plan_v3.json')
check('current weights and normalization rehashed unchanged',[a['sha256'] for a in assets]==[binding['weight_sha256'],binding['norm_stats_sha256'],oldplan['checkpoint_sha256'],oldplan['normalization_sha256']])
coverage=read('execution/15C_coverage.json');journals=0;records_count=0;cells=[]
for label,folder in coverage['successful_jobs'].items():
 base=R/folder;run=base/'results';s=read(run/'summary.json');eps=s['episodes'];plan=read(base/'15C_run_plan.json')
 check(label+' all planned scenarios reached without exception',len(eps)==len(plan['cases']) and {e['id'] for e in eps}=={c['id'] for c in plan['cases']} and all(e['reason'] is None and e['checks'] and all(e['checks'].values()) for e in eps))
 check(label+' no hidden wrong start or physical command',s['real_BCI_trials']==s['physical_robot_commands']==0 and all(not e['wrong_target_start'] and not e['physical_execution_allowed'] and all(d['goal']==d['intended_goal_on_dispatch'] for d in e['dispatches']) for e in eps))
 for transport in sorted({e['transport'] for e in eps}):
  rows=[e for e in eps if e['transport']==transport];cells.append((label,transport))
  check(label+transport+' ten common scenarios',len(rows)==10 and len({e['base_case'] for e in rows})==10)
  check(label+transport+' actual codec path recorded',all((e['packet_events']>0) if transport=='packet' else (e['packet_events']==0) for e in rows))
  b={e['base_case']:e for e in rows}
  check(label+transport+' stale confirmation and STOP protected',b['late_confirmation']['actual_sim_steps']==0 and b['STOP_during_execution']['final_state']=='CANCELLED' and b['STOP_during_execution']['actual_sim_steps']==8)
  check(label+transport+' active goal revision stops and reconfirms',all(b['goal_change_recovery']['checks'][k] for k in ['change_stops_and_drains_old','old_authorization_rejected','recovery_new_confirmation']) and len(b['goal_change_recovery']['dispatches'])==2)
  check(label+transport+' injected faults not BCI evidence',b['execution_failure_not_channel']['final_state']=='FAILED' and b['inference_failure_not_channel']['final_state']=='CANCELLED' and b['inference_failure_not_channel']['actual_sim_steps']==0 and len(b['inference_failure_not_channel']['dispatches'])==1)
 for e in eps:
  for proof in e['mapping_repair_evidence']:
   assert [x['prompt'] for x in proof['observations']]==['0','1','0','1']
   assert proof['goal_marginals_before']==proof['initial_repaired_goal_marginals']
 js=list(run.glob('*_journal.jsonl'));assert len(js)==sum(e['sessions'] for e in eps)
 with patch.object(replay,'_equal',side_effect=equal),patch.object(SimulationVLAAdapter,'__init__',side_effect=AssertionError('no backend')),patch.object(SimulationVLAAdapter,'submit',side_effect=AssertionError('no submit')),patch.object(SimulationVLAAdapter,'query',side_effect=AssertionError('no query')),patch.object(SimulationVLAAdapter,'stop',side_effect=AssertionError('no stop')):
  for q in js:
   j=EventJournal.from_jsonl(q.read_text());assert replay.replay_session_state(j);journals+=1;records_count+=len(j.records)
 check(label+' all journals independently reconstructed',len(js)>10)
 events=[json.loads(x) for x in (run/'simulator_events.jsonl').read_text().splitlines()];steps=[x for x in events if x['kind']=='actual_simulation_step'];count=Counter(x['label'] for x in steps)
 check(label+' actual steps agree with summary',all(count[e['id']]==e['actual_sim_steps'] for e in eps))
 check(label+' independent holding predicates and finite actions',all(len(e['action'])==7 and all(math.isfinite(v) for v in e['action']) and all(p['held']==(p['grasp_contact'] and p['lift_m']>.03) for p in e['predicates'].values()) for e in steps))
 last={x['label']:x for x in steps}
 check(label+' every claimed grasp matches simulator truth',all(last[e['id']]['predicates'][factory.OBJECTS[factory.GOALS.index(e['dispatched_goal'])][0]]['held'] for e in eps if e['task_success']))
 check(label+' actual rollout videos retained',all((run/(e['id']+'.mp4')).stat().st_size>1000 and (run/(e['id']+'_last.png')).read_bytes().startswith(b'\x89PNG') for e in eps if e['actual_sim_steps']>=8))
 calrecords=read(run/'independent_trajectories.json');cal=[x for x in calrecords if x['split']=='calibration'];held=[x for x in calrecords if x['split']=='calibration_holdout']
 check(label+' calibration heldout test states disjoint',len(cal)==4 and len(held)==2 and {x['init_state_index'] for x in cal}=={0,1} and {x['init_state_index'] for x in held}=={2} and s['calibration_holdout_test_init_indices']==[[0,1],[2],[3]])
 factory.EXECUTOR_ID='Pi05-LIBERO-v4' if label=='pi05' else 'StarVLA-LIBERO-v4'
 check(label+' independent outcomes only fit calibration',factory.empirical_outcomes(calrecords,s['model_checkpoint'])==factory.empirical_outcomes(cal,s['model_checkpoint'])==read(run/'calibrated_outcomes.json'))
 if label=='pi05':
  chunks=[x for x in events if x['kind']=='real_model_chunk'];check('pi05 conversion explicit without double normalization',chunks and all(x['controller_conversion']['rule']=='explicit_controller_bound_clipping_no_extra_unnormalization_or_gripper_flip' for x in chunks) and all(-1<=v<=1 for e in steps for v in e['action']))
 else:check('star calibration exactly reused from14C',sha(run/'independent_trajectories.json')==sha(R/'runs/pilot_robot_14C_full_pipeline/v3/results/independent_trajectories.json'))
 check(label+' owned service stopped',read(base/'orchestration.json')['owned_server_stopped'] and read(base/'orchestration.json')['worker_exit_code']==0)
 job=read('execution/'+('15C_remote_job_v1.json' if label=='pi05' else '15C_star_remote_job_v1.json'))
 check(label+' executed upload exactly bound',sha(base/'sources.zip')==job['archive_sha256'] and all(sha(x['path'])==x['sha256'] for x in job['files']))
 with zipfile.ZipFile(base/'sources.zip') as z:
  check(label+' executed migration namespace equals local',all(z.read(str(q.relative_to(R)).replace('\\','/'))==q.read_bytes() for q in (R/'bci_vla/simulation_transfer_v4').glob('*.py')))
check('three new cells plus14C direct StarVLA form software matrix',set(cells)=={('pi05','direct'),('pi05','packet'),('star','packet')} and read('execution/14C_integrity_result.json')['result']=='PASS')
p=read('execution/progress.json')
check('nonreal completion scoped separately from real migration',p['current_step'] in {'15','16'} and p['steps'][14]['nonreal_migration_completed'] and not coverage['real_2x2_completed'] and not coverage['second_real_BCI_tested'] and not coverage['formal_heldout_transfer_claim'])
check('research gates unchanged',not p['research_gates']['G2']['research_acceptance_passed'] and p['research_gates']['G3']['status']=='未验证')
check('all historical checks revalidated separately',all(read(f'execution/{v}_integrity_recheck_after_step15C.json')['result']=='PASS' for v in ['12R','13','14','15','14C']))
check('TXT BOM CRLF retained',all(q.read_bytes().startswith(b'\xef\xbb\xbf') and b'\n' not in q.read_bytes().replace(b'\r\n',b'') for q in R.rglob('*.txt')))
v=dict(result='PASS',count=len(checks),checks=checks,software_tests=401,independent_journals=journals,journal_records=records_count,max_float_roundoff=max(differences,default=0),nonreal_matrix_completed=True,real_2x2_completed=False,G2_passed=False,G3_passed=False)
destination='execution/15C_integrity_recheck_after_step16.json' if p['current_step']=='16' else 'execution/15C_integrity_result.json'
(R/destination).write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v,indent=2))
