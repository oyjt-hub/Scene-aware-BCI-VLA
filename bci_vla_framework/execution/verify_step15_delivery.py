"""Read-only verification of core freezing, replay equivalence and VLA simulation."""
from pathlib import Path
import hashlib,json,math,sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from bci_vla.adapters.vla import FakeVLAAdapter
from bci_vla.contracts.session import replay_session_state
from bci_vla.monitoring.journal import EventJournal
from bci_vla.protocols import thaw
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def hashes(p):return all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in read(p)['files'])
checks=[]
def check(name,ok):
    if not ok:raise AssertionError(name)
    checks.append(name)
coverage=read('execution/15_coverage.json');tests=read('execution/15_test_result.json');freeze=read('execution/15_pre_run_freeze.json')
p=read('execution/progress.json');config=read('configs/transfer/step15_binding.json')
check('step15 source and artifacts frozen',hashes('execution/15_source_manifest.json'))
check('all frozen core parameters and old simulator gate unchanged',len(freeze['source_files'])==25 and all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in freeze['source_files']))
check('prior09-14 evidence retained',all(hashes(f'execution/{v}_source_manifest.json') for v in ['09','10','11','12','12R','13','14']))
check('385 software regressions pass',tests['tests_run']==385 and tests['result']=='PASS' and tests['failures']==tests['errors']==tests['skipped']==0)
run=ROOT/'runs/transfer_step15_replay'
a=read(run/'direct_trace.json');b=read(run/'packet_trace.json')
check('alternate replay transport preserves all core checkpoints',a['checkpoints']==b['checkpoints'] and len(a['checkpoints'])==5)
journals=[]
for label in ['direct','packet']:
    journal=EventJournal.from_jsonl((run/(label+'_journal.jsonl')).read_text(encoding='utf-8'));journals.append(journal)
    with patch.object(FakeVLAAdapter,'__init__',side_effect=AssertionError('no backend')), \
         patch.object(FakeVLAAdapter,'submit',side_effect=AssertionError('no submit')), \
         patch.object(FakeVLAAdapter,'query',side_effect=AssertionError('no query')), \
         patch.object(FakeVLAAdapter,'stop',side_effect=AssertionError('no stop')):
        state=replay_session_state(journal)
    check('independent checkpoint replay without executor '+label,state==a['checkpoints'][-1])
semantic=lambda j:[{k:thaw(r[k]) for k in ['at','kind','payload']} for r in j.records]
check('journal payloads equal apart from instance identity',semantic(journals[0])==semantic(journals[1]))
check('no second physiological BCI inferred',config['software_transport']['not_a_second_physiological_BCI'] and config['second_real_BCI']['backend_id'] is None and config['second_real_BCI']['supports_known_target_probe'] is None and not coverage['real_second_BCI_tested'])
base=ROOT/'runs/transfer_step15_pi05_final';s=read(base/'v4/results/summary.json');episodes=s['episodes']
check('pi05 actual model simulator scope retained',s['scope']=='pi05_LIBERO_simulation_scripted_interaction_not_real_BCI' and s['real_BCI_trials']==s['physical_robot_commands']==0)
check('both pi05 development tasks completed',len(episodes)==3 and episodes[0]['task_success'] and episodes[2]['task_success'] and s['nominal_task_successes']==2)
check('fault episode retained as injected stop not task success',episodes[1]['termination']=='injected_stop' and not episodes[1]['task_success'] and episodes[1]['applied_actions']==12)
check('all guard checks and exceptions accounted',s['all_software_guards_passed'] and s['all_episodes_without_exception'] and all(e['error'] is None and all(e['checks'].values()) for e in episodes))
check('same planned suite seed tasks initial state horizon',all(e['suite']==freeze['common_suite'] and e['seed']==freeze['seed'] and e['init_state_index']==freeze['init_state_index'] and e['applied_actions']<=freeze['max_steps'] for e in episodes) and [episodes[0]['task_id'],episodes[2]['task_id']]==freeze['task_ids'])
events=[json.loads(line) for line in (base/'v4/results/events.jsonl').read_text().splitlines()]
counts={};calls={};current=None;inactive=False;terminal=[];clipped=0
for e in events:
    if e['kind']=='episode_start':current=e['episode'];counts[current]=calls[current]=0;inactive=False
    elif e['kind'] in {'stop_latched','task_change_no_old_action'}:inactive=True;assert e['snapshot']['queued_actions']==0
    elif e['kind']=='probe_pause':assert e['snapshot']['confirmed'] is None and e['snapshot']['queued_actions']==0
    elif e['kind']=='terminal_queue_cleared':
        inactive=True;terminal.append(e);assert e['snapshot']['queued_actions']==0 and e['snapshot']['confirmed'] is None
    elif e['kind']=='simulation_step':
        assert not inactive and e['episode']==current and e['step']==counts[current]
        assert len(e['action'])==7 and all(math.isfinite(v) and -1<=v<=1 for v in e['action'])
        counts[current]+=1
    elif e['kind']=='model_chunk':
        assert not inactive and e['episode']==current and e['shape'][0]==1 and e['shape'][2]==7 and e['shape'][1]>=8
        conversion=e['controller_conversion'];assert conversion['lower']==[-1.]*7 and conversion['upper']==[1.]*7
        clipped+=conversion['clipped_values'];calls[current]+=1
check('recorded pi05 actions bounded and no steps after invalidation',all(counts[e['episode']]==e['applied_actions'] and calls[e['episode']]==e['inference_calls'] for e in episodes))
check('terminal queues empty in events and final snapshots',len(terminal)==3 and all(e['gate_final']['queued_actions']==0 and e['gate_final']['confirmed'] is None for e in episodes))
check('explicit clipping counts retained',clipped==coverage['controller_clipped_values']==read('execution/15_backend_observations.json')['pi05_controller_clipped_values'] and clipped>0)
assets=read('execution/15_pi05_assets.json')
check('controller clipping matches source reference','np.clip(action, self.input_min, self.input_max)' in assets['controller_conversion_source'])
check('pi05 weight statistics and output semantics bound',s['checkpoint']==assets['files'][0]['path']==config['backend_B']['checkpoint'] and s['stats_sha256']==assets['files'][1]['sha256']==config['backend_B']['norm_stats_sha256'] and config['backend_B']['extra_gripper_flip'] is False and config['backend_B']['extra_unnormalization'] is False)
check('downloaded owned job artifacts intact',hashes(base/'collected_manifest.json'))
executed=read(base/'collected_manifest.json')['executed_sources']
check('executed worker adapter and gate match local copies',len(executed)==4 and all(e['matches_local'] and hashlib.sha256(Path(e['local_path']).read_bytes()).hexdigest()==e['sha256'] for e in executed))
orchestration=read(base/'v4/orchestration.json')
check('owned model service stopped with no original environment edits',orchestration['worker_exit_code']==0 and orchestration['owned_server_stopped'] and not orchestration['original_environment_modified'])
check('old failures and terminal cleanup regression retained',read(base/'v1/orchestration.json')['error'].startswith('RuntimeError') and all(e['applied_actions']==0 for e in read(base/'v2/results/episodes.json')) and read(base/'v3/results/episodes.json')[2]['gate_final']['queued_actions']==2)
check('videos and endpoint frames retained',all((base/'v4/results'/e['video']).stat().st_size>1000 and (base/'v4/results'/(e['episode']+'_last.png')).read_bytes().startswith(b'\x89PNG') for e in episodes))
check('adaptation costs recorded without invented human hours',read('execution/15_adaptation_audit.json')['core_and_threshold_changes']==0 and read('execution/15_adaptation_audit.json')['human_engineering_hours'] is None)
check('missing calibration and holdout not upgraded',not coverage['real_2x2_completed'] and not coverage['full_framework_migration_completed'] and not coverage['independent_consequence_calibration_completed'] and not coverage['formal_held_out_transfer_tested'])
check('development observations not performance claims',not s['formal_success_rate_claim'] and not read('execution/15_backend_observations.json')['paired_statistical_or_backend_superiority_claim'])
check('step15 incomplete and research gates unchanged',(p['current_step']=='15' or (p['current_step']=='14' and p['steps'][13].get('nonreal_closed_loop_completed'))) and p['steps'][14]['status']=='进行中' and p['steps'][13]['status']=='进行中' and p['research_gates']['G2']['research_acceptance_passed'] is False and p['research_gates']['G3']['status']=='未验证')
check('old12R13 and14 separately rechecked',all(read(f'execution/{v}_integrity_recheck_after_step15.json')['result']=='PASS' for v in ['12R','13','14']))
check('all TXT preserve BOM and CRLF',all(q.read_bytes().startswith(b'\xef\xbb\xbf') and b'\n' not in q.read_bytes().replace(b'\r\n',b'') for q in ROOT.rglob('*.txt')))
value={'date':'2026-10-16','result':'PASS','scope':'step15_replay_and_second_VLA_backend_not_full_migration','count':len(checks),'checks':checks,
    'software_tests':385,'frozen_core_sources_changed':0,'pi05_nominal_tasks_completed':2,'stop_injection_episodes':1,
    'real_second_BCI_tested':False,'full_2x2_migration_completed':False,'real_BCI_trials':0,'physical_robot_commands':0}
destination='execution/15_integrity_recheck_after_step14C.json' if p['steps'][13].get('nonreal_closed_loop_completed') else 'execution/15_integrity_result.json'
if p['steps'][14].get('nonreal_migration_completed'):
    destination='execution/15_integrity_recheck_after_step15C.json'
(ROOT/destination).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');print(json.dumps(value,indent=2))
