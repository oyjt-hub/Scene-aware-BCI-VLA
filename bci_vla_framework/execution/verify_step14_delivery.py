"""Verify saved LIBERO evidence and frozen sources without model or device I/O."""
from pathlib import Path
import hashlib,json,math,sys
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
checks=[]
def check(name,ok):
    if not ok:raise AssertionError(name)
    checks.append(name)
def hashes(p):return all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in read(p)['files'])
run=ROOT/'runs/pilot_robot_14_libero_simulation_v3/v3/results'
s=read(run/'summary.json');episodes=read(run/'episodes.json');tests=read('execution/14_test_result.json')
coverage=read('execution/14_coverage.json');p=read('execution/progress.json');config=read('configs/execution/libero_simulation_v1.json')
check('step14 source and evidence hashes match',hashes('execution/14_source_manifest.json'))
check('earlier frozen source manifests unchanged',all(hashes(f'execution/{v}_source_manifest.json') for v in ['09','10','11','12','12R','13']))
check('full software regression passes',tests['tests_run']==376 and tests['result']=='PASS' and tests['failures']==tests['errors']==tests['skipped']==0)
check('actual model simulation distinguished from real BCI',s['scope']=='real_VLA_LIBERO_simulation_scripted_interaction_not_real_BCI' and s['real_BCI_trials']==s['physical_robot_commands']==0)
check('three episodes include intentional unsuccessful stop',len(episodes)==3 and s['episodes']==episodes and episodes[1]['termination']=='injected_stop' and episodes[1]['task_success'] is False)
check('two nominal task predicates satisfied',all(e['task_success'] and e['termination']=='task_success' for e in [episodes[0],episodes[2]]) and s['nominal_task_successes']==s['nominal_episodes']==2)
check('nominal steps not hidden',episodes[0]['applied_actions']==117 and episodes[2]['applied_actions']==80)
check('fixed development initial conditions explicit',all(e['suite']=='libero_goal' and e['seed']==7 and e['init_state_index']==0 for e in episodes))
check('software guard checks all reported',s['all_software_guards_passed'] and all(all(e['checks'].values()) for e in episodes))
check('probe and stop clear pending actions',episodes[1]['checks']['probe_flush'] and episodes[1]['checks']['probe_blocks'] and episodes[1]['checks']['stop_blocks'] and episodes[1]['checks']['stop_flush'])
check('task change requires new authorization',episodes[0]['checks']['changed_task_flush'] and episodes[0]['checks']['old_context_blocked'] and episodes[0]['checks']['new_context_requires_confirmation'])
check('no episode exceptions suppressed',s['all_episodes_without_exception'] and all(e['error'] is None for e in episodes))
events=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
current=None;stopped=changed=False;counts={};chunks={};clocks=[]
for e in events:
    clocks.append(e['elapsed_s'])
    if e['kind']=='episode_start':
        current=e['episode'];counts[current]=0;chunks[current]=0;stopped=changed=False
        assert e['input_source']=='scripted_simulation_fixture'
    elif e['kind']=='stop_latched':stopped=True;assert e['snapshot']['queued_actions']==0
    elif e['kind']=='task_change_no_old_action':changed=True;assert e['snapshot']['confirmed'] is None
    elif e['kind']=='probe_pause':assert e['snapshot']['queued_actions']==0 and e['snapshot']['confirmed'] is None
    elif e['kind']=='model_chunk':
        assert not stopped and not changed and e['episode']==current and e['shape']==[1,8,7]
        assert e['image_shapes']==[[256,256,3],[256,256,3]] and e['latency_s']>0
        chunks[current]+=1
    elif e['kind']=='simulation_step':
        assert not stopped and not changed and e['episode']==current and e['step']==counts[current]
        assert len(e['action'])==7 and all(math.isfinite(v) and -1<=v<=1 for v in e['action'])
        counts[current]+=1
check('recorded model output and simulator inputs audited',all(counts[e['episode']]==e['applied_actions'] and chunks[e['episode']]==e['inference_calls'] for e in episodes))
check('trace times monotonic',clocks==sorted(clocks))
check('independent trace audit agrees',read('execution/14_trace_audit.json')['simulation_steps']==counts and not read('execution/14_trace_audit.json')['full_interaction_session_replay'])
assets=read('execution/14_libero_assets.json')
check('checkpoint identity and statistics bound',config['checkpoint']==s['checkpoint']==assets['files'][0]['path'] and config['checkpoint_sha256']==assets['files'][0]['sha256'] and s['stats_sha256']==assets['files'][2]['sha256'])
check('simulation image and action semantics not physical Piper',config['simulation_action_dimension']==7 and config['physical_execution_allowed'] is False and config['real_BCI_enabled'] is False)
check('gripper source reference records corrected direction','1.0 - 2.0' in read('execution/14_executed_sources.json')['gripper_reference']['function'] and '1 - 2' in config['gripper_conversion'])
executed=read('execution/14_executed_sources.json')['deployed_sources']
check('executed remote sources match saved local implementation',len(executed)==5 and all(r['matches_local'] and hashlib.sha256(Path(r['local']).read_bytes()).hexdigest()==r['sha256'] for r in executed))
orchestration=read('runs/pilot_robot_14_libero_simulation_v3/v3/orchestration.json')
check('owned server cleaned up after worker success',orchestration['worker_exit_code']==0 and orchestration['owned_server_stopped'] is True and 'error' not in orchestration)
check('downloaded artifacts hash match',hashes('runs/pilot_robot_14_libero_simulation_v3/collected_manifest.json'))
check('all three videos and rendered endpoints retained',all((run/e['video']).stat().st_size>1000 and (run/(e['episode']+'_last.png')).read_bytes().startswith(b'\x89PNG') for e in episodes))
check('development failures retained separately',read('runs/pilot_robot_14_libero_simulation/v1/orchestration.json')['error'].startswith('RuntimeError') and read('runs/pilot_robot_14_libero_simulation/v2/results/summary.json')['nominal_task_successes']==1)
check('no paper performance or formal success rate claim',not s['formal_success_rate_claim'] and not coverage['formal_success_rate_claim'] and not s['original_framework_ontology_integrated'])
check('full framework and real step14 remain unfinished',coverage['full_framework_simulation_completed'] is False and coverage['real_step14_completed'] is False and p['current_step'] in {'14','15'} and p['steps'][13]['status']=='进行中')
check('research gates not upgraded',p['research_gates']['G2']['research_acceptance_passed'] is False and p['research_gates']['G3']['status']=='未验证')
check('old reports preserved and separately rechecked',read('execution/13_integrity_result.json')['result']==read('execution/13_integrity_recheck_after_step14.json')['result']==read('execution/12R_integrity_result.json')['result']==read('execution/12R_integrity_recheck_after_step14.json')['result']=='PASS')
check('all TXT retain BOM CRLF',all(q.read_bytes().startswith(b'\xef\xbb\xbf') and b'\n' not in q.read_bytes().replace(b'\r\n',b'') for q in ROOT.rglob('*.txt')))
v={'date':'2026-10-16','result':'PASS','scope':'LIBERO_backend_and_guard_delivery_not_full_framework','count':len(checks),'checks':checks,
    'software_tests':376,'simulation_episodes':3,'nominal_task_successes':2,'fault_stop_episodes':1,
    'real_BCI_trials':0,'physical_robot_commands':0,'full_framework_simulation_completed':False,'real_step14_completed':False}
destination='execution/14_integrity_recheck_after_step15.json' if p['current_step']=='15' else 'execution/14_integrity_result.json'
if p['steps'][13].get('nonreal_closed_loop_completed'):
    destination='execution/14_integrity_recheck_after_step14C.json'
if p['steps'][14].get('nonreal_migration_completed'):
    destination='execution/14_integrity_recheck_after_step15C.json'
(ROOT/destination).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');print(json.dumps(v,indent=2))
