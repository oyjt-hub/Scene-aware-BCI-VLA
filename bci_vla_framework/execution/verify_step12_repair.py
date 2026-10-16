"""Audit versioned repair and preserve the unresolved research gate."""
from pathlib import Path
import gzip
import hashlib
import json
import re
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from bci_vla.protocols import thaw
from bci_vla.monitoring.journal import EventJournal
from bci_vla.benchmark.runner import summarize

RUN=ROOT/'runs/offline_step12_repair_v2'
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(row) for row in p.read_text(encoding='utf-8').splitlines()]
def hashes(p):return all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in read(p)['files'])
checks=[]
def check(name,ok):
    if not ok:raise AssertionError(name)
    checks.append(name)

summary=read('runs/offline_step12_repair_v2/summary.json');audit=read('runs/offline_step12_repair_v2/oracle_audit.json')
plan=read('runs/offline_step12_repair_v2/evaluation_private/run_plan.json');progress=read('execution/progress.json')
tests=read('execution/12R_test_result.json')
public=lines(RUN/'online/episodes.jsonl');private=lines(RUN/'evaluation_private/worlds.jsonl');joined=lines(RUN/'evaluation_joined.jsonl')
old=lines(ROOT/'runs/offline_step12_acceptance/evaluation_joined.jsonl')
pub={r['episode_id']:r for r in public};priv={r['episode_id']:r for r in private}
check('330 software checks pass and frozen311 unchanged',tests['result']=='PASS' and tests['tests_run']==330 and tests['failures']==tests['errors']==tests['skipped']==0 and read('execution/12_test_result.json')['tests_run']==311)
check('1440 all retained joined and opaque ids',len(public)==len(private)==len(joined)==len(pub)==summary['episodes']==1440 and set(pub)==set(priv)=={r['episode_id'] for r in joined} and all(re.fullmatch(r'episode-[0-9a-f]{32}',key) for key in pub))
check('original12 regression and additional12 development declared',plan['regression_seeds']==list(range(12)) and plan['additional_development_seeds']==list(range(12,24)) and plan['model_fitting_or_selection'] is False)
check('10 conditions24 seeds6 methods complete',len(plan['cases'])==10 and set(plan['methods'])=={'B4','B5','B7','B9','M','M2'} and all(sum(r['case_id']==case['id'] and r['method']==method for r in joined)==24 for case in plan['cases'] for method in plan['methods']))
check('versioned external proposer in every output',all(r['planner_ref']['version']=='authorization-aware-finite/2.0' and r['planner_ref']['role']=='external_advisory_proposer_not_execution_authority' for r in public))
check('all methods same unchanged threshold age and return',all(r['planner_ref']['gate']=={'version':'authorization-aware-finite/2.0','minimum_goal_probability':.8,'max_confirmation_age_s':2.,'return_s':.01} for r in public))
check('29 candidate predictions match actual authorizer',len(audit['gate_cases'])==29 and audit['all_actual_gate_predictions_match'] and all(r['candidate_authorizes']==r['actual_authorizes'] and abs(r['candidate_continuation_loss']-r['independent_O1_continuation_loss'])<1e-12 and r['replay_matches'] for r in audit['gate_cases']))
low=next(r for r in audit['gate_cases'] if r['prior']==[.4,0,.6,0] and r['return_s']==.01 and r['fixed_transition'])
check('original counterexample fixed as pause not execution',abs(low['predicted_goal_probability']-.32/.41)<1e-12 and not low['actual_authorizes'] and low['candidate_continuation_loss']==low['independent_O1_continuation_loss']==1 and low['actual_backend_submissions']==0)
check('high valid and delayed expired cases both present',any(r['actual_authorizes'] for r in audit['gate_cases']) and any(r['return_s']>2 and not r['actual_authorizes'] for r in audit['gate_cases']))
check('80 corrected O1 same-horizon queries match',len(audit['same_horizon_rows'])==80 and audit['max_same_horizon_regret']<=1e-9 and all(abs(r['planning_value_gap'])<=1e-9 for r in audit['same_horizon_rows']))
check('limited4state model not complete runtime optimality',all(r['state_count']==4 for r in audit['same_horizon_rows']) and audit['full_runtime_optimality_claim'] is False and audit['unknown_future_scene_changes_modeled'] is False)
check('unexpected actual authorization blocks eliminated in sampled run',summary['authorization_wrapper_blocked_episodes']==0 and all(not r['termination'].startswith('authorization_or_dispatch_blocked:') for r in public))
check('postcompute accepted branch checked before requesting confirmation',all(any(d.get('post_compute_authorization_prediction',{}).get('allowed') is True for d in r['decisions']) for r in public if r['commanded_goal'] is not None))
check('all8 mechanism and10 old fault checks repeated',len(read('runs/offline_step12_repair_v2/mechanism_checks.json')['checks'])==8 and all(read('runs/offline_step12_repair_v2/mechanism_checks.json')['checks'].values()) and read('runs/offline_step12_repair_v2/fault_acceptance.json')['all_passed'])
lookup={(r['case_id'],r['seed'],r['method']):r for r in joined}
check('original paired outcome counts not inflated by repair',all(all(lookup[(r['case_id'],r['seed'],r['method'])][k]==r[k] for k in ['faithful_virtual_completion','wrong_start','unstarted']) for r in old))
check('M and same-probe B9 still equal in240 pairs',len(summary['M_vs_B9_paired'])==240 and all(r['same_selected_actions'] and r['same_completion_and_wrong_start'] for r in summary['M_vs_B9_paired']))
check('per-condition metrics reproduce all episodes',summarize(joined)==summary['per_case_method'])
check('failed and unstarted full times retained',all(r['elapsed_s']>=2 and r['unresolved']==(not r['faithful_virtual_completion']) for r in joined) and all(abs(r['elapsed_s']-r['timeline_union_s'])<1e-4 for r in public))
for r in joined:
    p=pub[r['episode_id']]
    assert r['wrong_start']==(p['commanded_goal'] is not None and p['commanded_goal']!=r['truth_goal_at_dispatch_or_end'])
    assert r['faithful_virtual_completion']==(p['final_state']=='SUCCEEDED' and p['commanded_goal']==r['truth_goal_at_dispatch_or_end'])
check('independent user goal truth not replaced by contract success',any(r['wrong_start'] and pub[r['episode_id']]['final_state']=='SUCCEEDED' for r in joined))
forbidden={'seed','case','case_id','true_goal','true_channel','goal_change','channel_change','private_truth','ground_truth'}
def public_only(v):
    if isinstance(v,dict):return not forbidden.intersection(v) and all(public_only(child) for child in v.values())
    if isinstance(v,list):return all(public_only(child) for child in v)
    return True
check('no private truth fields in online outputs',all(public_only(r) for r in public))
journal_ids=set();record_count=0
with gzip.open(RUN/'online/journals.jsonl.gz','rt',encoding='utf-8') as source:
    for line in source:
        v=json.loads(line);identifier=v['episode_id'];assert identifier not in journal_ids;journal_ids.add(identifier)
        journal=EventJournal.from_jsonl(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in v['records']))
        assert journal.session_id==identifier and journal.records[-1]['record_hash']==pub[identifier]['journal_last_hash']
        assert len(journal.records)==pub[identifier]['journal_records']
        assert journal.records[-1]['payload']['state']['state']==pub[identifier]['final_state']
        assert public_only(thaw(journal.records))
        assert all(r['payload']['state']['execution_allowed'] is False and r['payload']['state']['physical_stop_verified'] is False for r in journal.records)
        record_count+=len(journal.records)
check('all1440 journal chains terminal anchors physical flags valid',journal_ids==set(pub))
check('all producer replays verified each checkpoint without backend',summary['all_session_journals_replayed'] and all(r['replay_matches_each_checkpoint'] for r in public))
check('run input source freeze intact',all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in plan['source_files']))
check('run artifacts intact',all(hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in read('runs/offline_step12_repair_v2/artifacts_manifest.json')['files']))
check('new repair sources reports and results frozen',hashes('execution/12R_source_manifest.json'))
check('original09-12 source and results retained',all(hashes(f'execution/{step}_source_manifest.json') for step in ['09','10','11','12']))
check('original12 delivery recheck passed separately',read('execution/12_integrity_recheck_after_repair.json')['result']=='PASS')
check('technical defect resolved separately from research gate',progress['research_gates']['G2']['engineering_revalidation']=='PASS' and progress['research_gates']['G2']['status']=='未通过' and progress['research_gates']['G2']['research_acceptance_passed'] is False)
step13_nonreal=progress['current_step'] in {'13','14','15'} and progress['steps'][12].get('nonreal_tests_completed') is True and progress['steps'][12].get('real_pilot_completed') is False
check('research gate retained while later software preparation is separately scoped',(progress['current_step']=='12' and progress['current_step_status']=='待研究效用复验' and progress['steps'][12]['status']=='未开始') or step13_nonreal)
check('no automatic research win or hardware claim',summary['G2_verdict']=='TECHNICAL_REPAIR_PASSED_RESEARCH_BENEFIT_UNPROVEN' and not summary['strategy_superiority_claim'] and not summary['real_success_rate_claim'] and summary['hardware_calls']==summary['cloud_calls']==0)
check('all TXT BOM CRLF',all(q.read_bytes().startswith(b'\xef\xbb\xbf') and b'\n' not in q.read_bytes().replace(b'\r\n',b'') for q in ROOT.rglob('*.txt')))
value={'date':'2026-10-16','result':'PASS','scope':'versioned_authorization_repair_delivery','count':len(checks),
    'checks':checks,'episodes':1440,'journal_records':record_count,'engineering_revalidation':'PASS','G2_passed':False,
    'hardware_calls':0,'cloud_calls':0,'step13_real_pilot_started':False}
destination=('execution/12R_integrity_recheck_after_step'+progress['current_step']+'.json') if step13_nonreal else 'execution/12R_integrity_result.json'
if progress['steps'][13].get('nonreal_closed_loop_completed'):
    destination='execution/12R_integrity_recheck_after_step14C.json'
if progress['steps'][14].get('nonreal_migration_completed'):
    destination='execution/12R_integrity_recheck_after_step15C.json'
(ROOT/destination).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(value,ensure_ascii=True,indent=2))
