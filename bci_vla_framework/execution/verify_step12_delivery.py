"""Check complete negative G2 evidence; passing this audit does not pass G2."""
from pathlib import Path
import gzip
import hashlib
import json
import re
import sys
from statistics import mean

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from bci_vla.protocols import thaw
from bci_vla.monitoring.journal import EventJournal
from bci_vla.benchmark.runner import summarize

RUN=ROOT/'runs/offline_step12_acceptance'
def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))
def lines(path):return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
def hashes(path):return all(Path(r['path']).is_file() and hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in read(path)['files'])
checks=[]
def check(name,ok):
    if not ok:raise AssertionError(name)
    checks.append(name)

progress=read('execution/progress.json');t=read('execution/12_test_result.json')
summary=read('runs/offline_step12_acceptance/summary.json');plan=read('runs/offline_step12_acceptance/evaluation_private/run_plan.json')
mechanisms=read('runs/offline_step12_acceptance/mechanism_checks.json');oracle=read('runs/offline_step12_acceptance/oracle_audit.json')
faults=read('runs/offline_step12_acceptance/fault_acceptance.json');gap=read('execution/12_runtime_alignment.json')
public=lines(RUN/'online/episodes.jsonl');private=lines(RUN/'evaluation_private/worlds.jsonl');joined=lines(RUN/'evaluation_joined.jsonl')
public_by_id={row['episode_id']:row for row in public};private_by_id={row['episode_id']:row for row in private}
check('01-12 execution complete not G2 success',all(s['status']=='已完成' for s in progress['steps'][:12]) and progress['steps'][11]['acceptance_result']=='G2未通过')
check('stopped at12 research revalidation and13 blocked',progress['current_step']=='12' and progress['current_step_status'] in {'待修复与复验','待研究效用复验'} and progress['steps'][12]['status']=='未开始' and progress['steps'][12]['blocked_by']==['G2'])
check('G2 failed G3 G4 unstarted',progress['research_gates']['G2']['status']=='未通过' and progress['research_gates']['G2']['research_acceptance_passed'] is False and all(progress['research_gates'][g]['status']=='未开始' for g in ['G3','G4']))
check('311 current tests pass with291 frozen prior',t['tests_run']==311 and t['result']=='PASS' and t['failures']==t['errors']==t['skipped']==0 and read('execution/11_test_result.json')['tests_run']==291)
check('720 episodes all kept unique and fully joined',len(public)==len(private)==len(joined)==summary['episodes']==720 and len(public_by_id)==720 and set(public_by_id)==set(private_by_id)=={r['episode_id'] for r in joined})
check('opaque ids do not encode case or seed',all(re.fullmatch(r'episode-[0-9a-f]{32}',row['episode_id']) for row in public))
check('10x12x6 common matrix complete',len(plan['cases'])==10 and len(plan['paired_seeds'])==12 and set(plan['methods'])=={'B4','B5','B7','B9','M','M2'} and all(sum(r['case_id']==case['id'] and r['method']==method for r in joined)==12 for case in plan['cases'] for method in plan['methods']))
check('required cells and negative controls present',{'C00','C10','C01','C11_simultaneous','stable_sporadic_error','wrong_strong_prior','probe_noncompliance','uninformative_probe'}<={case['id'] for case in plan['cases']})
check('all changes before common stable window explicit','ordering_not_observed' in private[0]['change_exposure'] and all(draw['window_start_s']>=1 for row in private for draw in row['draws']))

forbidden={'seed','case_id','case','true_goal','true_channel','private_truth','ground_truth','goal_change','channel_change'}
def public_only(value):
    if isinstance(value,dict):return not forbidden.intersection(value) and all(public_only(v) for v in value.values())
    if isinstance(value,list):return all(public_only(v) for v in value)
    return True
check('public outputs exclude private truth fields',all(public_only(row) for row in public))
check('no online world or private schedule objects',all('public_input' in d and not forbidden.intersection(d['public_input']) for row in public for d in row['decisions']))
check('all mechanism boundaries including uninformative control pass',len(mechanisms['checks'])==8 and all(mechanisms['checks'].values()))
check('all10 required injected faults pass and replay',len(faults['cases'])==10 and faults['all_passed'] and all(r['passed'] and r['replay_matches'] and not r['physical_stop_verified'] for r in faults['cases']))
check('80 finite same-horizon oracle checks and measured compute',len(oracle['rows'])==80 and oracle['max_same_horizon_regret']<=1e-9 and all(r['oracle_compute_ns']>0 and r['method_compute_ns']>0 for r in oracle['rows']))
check('limited oracle scope not complete POMDP','internal_finite_planning_model_only' in oracle['scope'] and oracle['optimal_pomdp_claim'] is False and {r['hidden_states'] for r in oracle['rows']}=={4})
check('actual runtime counterexample gate remains enforced',abs(gap['posterior_after_noisy_confirm']-.32/.41)<1e-12 and gap['posterior_after_noisy_confirm']<gap['authorization_threshold'] and gap['backend_submissions']==0 and gap['gate_weakened'] is False)

goals=read('configs/belief/simulation_binary_filter.json')['vocabulary']['goals'];known=[g['goal_id'] for g in goals if g['kind']=='task']
for row in joined:
    pub=public_by_id[row['episode_id']];world=private_by_id[row['episode_id']];case=world['case']
    at=pub['submission_time_s'] if pub['submission_time_s'] is not None else pub['elapsed_s']
    truth=known[int(case['goal_change'] and at>=case.get('goal_at',.5))]
    assert truth==row['truth_goal_at_dispatch_or_end']
    assert row['wrong_start']==(pub['commanded_goal'] is not None and pub['commanded_goal']!=truth)
    assert row['faithful_virtual_completion']==(pub['commanded_goal']==truth and pub['final_state']=='SUCCEEDED')
    assert row['unresolved']==(not row['faithful_virtual_completion'])
check('truth joined after outputs and contract success not substituted',any(row['wrong_start'] and public_by_id[row['episode_id']]['final_state']=='SUCCEEDED' for row in joined))
check('full failed and unstarted denominators retained',all(row['elapsed_s']>=2 and row['unresolved']==(not row['faithful_virtual_completion']) for row in joined) and any(row['unstarted'] for row in joined))
check('per-case metrics recompute from all joined rows',summarize(joined)==summary['per_case_method'])
check('full time includes nonuser phases and no double counting',all(abs(row['elapsed_s']-row['timeline_union_s'])<1e-4 and row['elapsed_s']+1e-9>=row['user_busy_s']>=row['probe_s']>=0 for row in public))
blocked=sum(row['termination'].startswith('authorization_or_dispatch_blocked:') for row in public)
check('authorization blocked outcomes retained',blocked==summary['authorization_wrapper_blocked_episodes'] and blocked>0)
check('M and same probe B9 pairing equal no superiority claim',len(summary['M_vs_B9_paired'])==120 and all(row['same_selected_actions'] and row['same_completion_and_wrong_start'] for row in summary['M_vs_B9_paired']) and summary['strategy_superiority_claim'] is False)
check('negative research verdict retained',summary['G2_verdict']=='NOT_PASSED_PENDING_MODEL_ALIGNMENT_AND_CLAIM_NARROWING' and summary['next_step13_allowed'] is False)

journal_ids=set();total_records=0
with gzip.open(RUN/'online/journals.jsonl.gz','rt',encoding='utf-8') as source:
    for line in source:
        v=json.loads(line);identifier=v['episode_id'];assert identifier not in journal_ids
        journal_ids.add(identifier);pub=public_by_id[identifier]
        journal=EventJournal.from_jsonl(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in v['records']))
        assert journal.session_id==identifier and len(journal.records)==pub['journal_records']
        assert journal.records[-1]['record_hash']==pub['journal_last_hash']
        last=thaw(journal.records[-1]['payload']['state']);assert last['state']==pub['final_state']
        assert all(r['payload']['state']['execution_allowed'] is False and r['payload']['state']['physical_stop_verified'] is False for r in journal.records)
        assert public_only(thaw(journal.records))
        total_records+=len(journal.records)
check('all720 journal chains session ids terminal anchors and physical flags valid',journal_ids==set(public_by_id))
check('producer replayed every episode no backend',summary['all_session_journals_replayed'] is True and all(r['replay_matches_each_checkpoint'] for r in public))
check('source freeze survived full run',all(hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'] for row in plan['source_files']))
manifest=read('runs/offline_step12_acceptance/artifacts_manifest.json')
check('run output hashes intact',all(hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256'] for row in manifest['files']))
check('12 source outputs and docs frozen',hashes('execution/12_source_manifest.json'))
check('11 and10 historic source outputs retained',hashes('execution/11_source_manifest.json') and hashes('execution/10_source_manifest.json'))
check('11 historical delivery recheck passes',read('execution/11_integrity_recheck_after_step12.json')['result']=='PASS')
check('12.1-12.8 mapped',all(f'12.{i}' in (ROOT/'execution/12_acceptance.txt').read_text(encoding='utf-8-sig') for i in range(1,9)))
check('no real validation or hardware cloud claim',summary['hardware_calls']==summary['cloud_calls']==0 and summary['real_success_rate_claim'] is False and progress['steps'][11]['real_validation']['deployment_allowed'] is False)
check('TXT BOM CRLF throughout',all(path.read_bytes().startswith(b'\xef\xbb\xbf') and b'\n' not in path.read_bytes().replace(b'\r\n',b'') for path in ROOT.rglob('*.txt')))
v={'date':'2026-10-16','scope':'step12_negative_G2_evidence_integrity','result':'PASS','count':len(checks),'checks':checks,
    'episodes':len(public),'journal_records':total_records,'G2_passed':False,'step13_started':False,
    'hardware_calls':0,'cloud_calls':0,'note':'delivery audit PASS does not mean research gate G2 passed'}
(ROOT/'execution/12_integrity_recheck_after_repair.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(v,ensure_ascii=True,indent=2))
