"""Run a bounded L1 synthetic development matrix and keep every episode."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime,timezone
from statistics import mean
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from bci_vla.protocols import thaw
from bci_vla.benchmark.offline import METHODS
from bci_vla.benchmark.synthetic_world import CASES
from bci_vla.benchmark.runner import run_episode,summarize
from bci_vla.benchmark.audits import mechanisms,oracle_audit,faults

def encode(v):return json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def write(path,v):path.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='runs/offline_step12_acceptance')
    parser.add_argument('--seeds',type=int,default=12);args=parser.parse_args()
    assert 1<=args.seeds<=100,'bounded development seed count required'
    folder=ROOT/args.output
    assert not (folder/'summary.json').exists(),'Use a new --output directory to preserve an existing completed run.'
    for name in ['online','evaluation_private']: (folder/name).mkdir(parents=True,exist_ok=True)
    sources=list((ROOT/'bci_vla/benchmark').glob('*.py'))+[Path(__file__).resolve()]
    for manifest in ['09','10','11']:
        m=json.loads((ROOT/f'execution/{manifest}_source_manifest.json').read_text(encoding='utf-8-sig'))
        sources.extend(Path(r['path']) for r in m['files'] if Path(r['path']).suffix in {'.py','.json'} and Path(r['path']).is_file())
    source_hashes=[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(sources))]
    plan={'scope':'L1_synthetic_development_not_formal_validation','created_utc':datetime.now(timezone.utc).isoformat(),
        'methods':METHODS,'cases':CASES,'paired_seeds':list(range(args.seeds)),'max_additional_diagnostics':2,
        'episode_budget_s':20,'policy_compute_budget_ns':5_000_000_000,
        'initial_common_task':'request_at0_window1to2_all_changes_before_window',
        'time':'modeled_setup_acquisition_return_execution_plus_measured_actual_compute; not physical time',
        'pairing':'same hash-based external draws keyed by seed/mode/mode_count/duration; adaptive requests can differ',
        'model_fitting_or_selection':False,'hardware_calls':0,'cloud_calls':0,
        'ordering_limitation':'C11 goal-first/channel-first ordering occurs before the common task window and is not temporally identified',
        'source_files':source_hashes}
    write(folder/'evaluation_private/run_plan.json',plan)
    mechanisms_result=mechanisms();oracle=oracle_audit();fault=faults()
    write(folder/'mechanism_checks.json',mechanisms_result);write(folder/'oracle_audit.json',oracle);write(folder/'fault_acceptance.json',fault)
    private_rows=[];public_rows=[];evaluations=[];counter=0
    with (folder/'online/episodes.jsonl').open('w',encoding='utf-8') as public_file, \
         (folder/'evaluation_private/worlds.jsonl').open('w',encoding='utf-8') as private_file, \
         (folder/'evaluation_joined.jsonl').open('w',encoding='utf-8') as eval_file, \
         gzip.open(folder/'online/journals.jsonl.gz','wt',encoding='utf-8') as journals:
        for case in CASES:
            for seed in range(args.seeds):
                for method in METHODS:
                    counter+=1;identifier='episode-'+uuid4().hex
                    public,evaluation,private,journal=run_episode(case,seed,method,identifier)
                    public_rows.append(public);evaluations.append(evaluation)
                    public_file.write(encode(public)+'\n');private_file.write(encode(private)+'\n');eval_file.write(encode(evaluation)+'\n')
                    journals.write(encode({'episode_id':identifier,'records':[thaw(r) for r in journal.records]})+'\n')
            print(encode({'completed_case':case['id'],'episodes_done':counter}),flush=True)
    groups=summarize(evaluations)
    overall=[]
    for method in METHODS:
        rows=[r for r in evaluations if r['method']==method];n=len(rows)
        overall.append({'method':method,'episodes':n,
            **{key:sum(bool(r[key]) for r in rows)/n for key in ['faithful_virtual_completion','wrong_start','unresolved','unstarted']},
            'mean_elapsed_s':mean(r['elapsed_s'] for r in rows),'mean_user_busy_s':mean(r['user_busy_s'] for r in rows),
            'mean_probe_s':mean(r['probe_s'] for r in rows),'mean_compute_ns':mean(r['compute_ns'] for r in rows),
            'termination_counts':dict(Counter(r['termination'] for r in rows))})
    lookup={(r['case_id'],r['seed'],r['method']):r for r in evaluations}
    public_by_id={r['episode_id']:r for r in public_rows};paired=[]
    for case in CASES:
        for seed in range(args.seeds):
            left=lookup[(case['id'],seed,'M')];right=lookup[(case['id'],seed,'B9')]
            def actions(row):return [d['selected_action']['action_id'] for d in public_by_id[row['episode_id']]['decisions']]
            paired.append({'case_id':case['id'],'seed':seed,'same_selected_actions':actions(left)==actions(right),
                'same_completion_and_wrong_start':all(left[k]==right[k] for k in ['faithful_virtual_completion','wrong_start','unstarted']),
                'elapsed_delta_s':left['elapsed_s']-right['elapsed_s']})
    blocked=sum(r['termination'].startswith('authorization_or_dispatch_blocked:') for r in evaluations)
    # This is an acceptance report, not an automatically granted research pass.
    # The frozen finite planner does not model the strict execution wrapper.
    summary={'scope':'L1_software_synthetic_replay_development_only','episodes':len(evaluations),
        'cases':len(CASES),'seeds_per_case':args.seeds,'methods':METHODS,'all_episodes_retained':True,
        'all_session_journals_replayed':True,'hardware_calls':0,'cloud_calls':0,
        'mechanism_checks_passed':all(mechanisms_result['checks'].values()),'fault_checks_passed':fault['all_passed'],
        'oracle_max_same_horizon_regret':oracle['max_same_horizon_regret'],
        'authorization_wrapper_blocked_episodes':blocked,'per_case_method':groups,'overall':overall,
        'M_vs_B9_paired':paired,'M_vs_B9_actions_all_equal':all(r['same_selected_actions'] for r in paired),
        'M_vs_B9_outcomes_all_equal':all(r['same_completion_and_wrong_start'] for r in paired),
        'strategy_superiority_claim':False,'real_success_rate_claim':False,
        'G2_verdict':'NOT_PASSED_PENDING_MODEL_ALIGNMENT_AND_CLAIM_NARROWING',
        'G2_reason':['finite O1/frozen09 accepted-confirmation cost omits strict11 authorization gate; internal oracle equality is not full-runtime optimality',
            'same-probe B9 retained; no distinct algorithmic contribution demonstrated',
            'all stress failures and stable-period probe overhead retained; no real probe compliance evidence'],
        'next_step13_allowed':False,'run_plan_sha256':hashlib.sha256((folder/'evaluation_private/run_plan.json').read_bytes()).hexdigest()}
    # Ensure the input freeze survived the complete run.
    assert all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in source_hashes)
    write(folder/'summary.json',summary)
    artifact_paths=[p for p in folder.rglob('*') if p.is_file()]
    write(folder/'artifacts_manifest.json',{'scope':summary['scope'],'files':[{'path':str(p.relative_to(ROOT)),
        'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(artifact_paths)]})
    print(encode({'result':'EXECUTED','episodes':len(evaluations),'wrapper_blocked':blocked,
        'M_vs_B9_actions_equal':summary['M_vs_B9_actions_all_equal'],'G2_verdict':summary['G2_verdict'],'output':str(folder)}),flush=True)


if __name__=='__main__':main()
