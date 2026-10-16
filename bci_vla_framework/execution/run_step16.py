"""Run every locked synthetic episode once; no model/hardware connections."""
from pathlib import Path
from datetime import datetime,timezone
import gzip,hashlib,json,sys
from collections import Counter
from statistics import mean
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from bci_vla.study_v1.runner import run_episode
from bci_vla.study_v1.statistics import paired_summary,judge
from bci_vla.protocols import thaw
from bci_vla.benchmark.audits import mechanisms,faults
def read(p):return json.loads((R/p).read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def encode(d):return json.dumps(d,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'
lock=read('execution/16_preregistration.json');assert all(hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256'] for x in lock['files'])
config=read('configs/study16.json');folder=R/'runs/registered_step16_synthetic_v1';assert not folder.exists();folder.mkdir()
save(folder/'start_receipt.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),preregistration_sha256=hashlib.sha256((R/'execution/16_preregistration.json').read_bytes()).hexdigest()))
for sub in ['online','evaluation_private','journals']:(folder/sub).mkdir()
save(folder/'mechanisms.json',mechanisms());save(folder/'faults.json',faults())
evaluations=[];actions={};counter=0
with gzip.open(folder/'online/episodes.jsonl.gz','wt',encoding='utf-8') as pub,gzip.open(folder/'evaluation_private/worlds.jsonl.gz','wt',encoding='utf-8') as priv:
 for case in config['cases']:
  with gzip.open(folder/'journals'/(case['id']+'.jsonl.gz'),'wt',encoding='utf-8') as journals:
   for seed in config['seeds']:
    for method in config['methods']:
     identifier='e-'+hashlib.sha256(f"study16|{case['id']}|{seed}|{method}".encode()).hexdigest()[:24]
     public,evaluation,private,journal=run_episode(case,seed,method,identifier)
     pub.write(encode(public));priv.write(encode(private));evaluations.append(evaluation)
     actions[case['id'],seed,method]=[d['selected_action']['action_id'] for d in public['decisions']]
     journals.write(encode(dict(episode_id=identifier,records=[thaw(r) for r in journal.records])))
     counter+=1
  print(json.dumps(dict(case_completed=case['id'],episodes=counter)),flush=True)
with (folder/'evaluation_joined.jsonl').open('w',encoding='utf-8') as f:
 for row in evaluations:f.write(encode(row))
primary=[paired_summary(evaluations,'M','B9',m,bootstrap=config['bootstrap_resamples'],alpha=config['primary_alpha_each']) for m in config['primary_metrics']]
secondary=[paired_summary(evaluations,'M',m,k,bootstrap=config['bootstrap_resamples']) for m in config['methods'] if m not in ['M','B9'] for k in config['primary_metrics']]
save(folder/'paired_statistics.json',dict(primary=primary,secondary_descriptive=secondary,decision=judge(primary)))
overall=[dict(method=m,episodes=len(rows),**{k:mean(float(r[k]) for r in rows) for k in ['faithful_virtual_completion','wrong_start','unresolved','unstarted','elapsed_s','user_busy_s','probe_s','suppressed_new_goal_start','false_new_goal_start','confident_wrong_channel_view']},
 termination_counts=dict(Counter(r['termination'] for r in rows))) for m in config['methods'] for rows in [[r for r in evaluations if r['method']==m]]]
summary=dict(scope=config['scope'],episodes=counter,planned_episodes=config['episodes'],all_rows_retained=counter==config['episodes'],overall=overall,
 runtime_errors=[r for r in evaluations if r['runtime_error']],all_journals_replayed=True,
 M_B9_action_sequences_equal=all(actions[c['id'],s,'M']==actions[c['id'],s,'B9'] for c in config['cases'] for s in config['seeds']),
 M_B9_outcomes_equal=all(all(next(r for r in evaluations if r['case_id']==c['id'] and r['seed']==s and r['method']=='M')[k]==next(r for r in evaluations if r['case_id']==c['id'] and r['seed']==s and r['method']=='B9')[k] for k in ['faithful_virtual_completion','wrong_start','unstarted']) for c in config['cases'] for s in config['seeds']),
 judgement=judge(primary),real_BCI_trials=0,physical_robot_commands=0,new_LIBERO_rollouts=0,subjective_results=None,formal_G4_passed=False,
 ended_utc=datetime.now(timezone.utc).isoformat())
assert all(hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256'] for x in lock['files'])
save(folder/'summary.json',summary)
save(folder/'artifacts_manifest.json',dict(files=[dict(path=str(q.resolve()),sha256=hashlib.sha256(q.read_bytes()).hexdigest()) for q in sorted(folder.rglob('*')) if q.is_file()]))
print(json.dumps({k:v for k,v in summary.items() if k not in ['overall','runtime_errors']},ensure_ascii=True))
