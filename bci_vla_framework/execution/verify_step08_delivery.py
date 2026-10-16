"""Delivery consistency for the offline joint-filter branch, not G2 evaluation."""
from pathlib import Path
import hashlib
import json
import math

root=Path(__file__).resolve().parents[1]
def load(p):return json.loads((root/p).read_text(encoding='utf-8-sig'))
checks=[]
def check(name,condition):
    if not condition:raise AssertionError(name)
    checks.append(name)

p=load('execution/progress.json');t=load('execution/08_test_result.json');trace=load('execution/08_alias_trace.json');c=load('configs/belief/simulation_binary_filter.json')
check('01-08 remain complete',all(s['status']=='已完成' for s in p['steps'][:8]))
check('progress may advance beyond 08',int(p['current_step'])>=9)
check('G1 conditional and G2-G4 unstarted',p['research_gates']['G1']['scope']=='conditional_theory_and_protocol_design' and all(p['research_gates'][g]['status']=='未开始' for g in ['G2','G3','G4']))
check('offline-only step08 scope',p['steps'][7]['execution_scope']=='offline_binary_joint_inference' and c['status']=='simulation_only' and c['deployment_allowed'] is False)
check('92 software checks pass',t['result']=='PASS' and t['tests_run']==92 and t['failures']==t['errors']==t['skipped']==0)
known=[g['goal_id'] for g in c['vocabulary']['goals'] if g['kind']=='task'];left,right=known
check('task aliases remain equally likely',all(math.isclose(trace['audit'][0]['after']['goal_marginals'][g],0.5) for g in known))
check('probe updates via correlation',math.isclose(trace['final']['goal_marginals'][right],18/19) and trace['audit'][1]['probe_prompt_symbol']=='0')
after=trace['audit'][0]['after']['belief']['mass'];mass={(x['goal_id'],x['mode_id']):x['probability'] for x in after}
check('factorization does not invent states',mass[(left,'normal')]==0 and mass[(right,'flip')]==0)
check('no execution authorization',trace['final']['execution_allowed'] is False and trace['scope']=='analytic_fixed_evidence_replay')
check('all 08.1-08.9 mapped',all(f'08.{i}' in (root/'execution/08_acceptance.txt').read_text(encoding='utf-8-sig') for i in range(1,10)))
check('source and final software hashes match',all(hashlib.sha256(Path(f['path']).read_bytes()).hexdigest()==f['sha256'] for f in load('execution/08_source_manifest.json')['files']))
check('TXT encoding intact',all(q.read_bytes().startswith(b'\xef\xbb\xbf') and b'\n' not in q.read_bytes().replace(b'\r\n',b'') for q in root.rglob('*.txt')))
v={'date':'2026-10-16','scope':'offline_joint_filter_delivery','result':'PASS','count':len(checks),'checks':checks,'not_evaluated':['real model validity','unknown-state emission calibration','clock uncertainty integration','active strategy','authorization or robot execution','G2-G4']}
(root/'execution/08_integrity_recheck_after_step09.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(v,ensure_ascii=True,indent=2))
