"""Offline delivery consistency, not G2 or deployment approval."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
checks=[]
def check(name,ok):
    if not ok:raise AssertionError(name)
    checks.append(name)

p=read('execution/progress.json'); t=read('execution/10_test_result.json')
registry=read('configs/comparisons/method_registry.json'); ablations=read('configs/comparisons/ablations.json')
audit=read('execution/10_oracle_audit.json'); trace=read('execution/10_decision_trace.json')
check('01-10 remain complete',all(s['status']=='已完成' for s in p['steps'][:10]))
check('progress can continue after 10',int(p['current_step'])>=11 and int(p['last_completed_step'])>=10)
check('G1 conditional and G2-G4 unstarted',p['research_gates']['G1']['scope']=='conditional_theory_and_protocol_design' and all(p['research_gates'][g]['status']=='未开始' for g in ['G2','G3','G4']))
check('169 software checks pass',t['tests_run']==169 and t['result']=='PASS' and t['failures']==t['errors']==t['skipped']==0)
check('complete method registry',{'B'+str(i) for i in range(11)}|{'B1P','M','M2','O1','O2'}=={m['id'] for m in registry['methods']})
check('eight virtual-only ablations',len(ablations['methods'])==8 and ablations['status']=='virtual_only' and ablations['deployment_allowed'] is False)
check('4 small fixtures and 12 equal-horizon checks',len(audit['cases'])==4 and sum(len(c['checks']) for c in audit['cases'])==12 and audit['max_decision_regret']<=1e-10)
check('exact values and measured computation recorded',all(r['oracle_compute_ns']>0 and r['method_compute_ns']>0 and abs(r['decision_regret_at_same_horizon'])<=1e-10 for c in audit['cases'] for r in c['checks']))
check('O2 explicit non-deployable noisy confirmation',all(c['omniscient_reference']['scope']=='non_deployable_omniscient_reference' and c['omniscient_reference']['confirmation_model']=='same_noisy_CONFIRM' for c in audit['cases']))
check('legacy unavailable not imputed',all(next(r for r in c['decisions'] if r['method']=='B0')['status']=='unavailable_external_legacy_trace' for c in trace['cases']))
check('no execution permissions',all(r['execution_allowed'] is False for c in trace['cases'] for r in c['decisions']))
check('real validation remains unfilled',read('configs/comparisons/tuning_and_selection.json')['chosen_baseline'] is None and registry['strongest_real_baseline'] is None and p['steps'][9]['real_validation']['deployment_allowed'] is False)
check('no research win or G2 claim',audit['strategy_superiority_claim'] is False and audit['G2_passed'] is False and audit['hardware_calls']==audit['cloud_calls']==0)
check('10.1-10.8 mapped',all(f'10.{i}' in (ROOT/'execution/10_acceptance.txt').read_text(encoding='utf-8-sig') for i in range(1,9)))
check('10 hashes intact',all(hashlib.sha256(Path(f['path']).read_bytes()).hexdigest()==f['sha256'] for f in read('execution/10_source_manifest.json')['files']))
check('09 hashes intact',all(hashlib.sha256(Path(f['path']).read_bytes()).hexdigest()==f['sha256'] for f in read('execution/09_source_manifest.json')['files']))
check('TXT BOM CRLF intact',all(q.read_bytes().startswith(b'\xef\xbb\xbf') and b'\n' not in q.read_bytes().replace(b'\r\n',b'') for q in ROOT.rglob('*.txt')))
v={'date':'2026-10-16','scope':'offline_comparison_delivery','result':'PASS','count':len(checks),'checks':checks,
   'not_evaluated':['real validation tuning','complete NOIR/RBC reproductions','formal performance comparison','robot execution','G2-G4']}
(ROOT/'execution/10_integrity_recheck_after_step11.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(v,ensure_ascii=True,indent=2))
