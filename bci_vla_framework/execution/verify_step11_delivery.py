"""Audit frozen offline step11 delivery. This cannot approve live deployment."""
from pathlib import Path
import ast
import hashlib
import json
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from bci_vla.adapters.vla import FakeVLAAdapter
from bci_vla.monitoring.journal import EventJournal
from bci_vla.contracts.session import replay_session_state
from bci_vla.protocols import thaw

def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
checks=[]
def check(name,ok):
    if not ok:raise AssertionError(name)
    checks.append(name)
def hashes(manifest):
    return all(Path(f['path']).is_file() and hashlib.sha256(Path(f['path']).read_bytes()).hexdigest()==f['sha256'] for f in read(manifest)['files'])

p=read('execution/progress.json');t=read('execution/11_test_result.json')
trace=read('execution/11_lifecycle_trace.json');replayed=read('execution/11_replay_result.json')
config=read('configs/execution/session.json');spec=read('configs/execution/fake_vla_interface.json')
check('01-11 remain complete',all(s['status']=='已完成' for s in p['steps'][:11]))
check('progress may continue after11',int(p['current_step'])>=12 and int(p['last_completed_step'])>=11)
check('G1 conditional scope retained',p['research_gates']['G1']['scope']=='conditional_theory_and_protocol_design')
check('291 software tests pass',t['tests_run']==291 and t['result']=='PASS' and t['failures']==t['errors']==t['skipped']==0)
check('frozen169 previous result retained',read('execution/10_test_result.json')['tests_run']==169)
check('fake only both configuration layers',config['scope']=='offline_fake_only' and config['physical_execution_allowed'] is False and spec['scope']=='offline_fixture' and spec['deployment_allowed'] is False)
check('real interface fields explicitly unverified',len(spec['unverified_live_fields'])>=6 and (ROOT/'docs/11_pi05_interface_checklist.txt').is_file())
check('11.1-11.12 mapped in main doc and acceptance',all(all(f'11.{i}' in (ROOT/q).read_text(encoding='utf-8-sig') for i in range(1,13)) for q in ['docs/11_execution_contract.txt','execution/11_acceptance.txt']))
check('ACK execution and endpoint stages distinct',trace['legal']['state_after_ack']=='SUBMITTED' and trace['legal']['state_after_completed_feedback']=='VERIFYING' and trace['legal']['final_state']=='SUCCEEDED')
check('one consumed authorization one fake submission',trace['legal']['authorization_consumed'] is True and trace['legal']['backend_submissions']==1)
check('unknown result no resubmit',trace['unknown_stop']['new_id_resubmit_rejected'] is True and trace['unknown_stop']['backend_submissions']==1)
check('stop request separate from fake feedback',trace['unknown_stop']['stop_before_feedback']=='REQUESTED' and trace['unknown_stop']['stop_after_feedback']=='SIMULATED_STOPPED' and trace['unknown_stop']['final_state']=='CANCELLED' and trace['physical_stop_verified'] is False)
check('no hardware cloud performance or G2 claims',t['hardware_calls']==t['cloud_calls']==trace['hardware_calls']==trace['cloud_calls']==0 and trace['G2_passed'] is False and trace['performance_claim'] is False)

results=[]
for path in ['execution/11_session_events.jsonl','execution/11_unknown_stop_events.jsonl']:
    journal=EventJournal.from_jsonl((ROOT/path).read_text(encoding='utf-8'))
    with patch.object(FakeVLAAdapter,'__init__',side_effect=AssertionError('backend construction forbidden')), \
         patch.object(FakeVLAAdapter,'submit',side_effect=AssertionError('backend dispatch forbidden')), \
         patch.object(FakeVLAAdapter,'query',side_effect=AssertionError('backend query forbidden')), \
         patch.object(FakeVLAAdapter,'stop',side_effect=AssertionError('backend stop forbidden')):
        state=replay_session_state(journal)
    results.append((journal,state))
check('two journals replay each checkpoint without backend',len(results)==2 and
    all(state==thaw(journal.records[-1]['payload']['state']) for journal,state in results) and
    [state['state'] for journal,state in results]==['SUCCEEDED','CANCELLED'])
check('saved replay output agrees',replayed['result']=='PASS' and replayed['backend_calls']==0 and all(c['exact_state_match'] and c['recomputed_each_checkpoint'] and c['backend_calls']==0 for c in replayed['cases']))
legal=results[0][0];records=legal.records
kinds={r['kind'] for r in records}
check('core policy task and execution events journaled',{'SESSION_STARTED','INPUT_RECEIVED','POLICY_RETURNED','AUTHORIZATION_GRANTED','SUBMISSION_INTENT','SUBMISSION_RETURNED','SIMULATED_EXECUTION_FEEDBACK','ENDPOINT_JUDGED'}<=kinds)
requests=[r['payload']['event']['request'] for r in records if r['kind']=='INTERACTION_REQUESTED']
check('TASK PROBE CONFIRM distinct and probe non executable',{q['mode'] for q in requests}=={'TASK_QUERY','CHANNEL_PROBE','CONFIRM'} and all(q['payload']['contract'] is None for q in requests if q['mode']=='CHANNEL_PROBE'))
auth=next(r['payload']['event'] for r in records if r['kind']=='AUTHORIZATION_GRANTED')
check('actual noisy confirmation freeze and model evidence recorded',auth['confirmation_is_noisy'] and auth['consume_once'] and auth['frozen_until_dispatch'] and auth['response']['payload']['sample_window']['source_kind']=='offline_replay' and len(auth['model_refs'])==6 and auth['authorization_prediction_gap_s']>0)
evidence=next(r['payload']['event']['evidence'] for r in records if r['kind']=='ENDPOINT_JUDGED')
check('endpoint submission context and causal time bound',evidence['session_id']==legal.session_id and evidence['submission_id']=='legal-command' and evidence['contract_id']==auth['contract_id'] and evidence['scene_id']==auth['scene_id'] and evidence['observed_at']['at']>results[0][1]['completion_feedback_at'])
check('physical permissions false throughout',all(r['payload']['state']['execution_allowed'] is False and r['payload']['state']['physical_stop_verified'] is False for journal,state in results for r in journal.records))

for folder in ['bci_vla/contracts','bci_vla/adapters/vla','bci_vla/monitoring']:
    for file in (ROOT/folder).glob('*.py'):
        tree=ast.parse(file.read_text(encoding='utf-8-sig'))
        imported={alias.name.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.Import) for alias in n.names}
        imported|={n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) and n.level==0 and n.module}
        check('no network or device SDK imports '+str(file.relative_to(ROOT)),not imported&{'socket','requests','urllib','httpx','serial','rospy','rclpy','torch','openpi','paramiko'})
check('11 source and output hashes intact',hashes('execution/11_source_manifest.json'))
check('10 source and output hashes intact',hashes('execution/10_source_manifest.json'))
check('09 source and output hashes intact',hashes('execution/09_source_manifest.json'))
check('10 historical integrity recheck passed',read('execution/10_integrity_recheck_after_step11.json')['result']=='PASS')
check('all TXT BOM CRLF',all(q.read_bytes().startswith(b'\xef\xbb\xbf') and b'\n' not in q.read_bytes().replace(b'\r\n',b'') for q in ROOT.rglob('*.txt')))
value={'date':'2026-10-16','step_id':'11','scope':'offline_fake_lifecycle_delivery','result':'PASS','count':len(checks),'checks':checks,
    'hardware_calls':0,'cloud_calls':0,'deployment_allowed':False,'G2_passed':False,
    'not_evaluated':['real BCI or pi0.5 identity/interface','physical execution and stop','durable exactly-once dispatch','sender authentication','step12 complete episodes','G2-G4']}
(ROOT/'execution/11_integrity_recheck_after_step12.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(value,ensure_ascii=True,indent=2))
