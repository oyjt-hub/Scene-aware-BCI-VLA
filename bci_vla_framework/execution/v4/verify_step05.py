"""Meaningful public-runner/replay/binding tests; no trained-model claims."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from bci_vla.evaluation_v4.runner import run_episode,make_public,make_question,observed_event,submission_request,text_serialization
from bci_vla.evaluation_v4.policies import RecentChoice,LegacyReference,fixed_inquiry
from bci_vla.interaction_contract_v4.validation import ContractError,canonical_digest
from bci_vla.data_v4.channel import answer
DATA=ROOT/'data/v4';OUT=ROOT/'execution/v4'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
 checks=[]
 def check(name,condition):checks.append({'name':name,'passed':bool(condition)})
 def rejects(name,call):
  try:call();check(name,False)
  except (ContractError,ValueError,AssertionError):check(name,True)
 summary=read(OUT/'05_runs/summary.json');captures=read(DATA/'captures/capture_manifest.json')['episodes']
 worlds=[json.loads(s) for s in (DATA/'private_worlds.jsonl').read_text(encoding='utf-8').splitlines()]
 runs=[read(Path(r['file'])) for r in summary['records']]
 check('B0_32_B1_binary_16',summary['methods']['B0']['runs']==32 and summary['methods'][LegacyReference.name]['runs']==16)
 check('B1_unsupported_explicit',len(summary['unavailable'])==16)
 check('actual_legacy_filter_updates',sum(r.get('legacy_filter_audit',{}).get('updates',0) for r in runs)==176 and all(r['legacy_filter_audit']['snapshot']['status']=='ready' and set(r['legacy_filter_audit']['update_reasons'])=={'joint_bayes_update'} for r in runs if 'legacy_filter_audit' in r))
 for r in runs:
  check('no_actions:'+r['method']+r['base_episode_id'],r['VLA_called'] is False and all(q['physical_dispatched'] is False and q['execution_authorized'] is False for q in r['submissions']))
  check('clock_monotonic:'+r['method']+r['base_episode_id'],all(a['at_s']<b['at_s'] for a,b in zip(r['journal'],r['journal'][1:])) and all(j['at_s']<24 for j in r['journal']))
  check('public_inputs_have_no_private_truth:'+r['method']+r['base_episode_id'],all(not {'hidden_truth','private_scoring_only','variant','channel_injected_mode'}&set(j['input']) for j in r['journal'] if 'input' in j))
  check('probe_never_replaces_confirmed:'+r['method']+r['base_episode_id'],all(j['input']['public_state']['confirmed_task']==j['public_after']['public_state']['confirmed_task'] for j in r['journal'] if j['kind']=='INTERACTION' and j['event']['question']['role']=='PROBE'))
  check('submitted_only_after_exact_confirmation:'+r['method']+r['base_episode_id'],all(j['event']['role']=='CONFIRM' and j['simulated_submission']['task']==j['event']['question']['proposal'] for j in r['journal'] if j.get('simulated_submission')))
 cap=next(c for c in captures if c['partition']=='validation' and len(c['frames'][0]['objects'])==2)
 world=next(w for w in worlds if w['base_episode_id']==cap['base_episode_id'] and w['variant']=='both_change')
 def stable_trace(r):return {k:v for k,v in r.items() if k!='compute_ns'}
 for method in [RecentChoice,LegacyReference]:
  a=run_episode(cap,world,method(),DATA);b=run_episode(cap,world,method(),DATA)
  check('deterministic_replay:'+method.name,stable_trace(a)==stable_trace(b))
  for control in ['STOP','CANCEL']:
   controlled=run_episode(cap,world,method(),DATA,control_at=(8,control))
   check('direct_control:'+method.name+control,controlled['journal'][-1]['event']['role']==control and all(s['submitted_at_s']<8 for s in controlled['submissions']))
  mapped=run_episode(cap,world,method(),DATA,mapping_at=8)
  check('mapping_change_invalidates_old_confirm:'+method.name,all(j['input']['public_state']['confirmed_task'] is None or j['input']['public_state']['confirmed_task']['task']['context']['mapping_revision']==2 for j in mapped['journal'] if j['at_s']>=8 and 'input' in j))
 class NoConfirm(RecentChoice):
  name='unit_no_confirm'
  def decide(self,p):return {'action':'TASK'}
 check('no_submission_without_confirmation',not run_episode(cap,world,NoConfirm(),DATA)['submissions'])
 class Bomb:
  name='unit_control_must_not_call_policy'
  def decide(self,p):raise AssertionError('control must bypass policy')
 check('STOP_bypasses_policy',run_episode(cap,world,Bomb(),DATA,control_at=(0,'STOP'))['journal'][0]['route']=='STOP_DIRECT_EXISTING_PATH')
 public=make_public(cap,0,[],None)
 for policy in [RecentChoice(),LegacyReference()]:
  pending=deepcopy(public);pending['public_state']['pending_question']=make_question(public,{'action':'TASK'},99)
  check('pending_requires_WAIT:'+policy.name,policy.decide(pending)['action']=='WAIT')
 q=make_question(public,{'action':'TASK'},1);ev=answer(world,q,0)
 altered=deepcopy(q);altered['question_id']='different-method-query';altered['context']['session_id']='different-method-session';altered['options'].reverse()
 check('noise_question_ids_session_and_order_independent',answer(world,altered,0)==ev)
 answer(world,q,4);answer(world,q,6)
 check('noise_extra_calls_do_not_change_future',answer(world,q,0)==ev)
 p=next(j['public_after'] for r in runs for j in r['journal'] if j.get('simulated_submission'))
 accepted=p['public_state']['confirmed_task'];proposal=accepted['task'];event=accepted['confirmation_receipt'];seen=set()
 check('positive_bound_submission',submission_request(p,proposal,event,seen)['execution_authorized'] is False)
 rejects('duplicate_submission_rejected',lambda:submission_request(p,proposal,event,seen))
 bad=deepcopy(proposal);bad['task_revision']+=1
 rejects('wrong_task_version_rejected',lambda:submission_request(p,bad,event,set()))
 bad_event=deepcopy(event);bad_event['role']='REJECT'
 rejects('reject_is_not_accept',lambda:submission_request(p,proposal,bad_event,set()))
 stale=deepcopy(p);stale['context']['mapping_revision']+=1
 rejects('stale_mapping_rejected',lambda:submission_request(stale,proposal,event,set()))
 expired=deepcopy(p);expired['as_of_s']=121
 rejects('expired_confirmation_rejected',lambda:submission_request(expired,proposal,event,set()))
 sealed=deepcopy(cap);sealed['partition']='test_sealed'
 rejects('formal_test_blocked',lambda:run_episode(sealed,world,RecentChoice(),DATA))
 multi=next(c for c in captures if len(c['frames'][0]['objects'])>2)
 rejects('B1_no_candidate_dropping',lambda:LegacyReference().decide(make_public(multi,0,[],None)))
 serialized=text_serialization(public);payload=json.loads(serialized['prompt'])
 check('B2_serialization_complete_not_truncated',payload['scene']['objects']==public['scene']['objects'] and payload['history']==public['history'] and not serialized['truncated'])
 check('B2_token_count_not_fabricated',serialized['token_count'] is None)
 scores={o['instance_id']:1/len(public['scene']['objects']) for o in public['scene']['objects']}
 check('fixed_inquiry_public_diagnostics_first',fixed_inquiry(public,scores)['action']=='PROBE')
 rejects('fixed_inquiry_rejects_unknown_candidate',lambda:fixed_inquiry(public,{'unknown-id':1.}))
 registry=read(ROOT/'configs/v4/baseline_registry.json')
 check('neural_methods_not_claimed_run',not registry['neural_methods_run'] and all(m['trainable_parameters'] is None for m in registry['methods'] if m['id'] not in ['B0',LegacyReference.name]))
 check('aux_supervision_controls_registered',{'B3_aux','B4_aux'}<={m['id'] for m in registry['methods']})
 for row in read(OUT/'05_before_manifest.json')['files']:check('preserved:'+row['path'],hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'])
 report={'scope':'LIBERO capture interaction-runner/legacy-adapter integrity only; no neural or robot success claim','status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','checks':len(checks),'passed':sum(c['passed'] for c in checks),'results':checks}
 (OUT/'05_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in report.items() if k!='results'}));print(json.dumps([c for c in checks if not c['passed']]))
 return 0 if report['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
