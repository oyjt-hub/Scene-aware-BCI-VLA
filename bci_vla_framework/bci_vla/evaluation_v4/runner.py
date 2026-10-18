"""LIBERO-capture public interaction loop. No policy inference or robot dispatch.

Actual images/states come from frozen LIBERO captures. Human-channel responses
are generated online for the policy's current question, never fixed answer replay.
Private schedule/scoring data are held by the harness, outside policy inputs.
"""
from copy import deepcopy
from pathlib import Path
import hashlib,json,time
from bci_vla.data_v4.channel import answer,diagnostic_summary,digest,world_at
from bci_vla.data_v4.contracts import validate_public
from bci_vla.interaction_contract_v4.validation import canonical_digest,control_route,ContractError
def make_public(capture,now,history,confirmed,revision=1,stopped=False,declared_action=None):
 if declared_action not in (None,'pick'):raise ValueError('Only an explicitly public pick protocol is supported')
 frame=capture['frames'][int(now>=10)]
 ctx={'session_id':'session-'+capture['base_episode_id'],'scene_id':capture['group_id'],'scene_revision':1,'mapping_id':'map-1','mapping_revision':revision}
 hist=deepcopy(history[-8:])
 for e in hist:
  if e['question'] is not None and e['question']['context']!=ctx:e.update(valid=False,invalid_reason='STALE_CONTEXT')
 if confirmed and confirmed['task']['context']!=ctx:confirmed=None
 selection=next((e['event_id'] for e in reversed(hist) if e['valid'] and e['role'] in ['TASK','INTENT_QUERY']),None)
 ids=[o['instance_id'] for o in frame['objects'] if o['object_valid']]
 # A common declared action is a PUBLIC menu for EVERY visible candidate. It
 # contains no selected/hidden goal and leaves OBJECT question noise unchanged.
 tasks=[] if declared_action is None else [{'task_id':'public-pick-'+oid,'task_revision':1,'suggestion_id':'public-pick-suggestion-'+oid,'context':deepcopy(ctx),'action':'pick','source_id':oid,'destination_id':None,'valid_until_s':now+120.,'capability_verified':False} for oid in ids]
 return {'record_type':'PUBLIC_INPUT','schema_version':'bci-vla-interaction/4.1','sample_id':digest([capture['base_episode_id'],now,len(history)])[:32],'context':ctx,'as_of_s':now,'provenance':{'bci':'simulated_channel','vision':'privileged_simulator','feature_profile_id':'libero-projected-rgb-stats-6-semantic-missing-1','real_eeg':False,'physical_robot':False},'scene':{'observed_at_s':10. if now>=10 else 0.,'image_ref':frame['image_path'],'image_valid':True,'roi_dim':6,'semantic_dim':1,'objects':deepcopy(frame['objects']),'truncation':{'original_count':len(frame['objects']),'retained_count':len(frame['objects']),'policy':'none','target_recall_known':False}},'history':hist,'current_event_id':hist[-1]['event_id'] if hist else None,'selection_event_id':selection,'task_options':tasks,'public_state':{'confirmed_task':deepcopy(confirmed),'pending_suggestion':None,'pending_question':None,'diagnostics':diagnostic_summary(ctx,[e for e in history if e['valid'] and e.get('question') and e['question']['context']==ctx],now,'probe-'+str(len(ids))),'stop_latched':stopped,'execution_authorized':False}}
def make_question(public,decision,sequence):
 ctx=deepcopy(public['context']);now=public['as_of_s'];role=decision['action'];ids=[o['instance_id'] for o in public['scene']['objects'] if o['object_valid']]
 q={'question_id':'q-'+str(sequence),'role':role,'context':ctx,'layout_id':'confirm-2' if role=='CONFIRM' else ('probe-' if role=='PROBE' else 'object-')+str(len(ids)),'issued_at_s':now,'expires_at_s':now+1.5,'presentation_ack':True,'options':[],'diagnostic_option_id':None,'proposal':None}
 if role=='CONFIRM':
  assert decision['source_id'] in ids
  q['proposal']={'task_id':'proposal-'+str(sequence),'task_revision':sequence,'suggestion_id':'suggestion-'+str(sequence),'context':deepcopy(ctx),'action':'pick','source_id':decision['source_id'],'destination_id':None,'valid_until_s':120.,'capability_verified':False}
  q['options']=[{'option_id':v,'symbol':'s'+str(i),'kind':'DECISION','reference':v} for i,v in enumerate(['accept','reject'])]
 else:
  q['options']=[{'option_id':'o'+str(i),'symbol':'s'+str(i),'kind':'DIAGNOSTIC' if role=='PROBE' else 'OBJECT','reference':oid} for i,oid in enumerate(ids)]
  if role=='PROBE':
   covered=public['public_state']['diagnostics']['covered_symbols'];target=next((o for o in q['options'] if o['symbol'] not in covered),q['options'][sequence%len(ids)])
   q['diagnostic_option_id']=target['option_id']
 return q
def observed_event(world,q,sequence,repeat=0):
 ev=answer(world,q,q['issued_at_s'],repeat);no=ev['score_kind']=='NONE'
 role='NO_RESPONSE' if no else 'REJECT' if q['role']=='CONFIRM' and ev['selected_option_id']=='reject' else q['role']
 return {'event_id':'e-'+str(sequence),'sequence':sequence,'role':role,'original_role':q['role'] if no or role=='REJECT' else None,'question':deepcopy(q),'started_at_s':q['issued_at_s']+.25,'ended_at_s':q['expires_at_s'] if no else q['issued_at_s']+1.25,'valid':True,'invalid_reason':None,'evidence':ev}
def submission_request(public,proposal,event,seen):
 validate_public(public)
 if public['public_state']['stop_latched']:raise ContractError('submission:stop_latched')
 if event['role']!='CONFIRM' or not event['valid'] or event['question']['proposal']!=proposal:raise ContractError('submission:exact_confirm_required')
 if event['question']['context']!=public['context'] or proposal['context']!=public['context'] or public['as_of_s']>=proposal['valid_until_s']:raise ContractError('submission:stale_or_expired')
 if public['public_state']['confirmed_task']!={'task':proposal,'confirmation_receipt':event}:raise ContractError('submission:confirmation_not_current')
 key=canonical_digest([proposal,event['event_id']])
 if key in seen:raise ContractError('submission:duplicate')
 seen.add(key)
 return {'kind':'SIMULATED_SUBMISSION_REQUEST','request_id':key,'task':deepcopy(proposal),'confirmation_event_id':event['event_id'],'submitted_at_s':public['as_of_s'],'execution_authorized':False,'VLA_called':False,'physical_dispatched':False}
def text_serialization(public):
 # Full public source fields, including numeric ROI vectors, without private truth.
 body={k:deepcopy(v) for k,v in public.items() if k not in ['sample_id','provenance']}
 prompt=json.dumps(body,sort_keys=True,ensure_ascii=False,separators=(',',':'))
 return {'prompt':prompt,'utf8_bytes':len(prompt.encode()),'token_count':None,'token_count_status':'not tokenized; exact current tokenizer required in V4-07','image_ref':public['scene']['image_ref'],'truncated':False}
def run_episode(capture,world,policy,data_root,horizon=24,control_at=None,mapping_at=None,*,declared_action=None):
 """Original development entry point; calibration/test remain forbidden."""
 if capture['partition'] not in ['train','validation']:raise ValueError('Runner only accepts development train/validation, not calibration or test')
 return _run_episode(capture,world,policy,data_root,horizon,control_at,mapping_at,calibration_audit=False,declared_action=declared_action)

def run_calibration_audit_episode(capture,world,policy,data_root,horizon=24,control_at=None,mapping_at=None,*,declared_action=None):
 """Frozen-policy audit only; preserve the real calibration_audit partition."""
 if capture['partition']!='calibration_audit':raise ValueError('Calibration audit runner accepts calibration_audit only; fit/test/train/validation are forbidden')
 return _run_episode(capture,world,policy,data_root,horizon,control_at,mapping_at,calibration_audit=True,declared_action=declared_action)

def _run_episode(capture,world,policy,data_root,horizon,control_at,mapping_at,*,calibration_audit,declared_action):
 if type(calibration_audit) is not bool:raise ValueError('Explicit runner scope required')
 allowed=('calibration_audit',) if calibration_audit else ('train','validation')
 if capture['partition'] not in allowed:raise ValueError('Capture partition does not match runner scope')
 if declared_action not in (None,'pick'):raise ValueError('Only an explicitly public pick protocol is supported')
 for frame in capture['frames']:
  assert hashlib.sha256((Path(data_root)/frame['image_path']).read_bytes()).hexdigest()==frame['image_sha256']
 history=[];confirmed=None;seen=set();questions={};journal=[];submissions=[];scoring=[];compute=[]
 for sequence in range(1,int(horizon/2)+1):
  now=(sequence-1)*2.;revision=2 if mapping_at is not None and now>=mapping_at else 1
  if confirmed and (confirmed['task']['context']['mapping_revision']!=revision or confirmed['task']['valid_until_s']<=now):confirmed=None
  if control_at is not None and now>=control_at[0]:
   event={'event_id':'e-control','sequence':sequence,'role':control_at[1],'original_role':None,'question':None,'started_at_s':now,'ended_at_s':now,'valid':True,'invalid_reason':None,'evidence':None};history.append(event);confirmed=None
   journal.append({'kind':'CONTROL','at_s':now,'event':event,'route':control_route(event)});break
  public=make_public(capture,now,history,confirmed,revision,declared_action=declared_action);validate_public(public)
  started=time.perf_counter_ns();decision=policy.decide(deepcopy(public));compute.append(time.perf_counter_ns()-started)
  if decision['action'] not in ['WAIT','TASK','INTENT_QUERY','PROBE','CONFIRM']:raise ValueError('unsupported_interaction_action')
  if decision['action']=='WAIT':journal.append({'kind':'WAIT','at_s':now,'input':public,'decision':decision});continue
  q=make_question(public,decision,sequence);key=canonical_digest([q['role'],q['context'],q['options'],q['diagnostic_option_id'],q['proposal'],int(now/2)])
  repeat=questions.get(key,0);questions[key]=repeat+1
  event=observed_event(world,q,sequence,repeat);history.append(event)
  if event['role']=='CONFIRM':confirmed={'task':deepcopy(q['proposal']),'confirmation_receipt':deepcopy(event)}
  elif event['role']=='REJECT':confirmed=None
  end_public=make_public(capture,event['ended_at_s'],history,confirmed,revision,declared_action=declared_action);validate_public(end_public)
  request=None
  if event['role']=='CONFIRM':
   request=submission_request(end_public,q['proposal'],event,seen);submissions.append(request)
   goal,_=world_at(world,event['ended_at_s']);scoring.append({'request_id':request['request_id'],'goal_at_submission':goal,'source_matches_latent_goal':q['proposal']['source_id']==goal,'not_robot_task_success':True})
  journal.append({'kind':'INTERACTION','at_s':now,'input':public,'decision':decision,'event':event,'public_after':end_public,'simulated_submission':request})
 result={'method':policy.name,'base_episode_id':capture['base_episode_id'],'partition':capture['partition'],'platform':'LIBERO actual frozen capture replay; online simulated BCI questions','horizon_s':horizon,'journal':journal,'submissions':submissions,'private_scoring_only':scoring,'compute_ns':compute,'real_dynamics_advanced':False,'training':False,'VLA_called':False,'formal_test':False}
 result['calibration_audit']=calibration_audit
 result['declared_action']=declared_action
 if getattr(policy,'legacy',None) is not None:
  result['legacy_filter_audit']={'snapshot':policy.legacy.filter.snapshot(),'update_reasons':[u['reason'] for u in policy.legacy.updates],'updates':sum(u['updated'] for u in policy.legacy.updates),'full_legacy_planner_used':False}
 return result
