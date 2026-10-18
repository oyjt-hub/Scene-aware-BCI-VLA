"""Versioned 4.1 LIBERO dataset, observable supervision and private lineage."""
from copy import deepcopy
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from bci_vla.data_v4.channel import answer,diagnostic_summary,digest,world_at
from bci_vla.interaction_contract_v4.validation import canonical_digest,_shape,_public,_private
DATA=ROOT/'data/v4'
VERSION='bci-vla-interaction/4.1'
VARIANTS=['stable','goal_change','channel_change','both_change','no_response','feature_dropout','mapping_change','candidate_miss','cancel','stop','ambiguous']
def schema():
 s=json.loads((ROOT/'configs/v4/interfaces.json').read_text(encoding='utf-8'))
 def replace(v):
  if isinstance(v,dict):
   if v.get('const')=='bci-vla-interaction/4.0':v['const']=VERSION
   for x in v.values():replace(x)
  elif isinstance(v,list):
   for x in v:replace(x)
 replace(s)
 s['$id']='urn:bci-vla:libero-development-data:4.1'
 s['$defs']['public_input']['properties']['provenance']['properties']['bci']={'const':'simulated_channel'}
 s['$defs']['private_labels']['properties']['split']={'enum':['train','validation','calibration_fit','calibration_audit']}
 s['x-protocol']['scope']='actual LIBERO images/privileged scene state plus simulated BCI; development only'
 s['x-protocol']['frozen_parent']='configs/v4/interfaces.json remains unchanged; same semantic validators'
 return s
def task(ctx,source):
 return {'task_id':'task-'+source,'task_revision':1,'suggestion_id':'suggestion-'+source,'context':deepcopy(ctx),'action':'pick','source_id':source,'destination_id':None,'valid_until_s':120.,'capability_verified':False}
def make_question(ctx,ids,slot,history):
 role=['TASK','PROBE','PROBE','INTENT_QUERY','PROBE','CONFIRM','PROBE','PROBE'][slot]
 q={'question_id':'q-'+str(slot),'role':role,'context':deepcopy(ctx),'layout_id':'probe-'+str(len(ids)) if role=='PROBE' else 'confirm-2' if role=='CONFIRM' else 'object-'+str(len(ids)), 'issued_at_s':2.+slot*2,'expires_at_s':3.+slot*2,'presentation_ack':True,'diagnostic_option_id':None,'proposal':None}
 if role=='CONFIRM':
  last=next((e for e in reversed(history) if e['valid'] and e['role'] in ['TASK','INTENT_QUERY']),None)
  selected=ids[0]
  if last:
   selected=next(o['reference'] for o in last['question']['options'] if o['option_id']==last['evidence']['selected_option_id'])
  q['proposal']=task(ctx,selected)
  q['options']=[{'option_id':v,'symbol':'s'+str(i),'kind':'DECISION','reference':v} for i,v in enumerate(['accept','reject'])]
 else:
  q['options']=[{'option_id':'o'+str(i),'symbol':'s'+str(i),'kind':'DIAGNOSTIC' if role=='PROBE' else 'OBJECT','reference':oid} for i,oid in enumerate(ids)]
  if role=='PROBE':q['diagnostic_option_id']='o'+str((slot-1)%len(ids))
 return q
def generate(capture,variant):
 movable=[o['instance_id'] for o in capture['frames'][0]['objects'] if o['object_valid']]
 assert len(movable)>=2,'Need two in-frame candidates'
 # Targets are sampled from all valid observed instances; never use benchmark goal to select candidates.
 movable=sorted(movable);a,b=movable[:2]
 world={'episode_id':capture['base_episode_id'],'initial_goal':a,'later_goal':b if variant in ['goal_change','both_change'] else a,'initial_success':.9,'later_success':.35 if variant in ['channel_change','both_change'] else .9,'change_at_s':8.}
 history=[];confirmed=None;records=[]
 for slot in range(8):
  ctx={'session_id':'session-'+capture['base_episode_id'],'scene_id':capture['group_id'],'scene_revision':1,'mapping_id':'map-1','mapping_revision':2 if variant=='mapping_change' and slot>=3 else 1}
  frame=capture['frames'][int(slot>=4)];objects=deepcopy(frame['objects'])
  if variant=='candidate_miss':objects=[o for o in objects if o['instance_id']!=a]
  if variant=='feature_dropout':
   for o in objects:
    if o['instance_id']==a:o.update(roi_feature=None,roi_valid=False)
  ids=[o['instance_id'] for o in objects if o['object_valid']]
  assert ids
  for old in history:
   if old['question'] is not None and old['question']['context']!=ctx:old.update(valid=False,invalid_reason='STALE_CONTEXT')
  if confirmed and confirmed['task']['context']!=ctx:confirmed=None
  q=make_question(ctx,ids,slot,history)
  ev=answer(world,q,q['issued_at_s'],force_no_response=variant=='no_response' and slot==6)
  no=ev['score_kind']=='NONE';role='NO_RESPONSE' if no else q['role']
  if q['role']=='CONFIRM' and not no and ev['selected_option_id']=='reject':role='REJECT'
  event={'event_id':'e-'+str(slot),'sequence':slot+1,'role':role,'original_role':q['role'] if no or role=='REJECT' else None,'question':q,'started_at_s':q['issued_at_s']+.1,'ended_at_s':q['expires_at_s'] if no else q['issued_at_s']+.6,'valid':True,'invalid_reason':None,'evidence':ev}
  if variant in ['cancel','stop'] and slot==7:event.update(role=variant.upper(),original_role=None,question=None,evidence=None)
  history.append(event)
  if role=='CONFIRM':confirmed={'task':deepcopy(q['proposal']),'confirmation_receipt':deepcopy(event)}
  if role=='REJECT' or event['role'] in ['STOP','CANCEL']:confirmed=None
  now=event['ended_at_s'];summary=diagnostic_summary(ctx,history,now,'probe-'+str(len(ids)))
  selection=next((e['event_id'] for e in reversed(history) if e['valid'] and e['role'] in ['TASK','INTENT_QUERY']),None)
  public={'record_type':'PUBLIC_INPUT','schema_version':VERSION,'sample_id':digest([capture['base_episode_id'],variant,slot])[:32],'context':ctx,'as_of_s':now,'provenance':{'bci':'simulated_channel','vision':'privileged_simulator','feature_profile_id':'libero-projected-rgb-stats-6-semantic-missing-1','real_eeg':False,'physical_robot':False}, 'scene':{'observed_at_s':10. if slot>=4 else 0.,'image_ref':frame['image_path'],'image_valid':True,'roi_dim':6,'semantic_dim':1,'objects':objects,'truncation':{'original_count':len(objects),'retained_count':len(objects),'policy':'none','target_recall_known':False}},'history':deepcopy(history[-8:]),'current_event_id':event['event_id'],'selection_event_id':selection,'task_options':[],'public_state':{'confirmed_task':deepcopy(confirmed),'pending_suggestion':None,'pending_question':None,'diagnostics':summary,'stop_latched':event['role']=='STOP','execution_authorized':False}}
  # Observable-set supervision: do not force latent goal from one noisy answer.
  label_ids=ids[:]
  identifiable=len(label_ids)>=2
  sha=canonical_digest(public);goal,_=world_at(world,now)
  future=None
  if summary['status']=='SUFFICIENT':
   future_q=make_question(ctx,ids,1,history);future_q.update(issued_at_s=now+2,expires_at_s=now+3,question_id='private-future-probe')
   future_q['diagnostic_option_id']=next(o['option_id'] for o in future_q['options'] if o['symbol']==summary['covered_symbols'][0])
   future_ev=answer(world,future_q,now+2)
   future={'question':future_q,'answer':future_ev,'success':future_ev['selected_option_id']==future_q['diagnostic_option_id']}
  private={'record_type':'PRIVATE_LABELS','schema_version':VERSION,'sample_id':public['sample_id'],'variant_id':'latent-0','public_input_sha256':sha,'split':capture['partition'],'hidden_truth':{'goal':task(ctx,goal),'channel_injected_mode':variant,'changed_at_s':8. if variant in ['goal_change','channel_change','both_change'] else None},'identifiability':{'intent':'AMBIGUOUS' if identifiable else 'UNOBSERVABLE','channel':'SUPPORTED' if future else 'INSUFFICIENT'},'labels':{'intent':{'mode':'SET' if identifiable else 'NONE','candidate_ids':label_ids if identifiable else [],'outside':variant=='candidate_miss' if identifiable else False,'mask':identifiable},'main_task':{'mode':'CLARIFICATION','task':None,'text':'Which object and task do you mean?','mask':True},'channel':{'mask':future is not None,'next_probe_success':future['success'] if future else None,'target_time_s':now+2 if future else None},'interaction':{'mask':True,'action':'WAIT' if event['role'] in ['STOP','CANCEL'] else 'CHECK_CHANNEL' if summary['status']!='SUFFICIENT' else 'ASK_INTENT','teacher':'public_rule_v1','teacher_input_sha256':sha}}}
  # outside is a private metric stratum, not a privileged observable training target.
  private['labels']['intent']['outside']=bool(identifiable)
  if event['role'] in ['STOP','CANCEL']:
   private['labels']['main_task']={'mode':'NONE','task':None,'text':None,'mask':False}
  records.append((public,private,future))
 return records,world
def main():
 if (DATA/'data_manifest.json').exists():
  assert sys.argv[1:]==['--rebuild-unfrozen'] and not (ROOT/'execution/v4/04_acceptance.json').exists(),'Preserve frozen generated dataset'
  assert json.loads((DATA/'data_manifest.json').read_text(encoding='utf-8'))['training_run'] is False
 capture=json.loads((DATA/'captures/capture_manifest.json').read_text());assert capture['status']=='PASS' and len(capture['episodes'])==48
 s=schema();(ROOT/'configs/v4/interfaces_data_4_1.json').write_text(json.dumps(s,indent=2)+'\n',encoding='utf-8')
 defs=s['$defs'];counts={};lineage=[];worlds=[];langs=[];files={};counterpairs=[]
 def emit(part,kind,value):
  key=(part,kind)
  if key not in files:
   path=DATA/part/(kind+'.jsonl');path.parent.mkdir(exist_ok=True);files[key]=path.open('w',encoding='utf-8')
  files[key].write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
 for cap in capture['episodes']:
  counts[cap['partition']]=counts.get(cap['partition'],0)+1
  for variant in VARIANTS:
   records,world=generate(cap,variant);worlds.append({'base_episode_id':cap['base_episode_id'],'variant':variant,**world})
   for public,private,future in records:
    _shape(public,defs['public_input'],defs);_public(public);_shape(private,defs['private_labels'],defs);_private(private,public)
    emit(cap['partition'],'public_input',public);emit(cap['partition'],'private_labels',private)
    row={'sample_id':public['sample_id'],'base_episode_id':cap['base_episode_id'],'group_id':cap['group_id'],'partition':cap['partition'],'variant':variant,'public_sha256':canonical_digest(public),'frame_sha256':cap['frames'][int(public['as_of_s']>=10)]['image_sha256']}
    lineage.append(row)
    if future:emit(cap['partition'],'private_future_probe',dict(sample_id=public['sample_id'],evidence=future))
    confirmed=public['public_state']['confirmed_task']
    if confirmed:
     source=confirmed['task']['source_id'];alias=next(o['alias'] for o in public['scene']['objects'] if o['instance_id']==source)
     held=cap['frames'][int(public['as_of_s']>=10)]['predicates'].get(source,{}).get('grasp_contact',False)
     langs.append({'sample_id':public['sample_id'],'partition':cap['partition'],'source':'public confirmed contract plus actual privileged LIBERO grasp predicate; template annotation, not model output','main_task_restatement':'Pick up '+alias+'.','next_subtask':None if held else 'Pick up '+alias+'.','next_subtask_loss_mask':not held,'place_phase_available':False,'task_execution_verified':False})
    if variant=='ambiguous':
     twin=deepcopy(private);twin['variant_id']='latent-1';twin['hidden_truth']['goal']=task(public['context'],next(o['instance_id'] for o in public['scene']['objects'] if o['instance_id']!=private['hidden_truth']['goal']['source_id'] and o['object_valid']))
     assert twin['labels']==private['labels'];_private(twin,public);emit(cap['partition'],'private_labels',twin)
     counterpairs.append({'sample_id':public['sample_id'],'partition':cap['partition'],'variants':['latent-0','latent-1'],'same_public_sha256':canonical_digest(public)})
 for f in files.values():f.close()
 for name,rows in [('lineage',lineage),('private_worlds',worlds),('language_annotations',langs),('counterfactual_pairs',counterpairs)]:
  (DATA/(name+'.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
 manifest={'version':'libero-data-v4.1-dev1','schema':'configs/v4/interfaces_data_4_1.json','platform':'actual LIBERO','brain_input':'simulated; no EEG','base_episodes':len(capture['episodes']),'base_episodes_per_partition':counts,'public_rows':len(lineage),'private_label_rows':len(lineage)+len(counterpairs),'counterfactual_pairs':len(counterpairs),'language_annotation_rows':len(langs),'variants':VARIANTS,'roi_features':'6D actual image RGB crop mean/std, not pretrained ROI embeddings','semantic_features':'missing with mask false; no fabricated embedding','vision_provenance':'privileged simulator geometry, not Florence/SAM detection','successful_task_demonstrations':False,'place_phase_demonstrations_available':False,'formal_test_generated':False,'training_run':False,'limitations':['Development-only capture; neutral steps are not successful manipulation rollouts.','Main proposal target remains clarification under ambiguous evidence; confirmed-task restatement is not main-task generation supervision.','Next-subtask templates only cover actually observed ungrasped phase; no invented placement labels.','Object asset types may overlap, scene/instance groups are disjoint.']}
 (DATA/'data_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(manifest,ensure_ascii=False))
if __name__=='__main__':main()
