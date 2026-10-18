"""Validated public records -> allowlisted tensors, with identity metadata separate."""
from copy import deepcopy
import math
import torch
from .config import ModuleConfig,EVENT_DIM,OPTION_DIM,STATE_DIM,DIAG_DIM,SUMMARY_DIM
from bci_vla.interaction_contract_v4.validation import validate_document,_diag_features,canonical_digest
from bci_vla.data_v4.contracts import validate_public

SCORES=['RAW_SCORE','CALIBRATED_PROBABILITY','HARD_LABEL','NONE']
ROLES=['TASK','INTENT_QUERY','CONFIRM','REJECT','NO_RESPONSE']
ORIGINAL=['TASK','INTENT_QUERY','PROBE','CONFIRM']
KINDS=['OBJECT','TASK','DIAGNOSTIC','DECISION']

def bounded_score(value,kind):
 return math.tanh(value) if kind=='RAW_SCORE' else value

def collate_public(records,config=None):
 c=config or ModuleConfig();b=len(records);n=c.max_objects;h=c.history_events;m=16;o=c.task_options
 if not b:raise ValueError('empty batch is not an empty observation')
 def zeros(*shape,dtype=torch.float32):return torch.zeros(*shape,dtype=dtype)
 x={'objects':zeros(b,n,c.roi_dim+c.semantic_dim+6),'selection':zeros(b,n,7),'candidate_mask':zeros(b,n,dtype=torch.bool),
  'option_src':zeros(b,h,m,n),'option_dst':zeros(b,h,m,n),'option_numeric':zeros(b,h,m,OPTION_DIM),'option_mask':zeros(b,h,m,dtype=torch.bool),
  'event_src':zeros(b,h,n),'event_dst':zeros(b,h,n),'event_numeric':zeros(b,h,EVENT_DIM),'event_mask':zeros(b,h,dtype=torch.bool),
  'task_src':zeros(b,o,n),'task_dst':zeros(b,o,n),'task_numeric':zeros(b,o,OPTION_DIM),'task_mask':zeros(b,o,dtype=torch.bool),
  'state_src':zeros(b,3,n),'state_dst':zeros(b,3,n),'state_numeric':zeros(b,3,STATE_DIM),'state_mask':zeros(b,3,dtype=torch.bool),
  'pending_src':zeros(b,m,n),'pending_dst':zeros(b,m,n),'pending_numeric':zeros(b,m,OPTION_DIM),'pending_mask':zeros(b,m,dtype=torch.bool),
  'diag_history':zeros(b,h,DIAG_DIM),'diag_mask':zeros(b,h,dtype=torch.bool),'diag_summary':zeros(b,SUMMARY_DIM),
  'stop':zeros(b,dtype=torch.bool),'pending':zeros(b,dtype=torch.bool)}
 metadata=[]
 for bi,p in enumerate(records):
  if p.get('schema_version')=='bci-vla-interaction/4.1':validate_public(p)
  else:validate_document(p)
  if p['scene']['roi_dim']!=c.roi_dim or p['scene']['semantic_dim']!=c.semantic_dim:raise ValueError('feature dimensions must match frozen profile')
  objects=p['scene']['objects'];indices={v['instance_id']:i for i,v in enumerate(objects)}
  if len(objects)>n or len(p['history'])>h:raise ValueError('no silent tensor truncation')
  task_lookup={v['task_id']:v for v in p['task_options']}
  def pointer(dst,oid):
   if oid is not None and oid in indices:dst[indices[oid]]=1.
  def task_link(source,dest,numeric,task):
   pointer(source,task['source_id']);pointer(dest,task['destination_id']);numeric[7+['pick','place'].index(task['action'])]=1.
  def option_into(source,dest,num,opt,score=None,kind='NONE',selected=False):
   num[3+KINDS.index(opt['kind'])]=1.;num[2]=float(selected)
   if score is not None:num[0]=bounded_score(score,kind);num[1]=1.
   if opt['kind']=='OBJECT':pointer(source,opt['reference'])
   elif opt['kind']=='TASK':task_link(source,dest,num,task_lookup[opt['reference']])
   elif opt['kind']=='DECISION':num[9+['accept','reject'].index(opt['reference'])]=1.
  for i,v in enumerate(objects):
   valid=v['object_valid'];x['candidate_mask'][bi,i]=valid
   if not valid:continue
   roi=v['roi_feature'] if v['roi_valid'] else [0.]*c.roi_dim
   sem=v['semantic_feature'] if v['semantic_valid'] else [0.]*c.semantic_dim
   vector=roi+sem+v['bbox_xyxy']+[float(v['roi_valid']),float(v['semantic_valid'])]
   if any(not math.isfinite(v) or abs(v)>1e4 for v in vector):raise ValueError('feature outside supported normalized range')
   x['objects'][bi,i]=torch.tensor(vector)
  state=p['public_state'];now=p['as_of_s'];ti=di=0
  for event in p['history']:
   if not event['valid'] or event['question'] is None:continue
   q=event['question'];e=event['evidence'];kind=e['score_kind']
   if q['role']=='PROBE':
    d=_diag_features(event,now);row=x['diag_history'][bi,di]
    row[SCORES.index(kind)]=1.;row[4]=float(d['known_target_match']);row[5]=float(d['no_response'])
    if d['known_target_score'] is not None:row[6]=bounded_score(d['known_target_score'],kind);row[7]=1.
    for j,value in enumerate(d['other_scores_sorted']):row[8+j]=bounded_score(value,kind);row[23+j]=1.
    row[38]=min(d['duration_s']/5.,4.);row[39]=min(d['age_s']/30.,4.)
    x['diag_mask'][bi,di]=True;di+=1;continue
   row=x['event_numeric'][bi,ti];row[ROLES.index(event['role'])]=1.
   if event['original_role'] is not None:row[5+ORIGINAL.index(event['original_role'])]=1.
   row[9+SCORES.index(kind)]=1.;row[13]=min((now-event['ended_at_s'])/30.,4.);row[14]=min((event['ended_at_s']-event['started_at_s'])/5.,4.)
   if q['proposal'] is not None:
    t=q['proposal'];pointer(x['event_src'][bi,ti],t['source_id']);pointer(x['event_dst'][bi,ti],t['destination_id']);row[15+['pick','place'].index(t['action'])]=1.
   values={v['option_id']:v['value'] for v in e['values']}
   for j,opt in enumerate(q['options']):
    option_into(x['option_src'][bi,ti,j],x['option_dst'][bi,ti,j],x['option_numeric'][bi,ti,j],opt,values.get(opt['option_id']),kind,opt['option_id']==e['selected_option_id'])
    x['option_mask'][bi,ti,j]=True
   x['event_mask'][bi,ti]=True;ti+=1
   if event['event_id']==p['selection_event_id']:
    for opt in q['options']:
     i=indices[opt['reference']];v=x['selection'][bi,i];v[0]=bounded_score(values[opt['option_id']],kind);v[1]=1.;v[2]=float(opt['option_id']==e['selected_option_id']);v[3+SCORES.index(kind)]=1.
  for j,t in enumerate(p['task_options']):
   task_link(x['task_src'][bi,j],x['task_dst'][bi,j],x['task_numeric'][bi,j],t);x['task_numeric'][bi,j,4]=1.;x['task_mask'][bi,j]=True
  pending=state['pending_question']
  tasks=[None if state['confirmed_task'] is None else state['confirmed_task']['task'],state['pending_suggestion'],None if pending is None else pending['proposal']]
  for j,t in enumerate(tasks):
   row=x['state_numeric'][bi,j];row[10+j]=1.
   if t is not None:
    pointer(x['state_src'][bi,j],t['source_id']);pointer(x['state_dst'][bi,j],t['destination_id']);row[0]=1.;row[1+['pick','place'].index(t['action'])]=1.;x['state_mask'][bi,j]=True
  row=x['state_numeric'][bi,2];row[9]=float(state['stop_latched']);x['state_mask'][bi,2]=True
  if pending is not None:
   row[3+['TASK','INTENT_QUERY','PROBE','CONFIRM'].index(pending['role'])]=1.;row[8]=min((pending['expires_at_s']-now)/30.,4.)
   if pending['role']!='PROBE':
    for j,opt in enumerate(pending['options']):option_into(x['pending_src'][bi,j],x['pending_dst'][bi,j],x['pending_numeric'][bi,j],opt);x['pending_mask'][bi,j]=True
  d=state['diagnostics'];age=30. if d['latest_ended_at_s'] is None else now-d['latest_ended_at_s']
  x['diag_summary'][bi]=torch.tensor([d['attempt_count']/64.,d['correct_count']/64.,d['no_response_count']/64.,len(d['covered_symbols'])/16.,min(age/30.,4.),float(d['truncated']),float(d['latest_ended_at_s'] is not None),float(d['status']=='SUFFICIENT'),float(d['status']=='EXPIRED'),float(d['status']=='INSUFFICIENT')])
  x['stop'][bi]=state['stop_latched'];x['pending'][bi]=pending is not None
  metadata.append({'input_sha256':canonical_digest(p),'sample_id':p['sample_id'],'instance_ids':[v['instance_id'] for v in objects],'aliases':[v['alias'] for v in objects],'source':'public input binding only; not model features'})
 return x,metadata
