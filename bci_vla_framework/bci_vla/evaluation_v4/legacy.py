"""Use the actual legacy JointBeliefFilter on its binary applicable slice.

The V2 event bridge is belief-only; V4 submission/confirmation checks are separate.
No old robot authorization, empirical robot consequence or full planner is claimed.
"""
from copy import deepcopy
from pathlib import Path
import json
from datetime import datetime,timedelta,timezone
from bci_vla.core.joint_belief import GoalVocabulary,JointBeliefFilter,TaskQuestionBinding
from bci_vla.models.channel import ChannelModel,JointTransitionModel
from bci_vla.protocols import Event,JointBelief
from bci_vla.adapters.bci.replay import IngestResult
from bci_vla.scene.tasks import goal_id
ROOT=Path(__file__).resolve().parents[2]
def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))
def stamp(t):return (datetime(2026,10,17,tzinfo=timezone.utc)+timedelta(seconds=t)).isoformat()
class LegacyBinary:
 def __init__(self,public):
  self.ids=[o['instance_id'] for o in public['scene']['objects'] if o['object_valid']]
  if len(self.ids)!=2:raise ValueError('legacy_binary_scope_requires_exactly_two_candidates')
  self.goals=[goal_id({'action':'pick','source_id':oid,'destination_id':None}) for oid in self.ids]
  ctx=public['context'];self.context={'session_id':ctx['session_id'],'task_id':'binary-interaction','task_revision':1,'scene_id':ctx['scene_id'],'scene_revision':ctx['scene_revision'],'mapping_id':ctx['mapping_id'],'stimulus_version':'mapping-'+str(ctx['mapping_revision']),'channel_profile_id':'v4-analytic-simulation-binary'}
  vocab=GoalVocabulary({'schema_version':'goal-vocabulary/1.0','scene_id':ctx['scene_id'],'scene_revision':ctx['scene_revision'],'goals':[{'goal_id':g,'kind':'task','intent':{'action':'pick','source_id':oid,'destination_id':None}} for g,oid in zip(self.goals,self.ids)]+[{'goal_id':v,'kind':v,'intent':None} for v in ['unknown','none']]})
  c=read('configs/channels/simulation_binary_v1.json');c.update(profile_id=self.context['channel_profile_id'],mode_ids=['stable','degraded'])
  c['scope'].update(mapping_id=ctx['mapping_id'],stimulus_version=self.context['stimulus_version'],layout_id='binary-layout',duration_s=[1.])
  c['matrices']={mode:{z:{'1.0':{u:{u:(1-nr)*p,str(1-int(u)):(1-nr)*(1-p),'NO_RESPONSE':nr} for u in ['0','1']}} for z,p,nr in [('stable',.9,.03),('degraded',.35,.25)]} for mode in ['TASK_QUERY','CHANNEL_PROBE','CONFIRM']}
  c['compliance']={'known_target':{u:{u:1.,str(1-int(u)):0.,'NO_RESPONSE':0.,'UNKNOWN':0.} for u in ['0','1']}}
  tr=read('configs/belief/simulation_canonical_transition.json');tr.update(goal_ids=self.goals,mode_ids=['stable','degraded'])
  self.filter=JointBeliefFilter(JointBelief({'goal_ids':self.goals+['unknown','none'],'mode_ids':['stable','degraded','unknown_channel'],'mass':[{'goal_id':g,'mode_id':z,'probability':.25} for g in self.goals for z in ['stable','degraded']],'as_of_event_id':'uniform_public_prior'}),vocabulary=vocab,channel_model=ChannelModel(c),transition_model=JointTransitionModel(tr),current_context=self.context,layout_id='binary-layout',initial_at=datetime.fromisoformat(stamp(public['as_of_s'])))
  self.binding=None;self.seen=set();self.updates=[]
 def observe(self,event):
  if event['event_id'] in self.seen:return
  self.seen.add(event['event_id'])
  if event['question'] is None or not event['valid']:return
  q4=event['question'];role=q4['role'];mode='TASK_QUERY' if role in ['TASK','INTENT_QUERY'] else 'CHANNEL_PROBE' if role=='PROBE' else 'CONFIRM'
  q=read('examples/protocols/'+('channel_probe_request.json' if mode=='CHANNEL_PROBE' else 'confirm_request.json' if mode=='CONFIRM' else 'task_query_request.json'))
  q.update(self.context,event_id='legacy-'+event['event_id'],interaction_id='legacy-i-'+event['event_id'],sequence=event['sequence'],sent_at=stamp(q4['issued_at_s']),expires_at=stamp(q4['expires_at_s']))
  q['query_id']=q['event_id'] if mode!='CHANNEL_PROBE' else None;q['probe_id']=q['event_id'] if mode=='CHANNEL_PROBE' else None
  if mode=='CONFIRM':
   options=[{'option_id':o['option_id'],'symbol':str(i),'semantic_ref':'confirmation:'+o['reference']} for i,o in enumerate(q4['options'])]
   proposal=q4['proposal'];q['payload']['contract']={'contract_id':'legacy-contract-'+event['event_id'],'task_id':self.context['task_id'],'revision':1,'scene_id':self.context['scene_id'],'scene_revision':self.context['scene_revision'],'intent':{'action':'pick','source_id':proposal['source_id'],'destination_id':None},'status':'proposed','valid_until':stamp(proposal['valid_until_s']),'confirmation_event_id':None}
  else:options=[{'option_id':o['option_id'],'symbol':str(i),'semantic_ref':('probe:'+str(i)) if mode=='CHANNEL_PROBE' else 'goal:'+self.goals[self.ids.index(o['reference'])]} for i,o in enumerate(q4['options'])]
  q['payload'].update(question='Select the intended LIBERO object' if mode=='TASK_QUERY' else 'Check the highlighted symbol' if mode=='CHANNEL_PROBE' else 'Accept the proposed pick?',options=options)
  if mode=='CHANNEL_PROBE':
   target=next(str(i) for i,o in enumerate(q4['options']) if o['option_id']==q4['diagnostic_option_id'])
   q['payload']['probe_prompt']={'symbol':target,'source':'display_controller','presented_at':{'host_id':'ui','at':stamp(q4['issued_at_s']),'offset_ms':0,'uncertainty_ms':0},'presentation_ack':True}
  binding=TaskQuestionBinding({'question':q['payload']['question'],'mapping_id':q['mapping_id'],'stimulus_version':q['stimulus_version'],'options':options},self.filter.vocabulary) if mode=='TASK_QUERY' else None
  r=read('examples/protocols/task_query_response.json');r.update(self.context,event_id='response-'+event['event_id'],sequence=event['sequence'],mode=mode,interaction_id=q['interaction_id'],query_id=q['query_id'],probe_id=q['probe_id'],response_to=q['event_id'],sent_at=stamp(event['ended_at_s']),expires_at=q['expires_at'])
  r['expires_at']=stamp(event['ended_at_s']+2.)  # response-message TTL, distinct from question deadline
  no=event['role']=='NO_RESPONSE';selected=event['evidence']['selected_option_id']
  symbol=None if no else next(str(i) for i,o in enumerate(q4['options']) if o['option_id']==selected)
  r['payload'].update(status='NO_RESPONSE' if no else 'OBSERVATION',symbol=symbol,raw_scores=None,score_kind=None,calibration_id=None)
  if no:r['payload']['sample_window']=None
  else:r['payload']['sample_window'].update(window_id='window-'+event['event_id'],recording_id='simulated-channel-on-LIBERO-capture',sample_start=event['sequence']*1000,sample_end=event['sequence']*1000+250,started_at=stamp(q4['issued_at_s']+.25),ended_at=stamp(q4['issued_at_s']+1.25))
  result=IngestResult(True,'synthetic_categorical_bridge',Event.parse(r),Event.parse(q),{'TASK_QUERY':'TASK','CHANNEL_PROBE':'PROBE','CONFIRM':'CONFIRM'}[mode])
  update=self.filter.update(result,binding=binding,compliance_id='known_target' if mode=='CHANNEL_PROBE' else None,assumption='state_constant_within_window',finalized=True,nonresponse_window={'start':stamp(q4['issued_at_s']+.25),'end':stamp(q4['issued_at_s']+1.25),'clock_domain':'reference'} if no else None)
  self.updates.append(update)
 def estimate(self):
  snapshot=self.filter.snapshot()
  scores={oid:snapshot['goal_marginals'][g] for oid,g in zip(self.ids,self.goals)}
  return scores,snapshot
