"""Policies receive only current public information, never latent world/scorer."""
from .legacy import LegacyBinary
def fixed_inquiry(public,scores,inference_ready=True):
 """Shared MVP rule for future neural estimates and the binary reference.

 Scores are uncalibrated model belief outputs; .8 is a development default.
 This rule is fixed code, not a learned inquiry policy.
 """
 import math
 if public['public_state']['stop_latched'] or public['public_state']['pending_question'] is not None:return {'action':'WAIT'}
 if not inference_ready:return {'action':'WAIT','reason':'inference_not_ready'}
 ids={o['instance_id'] for o in public['scene']['objects'] if o['object_valid']}
 if not ids:return {'action':'WAIT'}
 if set(scores)!=ids or any(not math.isfinite(v) or v<0 or v>1 for v in scores.values()) or not math.isclose(sum(scores.values()),1.,abs_tol=1e-6):raise ValueError('candidate_distribution_axis_or_mass')
 if public['public_state']['diagnostics']['status']!='SUFFICIENT':return {'action':'PROBE'}
 events=[e for e in public['history'] if e['valid'] and e['role'] in ['TASK','INTENT_QUERY']]
 if not events or public['as_of_s']-events[-1]['ended_at_s']>=4:return {'action':'TASK'}
 source=max(scores,key=scores.get)
 if scores[source]<.8:return {'action':'INTENT_QUERY'}
 current=public['public_state']['confirmed_task']
 if current and current['task']['source_id']==source:return {'action':'TASK'}
 return {'action':'CONFIRM','source_id':source}
class RecentChoice:
 name='B0'
 def __init__(self):self.legacy=None
 def decide(self,p):
  if p['public_state']['stop_latched'] or p['public_state']['pending_question'] is not None:return {'action':'WAIT'}
  candidates=[o['instance_id'] for o in p['scene']['objects'] if o['object_valid']]
  if not candidates:return {'action':'WAIT'}
  events=[e for e in p['history'] if e['valid'] and e['role'] in ['TASK','INTENT_QUERY']]
  if not events or p['as_of_s']-events[-1]['ended_at_s']>=4:return {'action':'TASK'}
  e=events[-1];scores=sorted((v['value'] for v in e['evidence']['values']),reverse=True)
  if len(scores)>1 and scores[0]-scores[1]<.5:return {'action':'INTENT_QUERY'}
  source=next(o['reference'] for o in e['question']['options'] if o['option_id']==e['evidence']['selected_option_id'])
  current=p['public_state']['confirmed_task']
  if current and current['task']['source_id']==source:return {'action':'TASK'}
  return {'action':'CONFIRM','source_id':source}
class LegacyReference(RecentChoice):
 name='B1_binary_filter_fixed_policy'
 def decide(self,p):
  if p['public_state']['stop_latched'] or p['public_state']['pending_question'] is not None:return {'action':'WAIT'}
  context=p['context']
  if self.legacy is None or self.legacy.context['stimulus_version']!='mapping-'+str(context['mapping_revision']):self.legacy=LegacyBinary(p)
  for e in p['history']:self.legacy.observe(e)
  scores,snapshot=self.legacy.estimate()
  return fixed_inquiry(p,scores,snapshot['point_inference_ready'])
