"""Public-clock trajectories and method-independent, question-keyed noise."""
from copy import deepcopy
import hashlib,json,math,random

def digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def question_key(question):
 ctx=question['context'];proposal=question['proposal']
 normalized_context={k:ctx[k] for k in ['scene_id','scene_revision','mapping_id','mapping_revision']}
 known=next((o['symbol'] for o in question['options'] if o['option_id']==question['diagnostic_option_id']),None)
 return {'role':question['role'],'context':normalized_context,'layout':question['layout_id'],
  'options':sorted([(o['symbol'],o['kind'],o['reference']) for o in question['options']]),
  'known_target':known,'proposal':None if proposal is None else {k:proposal[k] for k in ['action','source_id','destination_id']}}

def noise_draw(episode_id,at_s,question,repeat=0):
 # No method name, policy, number of earlier questions, split or latent fault label.
 seed=digest(['channel-v1',episode_id,int(math.floor(at_s/2)),question_key(question),repeat])
 rng=random.Random(int(seed,16))
 return rng.random(),rng.random(),rng.random(),rng

def world_at(world,at_s):
 changed=at_s>=world['change_at_s']
 return (world['later_goal'] if changed else world['initial_goal'],world['later_success'] if changed else world['initial_success'])

def answer(world,question,at_s,repeat=0,force_no_response=False):
 goal,p=world_at(world,at_s);options=sorted(question['options'],key=lambda o:o['symbol']);role=question['role']
 if role=='PROBE':intended=question['diagnostic_option_id']
 elif role=='CONFIRM':
  decision='accept' if question['proposal']['source_id']==goal else 'reject'
  intended=next(o['option_id'] for o in options if o['reference']==decision)
 else:intended=next((o['option_id'] for o in options if o['reference']==goal),None)
 no,success,error,rng=noise_draw(world['episode_id'],at_s,question,repeat)
 if force_no_response or intended is None or no>(.97 if p>.7 else .75):
  return {'score_kind':'NONE','values':[],'selected_option_id':None,'calibration':None}
 other=[o['option_id'] for o in options if o['option_id']!=intended]
 selected=intended if success<p or not other else other[min(len(other)-1,int(error*len(other)))]
 return {'score_kind':'RAW_SCORE','values':[{'option_id':o['option_id'],'value':rng.gauss(1.5 if o['option_id']==selected else .0,.35)} for o in options], 'selected_option_id':selected,'calibration':None}

def diagnostic_summary(context,history,now,layout):
 rows=[deepcopy(e) for e in history if e['valid'] and e['question'] is not None and e['question']['role']=='PROBE' and e['question']['layout_id']==layout and 0<=now-e['ended_at_s']<=30]
 covered=sorted({next(o['symbol'] for o in e['question']['options'] if o['option_id']==e['question']['diagnostic_option_id']) for e in rows})
 latest=max((e['ended_at_s'] for e in rows),default=None)
 correct=sum(e['role']!='NO_RESPONSE' and e['evidence']['selected_option_id']==e['question']['diagnostic_option_id'] for e in rows)
 return {'context':deepcopy(context),'layout_id':layout,'window_s':30,'ttl_s':15,'min_attempts':3,'min_covered_symbols':2,'receipts':rows[-64:], 'attempt_count':len(rows[-64:]),'correct_count':int(correct),'no_response_count':sum(e['role']=='NO_RESPONSE' for e in rows),'covered_symbols':covered,'latest_ended_at_s':latest,'status':'EXPIRED' if latest is not None and now-latest>15 else 'SUFFICIENT' if len(rows)>=3 and len(covered)>=2 else 'INSUFFICIENT','truncated':False}
