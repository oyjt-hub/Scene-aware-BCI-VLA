"""Private, time-varying evaluation world with auditable external draws."""
from copy import deepcopy
import hashlib
from ..benchmark.synthetic_world import PrivateWorld
from ..benchmark.offline import read
from ..protocols import thaw

CASES=[
 dict(id='C00',goal_change=False,channel_change=False),
 dict(id='C10',goal_change=True,channel_change=False),
 dict(id='C01',goal_change=False,channel_change=True),
 dict(id='C11',goal_change=True,channel_change=True),
 dict(id='goal_first_observable',goal_change=True,channel_change=True,goal_at=.5,channel_at=2.5),
 dict(id='channel_first_observable',goal_change=True,channel_change=True,goal_at=2.5,channel_at=.5),
 dict(id='wrong_strong_prior',goal_change=True,channel_change=False,prior=[.97,.01,.01,.01]),
 dict(id='probe_noncompliance',goal_change=False,channel_change=True,actual_compliance='uninformative'),
 dict(id='public_uninformative_probe',goal_change=False,channel_change=True,actual_compliance='uninformative',public_compliance='uninformative'),
 dict(id='stable_sporadic_error',goal_change=False,channel_change=False,force_first_error=True),
 dict(id='severity_uniform20',goal_change=False,channel_change=True,contamination=.2),
 dict(id='within_window_change',goal_change=True,channel_change=False,goal_at=1.5),
 dict(id='frequent_reversal',goal_change=True,channel_change=True,goal_times=[.5,2.5,4.5],channel_times=[.5,4.5]),
 dict(id='executor_failure35',goal_change=False,channel_change=False,execution_failure_probability=.35),
]

class StudyWorld(PrivateWorld):
 def times(self,axis):
  return self.case.get(axis+'_times',[self.case.get(axis+'_at',.5)] if self.case[axis+'_change'] else [])
 def truth_at(self,at):
  return self.goals[sum(t<=at for t in self.times('goal'))%2],('flip' if sum(t<=at for t in self.times('channel'))%2 else 'normal')
 def draw(self,key):return int(hashlib.sha256(f'{self.seed}|{key}'.encode()).hexdigest()[:16],16)/2**64
 def emit(self,request,*,start,duration):
  q=request.data;mode=q['mode'];goal,z=self.truth_at(start)
  self.counts[mode]=self.counts.get(mode,0)+1
  crossing=any(start<t<=start+duration for a in ['goal','channel'] for t in self.times(a))
  intended=(q['payload']['probe_prompt']['symbol'] if mode=='CHANNEL_PROBE' else str(self.goals.index(goal)) if mode=='TASK_QUERY' else
   '0' if thaw(q['payload']['contract']['intent'])==read('examples/diagnosis/state.json')['contracts'][goal]['intent'] else '1')
  scope=dict(mapping_id=q['mapping_id'],stimulus_version=q['stimulus_version'],layout_id='fixture-two-markers',sample_rate_hz=250)
  dist={y:self.channel.likelihoods(mode,intended,y,scope=scope,duration_s=duration,
   compliance_id=self.case.get('actual_compliance','imperfect') if mode=='CHANNEL_PROBE' else None,
   assumption='state_constant_within_window')[z] for y in ['0','1','NO_RESPONSE']}
  epsilon=self.case.get('contamination',0)
  dist={y:(1-epsilon)*p+epsilon/3 for y,p in dist.items()}
  draw=self.draw(f'{mode}|{self.counts[mode]}|{duration}');total=0;output='NO_RESPONSE'
  for y,p in dist.items():
   total+=p
   if draw<total:output=y;break
  forced=self.case.get('force_first_error',False) and mode=='TASK_QUERY' and self.counts[mode]==1
  if forced:output='1'
  # Explicit invalid-window suppression, not an accuracy claim about real EEG.
  if crossing:output='NO_RESPONSE'
  self.draws.append(dict(mode=mode,draw=draw,output=output,goal=goal,channel=z,window_start_s=start,duration_s=duration,
   distribution=dist,invalid_nonstationary_window_suppressed=crossing,forced_sporadic_error=forced))
  return output
 def endpoint(self,session,observed):
  from ..contracts import EndpointEvidence
  evidence=super().endpoint(session,observed).to_dict()
  failure=self.draw('executor-failure')<self.case.get('execution_failure_probability',0)
  if failure:
   for p in evidence['trajectory']['final_predicates']:p['value']=False
  return EndpointEvidence(evidence)
 def record(self):
  return dict(case=deepcopy(self.case),seed=self.seed,scope='private_synthetic_not_real_BCI',draws=deepcopy(self.draws),
   goal_change_times=self.times('goal'),channel_change_times=self.times('channel'),
   executor_failure_draw=self.draw('executor-failure'),within_window_change_handling='suppressed as NO_RESPONSE')
