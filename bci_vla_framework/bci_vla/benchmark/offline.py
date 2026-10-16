"""Public-only controller and fixtures for the step12 offline acceptance run.

The controller receives no schedule, factor label, seed, goal or channel truth.
It selects with the frozen step10 methods and executes through step11 checks.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
from time import perf_counter_ns

from ..protocols import JointBelief, ProtocolError, thaw, require
from ..models.channel import ChannelModel, JointTransitionModel
from ..models.consequence import OutcomeTable
from ..scene.state import GroundedScene
from ..core.joint_belief import GoalVocabulary, JointBeliefFilter
from ..core.diagnostic_policy import DiagnosticPolicy, PlanningState
from ..baselines.finite_model import FiniteModel
from ..baselines.methods import ComparisonSuite
from ..adapters.vla import FakeVLAAdapter
from ..contracts.session import InteractionSession

ROOT=Path(__file__).resolve().parents[2]
BASE=datetime(2026,10,15,tzinfo=timezone.utc)
METHODS=('B4','B5','B7','B9','M','M2')
def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))
def stamp(seconds):return (BASE+timedelta(seconds=seconds)).isoformat()


def make_session(public_id,*,prior=None,probe_compliance='imperfect',fixed_transition=False):
    f=read('configs/belief/simulation_binary_filter.json')
    f['current_context']['session_id']=public_id
    vocabulary=GoalVocabulary(f['vocabulary']);initial=deepcopy(f['initial_joint'])
    weights=prior if prior is not None else (.81,.09,.09,.01)
    initial['mass']=[{'goal_id':g,'mode_id':z,'probability':p}
        for (g,z),p in zip(((g,z) for g in vocabulary.known_goal_ids for z in ('normal','flip')),weights)]
    channel=ChannelModel(read('configs/channels/simulation_binary_v1.json'))
    transition=JointTransitionModel(read('examples/joint_belief/fixed_state_transition.json' if fixed_transition
        else 'configs/belief/simulation_canonical_transition.json'))
    filt=JointBeliefFilter(JointBelief(initial),vocabulary=vocabulary,channel_model=channel,transition_model=transition,
        current_context=f['current_context'],layout_id=f['layout_id'],initial_at=BASE)
    pc=read('configs/diagnosis.json');pc['context']=dict(f['current_context']);pc['actions']['compliance_id']=probe_compliance
    policy=DiagnosticPolicy(pc,channel_model=channel,transition_model=transition,
        outcome_table=OutcomeTable(read('configs/consequences/simulation_canonical_outcomes.json')))
    scene=read('examples/scene_outcomes/scene.json');scene['identity']['session_id']=public_id
    config=read('configs/execution/session.json');config['probe_compliance_id']=probe_compliance
    return InteractionSession(current_context=f['current_context'],scene=GroundedScene(scene),belief_filter=filt,
        policy=policy,backend=FakeVLAAdapter(),config=config,started_at=BASE)


def observation(now):
    obs=read('configs/execution/fake_vla_interface.json')['fixture_observation'];obs['captured_at']=stamp(now)
    return obs


def response(request,symbol,*,start,duration,index):
    q=request.to_dict();r=read('examples/protocols/confirm_response.json')
    for k in ('session_id','task_id','task_revision','scene_id','scene_revision','mapping_id','stimulus_version',
              'channel_profile_id','mode','interaction_id','query_id','probe_id'):r[k]=q[k]
    end=start+duration
    r.update(event_id='sample-'+str(index),response_to=q['event_id'],sequence=index,host_id='bci',
        sent_at=stamp(end),expires_at=q['expires_at'],clock_offset_ms=0,clock_uncertainty_ms=0)
    r['payload'].update(status='NO_RESPONSE' if symbol=='NO_RESPONSE' else 'OBSERVATION',
        symbol=None if symbol=='NO_RESPONSE' else symbol)
    if symbol=='NO_RESPONSE':r['payload'].update(sample_window=None,raw_scores=None,score_kind=None,calibration_id=None)
    else:r['payload']['sample_window'].update(window_id='window-'+str(index),recording_id='software-public-recording',
        sample_start=index*10000,sample_end=index*10000+int(duration*250),started_at=stamp(start),ended_at=stamp(end),
        source_kind='offline_replay',sample_rate_hz=250)
    return r


class OfflineController:
    """The online half: only public session and method configuration enter."""
    def __init__(self,session,method,*,budget_s=20,diagnostic_limit=2,compute_budget_ns=5_000_000_000):
        require(method in METHODS,'unsupported_offline_method')
        require(type(diagnostic_limit) is int and diagnostic_limit>=0,'invalid_diagnostic_limit')
        require(type(compute_budget_ns) is int and compute_budget_ns>=0,'invalid_compute_budget')
        self.session=session;self.method=method;self.budget_s=budget_s;self.limit=diagnostic_limit
        self.compute_budget_ns=compute_budget_ns;self.model=FiniteModel(session.policy)
        self.suite=ComparisonSuite(self.model,read('configs/comparisons/parameters.json'))
        self.contracts=read('examples/diagnosis/state.json')['contracts']
        self.memory={'last_confirmed_goal':self.model.goals[0],'diagnostic_count':0}
        self.decisions=[];self.diagnostics=0;self.now=0.;self.index=0;self.intervals=[];self.termination=None

    def planning_state(self):
        s=self.session;snap=s.filter.snapshot()
        gap=self.now-(datetime.fromisoformat(snap['as_of_reference_time'])-BASE).total_seconds()
        belief=self.model.joint(self.model.predict(self.model.vector(JointBelief(snap['belief'])),max(0,gap)))
        return PlanningState(belief=belief,scene_id=s.context['scene_id'],scene_revision=s.context['scene_revision'],
            task_id=s.context['task_id'],task_revision=s.context['task_revision'],now_s=BASE.timestamp()+self.now,
            used_s=self.now,remaining_s=max(0,self.budget_s-self.now),scene_valid_until_s=BASE.timestamp()+60,
            contracts=self.contracts,stopped=s.stop_latched,pending=s.adapter.pending_request is not None,
            inference_ready=snap['point_inference_ready'])

    def choose(self):
        state=self.planning_state();start=perf_counter_ns()
        decision=self.suite.choose(self.method,state,memory=self.memory,horizon_remaining=max(0,self.limit-self.diagnostics))
        elapsed=perf_counter_ns()-start
        decision['actual_call_compute_ns']=elapsed
        decision['public_input']={'belief':state.belief.to_dict(),'now_s':self.now,
            'remaining_s':state.remaining_s,'memory':deepcopy(self.memory),'horizon_remaining':max(0,self.limit-self.diagnostics)}
        self.decisions.append(decision)
        self.intervals.append({'phase':'policy_compute','start_s':self.now,'end_s':self.now+elapsed/1e9,'user_busy':False})
        self.now=round(self.now+elapsed/1e9,6)
        if elapsed>self.compute_budget_ns:
            self.stop('policy_compute_budget_exceeded');return None
        if self.now>=self.budget_s:
            self.stop('episode_deadline_exceeded');return None
        action=decision['selected_action']
        if action['kind'] in {'PAUSE','EXIT'}:
            self.stop('policy_'+action['kind'].lower());return None
        if self.now+action['elapsed_s']>self.budget_s:
            self.stop('action_would_exceed_deadline');return None
        return action

    def stop(self,reason):
        self.termination=reason
        if not self.session.stop_latched:self.session.request_stop(now=stamp(self.now))

    def begin(self,action):
        s=self.session;mode=action['kind']
        require(mode in {'TASK_QUERY','CHANNEL_PROBE','CONFIRM'},'unsupported_benchmark_action')
        if mode=='CONFIRM':
            s.propose(self.contracts[action['goal_id']],now=stamp(self.now))
        args={} if mode!='CHANNEL_PROBE' else {'prompt_symbol':action['prompt'],'presentation_ack':True,
            'presented_at':{'host_id':'ui','at':stamp(self.now),'offset_ms':0,'uncertainty_ms':0}}
        q=s.begin_interaction(mode,now=stamp(self.now),monotonic_now=self.now,
            expires_in_s=max(action['elapsed_s']+.1,min(30,self.budget_s-self.now)),**args)
        start=round(self.now+action['setup_s'],6);end=round(start+action['duration_s'],6)
        self.intervals.append({'phase':mode,'start_s':self.now,'end_s':end,'user_busy':True})
        self.index+=1
        return q,start,end

    def receive(self,q,symbol,action,start,end):
        self.now=end
        result=self.session.ingest(response(q,symbol,start=start,duration=action['duration_s'],index=self.index),
            now=stamp(end),monotonic_now=end)
        require(result.accepted,'benchmark_generated_input_rejected:'+result.reason)
        if action['kind']!='CONFIRM':
            self.diagnostics+=int(self.index>1)
            self.memory['diagnostic_count']=self.diagnostics
        if self.session.state=='PAUSED':self.termination='response_or_inference_paused'
        return self.session.state

    def submit(self):
        self.now=round(self.now+.01,6)
        self.intervals.append({'phase':'return_authorization_dispatch','start_s':self.now-.01,'end_s':self.now,'user_busy':False})
        try:
            self.session.authorize(now=stamp(self.now))
            self.session.dispatch('command-1',observation(self.now),now=stamp(self.now))
        except ProtocolError as exc:
            self.stop('authorization_or_dispatch_blocked:'+str(exc));return False
        return True


def union_duration(intervals):
    spans=sorted((r['start_s'],r['end_s']) for r in intervals)
    if not spans:return 0.
    total=0.;left,right=spans[0]
    for a,b in spans[1:]:
        if a>right:total+=right-left;left,right=a,b
        else:right=max(right,b)
    return total+right-left
