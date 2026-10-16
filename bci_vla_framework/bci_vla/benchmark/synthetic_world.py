"""Evaluation-private finite worlds. Never supplied to an online controller."""
from copy import deepcopy
import hashlib

from ..protocols import thaw
from .offline import BASE, stamp, read
from ..contracts import EndpointEvidence

CASES=(
    {'id':'C00','goal_change':False,'channel_change':False},
    {'id':'C10','goal_change':True,'channel_change':False},
    {'id':'C01','goal_change':False,'channel_change':True},
    {'id':'C11_simultaneous','goal_change':True,'channel_change':True},
    {'id':'C11_goal_first','goal_change':True,'channel_change':True,'goal_at':.25,'channel_at':.75},
    {'id':'C11_channel_first','goal_change':True,'channel_change':True,'goal_at':.75,'channel_at':.25},
    {'id':'stable_sporadic_error','goal_change':False,'channel_change':False,'force_first_error':True},
    {'id':'wrong_strong_prior','goal_change':True,'channel_change':False,'prior':(.97,.01,.01,.01)},
    {'id':'probe_noncompliance','goal_change':False,'channel_change':True,'actual_compliance':'uninformative'},
    {'id':'uninformative_probe','goal_change':False,'channel_change':True,'public_compliance':'uninformative','actual_compliance':'uninformative'},
)


class PrivateWorld:
    def __init__(self,case,seed,channel,goals):
        self.case=deepcopy(case);self.seed=seed;self.channel=channel;self.goals=tuple(goals)
        self.draws=[];self.counts={}

    def truth_at(self,at):
        return (self.goals[int(self.case['goal_change'] and at>=self.case.get('goal_at',.5))],
            'flip' if self.case['channel_change'] and at>=self.case.get('channel_at',.5) else 'normal')

    def emit(self,request,*,start,duration):
        q=request.data;mode=q['mode'];goal,z=self.truth_at(start)
        # Every private change precedes the common initial task window; none
        # takes place inside an acquisition. Order before that window is unobserved.
        assert self.truth_at(start)==self.truth_at(start+duration)
        symbol_by_goal={self.goals[0]:'0',self.goals[1]:'1'}
        if mode=='CHANNEL_PROBE':intended=q['payload']['probe_prompt']['symbol']
        elif mode=='TASK_QUERY':intended=symbol_by_goal[goal]
        else:intended='0' if thaw(q['payload']['contract']['intent'])==read('examples/diagnosis/state.json')['contracts'][goal]['intent'] else '1'
        scope={'mapping_id':q['mapping_id'],'stimulus_version':q['stimulus_version'],
            'layout_id':'fixture-two-markers','sample_rate_hz':250}
        dist={y:self.channel.likelihoods(mode,intended,y,scope=scope,duration_s=duration,
            compliance_id=self.case.get('actual_compliance','imperfect') if mode=='CHANNEL_PROBE' else None,
            assumption='state_constant_within_window')[z] for y in ('0','1','NO_RESPONSE')}
        self.counts[mode]=self.counts.get(mode,0)+1
        key=f'{self.seed}|{mode}|{self.counts[mode]}|{duration}'
        draw=int(hashlib.sha256(key.encode()).hexdigest()[:16],16)/2**64
        accum=0.;output='NO_RESPONSE'
        for y,p in dist.items():
            accum+=p
            if draw<accum:output=y;break
        forced=self.case.get('force_first_error',False) and mode=='TASK_QUERY' and self.counts[mode]==1
        if forced:output='1' # stable old/normal, deliberately isolated decoder error
        self.draws.append({'mode':mode,'draw':draw,'output':output,'forced_sporadic_error':forced,
            'window_start_s':start,'goal':goal,'channel':z,'distribution':dist})
        return output

    def endpoint(self,session,observed):
        p=session.submission_contract.data['proposal'];tr=read('examples/scene_outcomes/trajectory.json')
        tr.update(trajectory_id='synthetic-endpoint',split='test',source_ref='L1-independent-command-outcome-generator',
            commanded_intent=thaw(p['intent']))
        for row in tr['final_predicates']:row['value']=row['object_id']==p['intent']['destination_id']
        return EndpointEvidence({'evidence_id':'synthetic-evidence','session_id':session.context['session_id'],
            'submission_id':'command-1','contract_id':p['contract_id'],'contract_revision':p['revision'],
            'scene_id':p['scene_id'],'scene_revision':p['scene_revision'],'source':'simulation_fixture_annotation',
            'observed_at':{'host_id':'fixture-annotator','at':stamp(observed),'offset_ms':0,'uncertainty_ms':0},'trajectory':tr})

    def record(self):
        return {'case':self.case,'seed':self.seed,'scope':'evaluation_private_L1_synthetic',
            'initial_goal':self.goals[0],'initial_channel':'normal','draws':deepcopy(self.draws),
            'change_exposure':'all_changes_before_common_initial_window; ordering_not_observed_in_windows'}
