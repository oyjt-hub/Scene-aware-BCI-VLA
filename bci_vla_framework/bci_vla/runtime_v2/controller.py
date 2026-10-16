"""All methods use one versioned authorization/time specification."""
from copy import deepcopy
from ..protocols import thaw,require,ProtocolError
from ..baselines.methods import ComparisonSuite
from ..benchmark.offline import OfflineController,read,stamp,observation
from .planning import AuthorizationAwarePolicy,AuthorizationAwareFiniteModel,VERSION


def gate_for(session,return_s=.01):
    return {'version':VERSION,'minimum_goal_probability':session.config['minimum_goal_probability'],
        'max_confirmation_age_s':session.config['max_confirmation_age_s'],'return_s':return_s}


class RuntimeController(OfflineController):
    def __init__(self,session,method,**kwargs):
        super().__init__(session,method,**kwargs)
        gate=gate_for(session)
        policy=AuthorizationAwarePolicy(thaw(session.policy.config),gate=gate,channel_model=session.policy.channel,
            transition_model=session.policy.transition,outcome_table=session.policy.outcomes)
        self.model=AuthorizationAwareFiniteModel(policy,gate)
        self.suite=ComparisonSuite(self.model,read('configs/comparisons/parameters.json'))
        # This is an external proposing planner. The unchanged session authorizer
        # still owns evidence and its own original policy/config fingerprints.
        self.planner_ref={'version':VERSION,'gate':gate,'role':'external_advisory_proposer_not_execution_authority'}

    def choose(self):
        action=super().choose()
        if self.decisions:self.decisions[-1]['planner_ref']=deepcopy(self.planner_ref)
        if action is None:return None
        current=self.planning_state()
        candidate=next(a for a in self.model.catalog.candidates() if a.action_id==action['action_id'])
        reason=self.model.catalog.legality(candidate,current)
        if reason:self.stop('post_compute_action_invalid:'+reason);return None
        if candidate.kind=='CONFIRM':
            # Computation has taken real time since planning. Recheck the exact
            # accepted observation branch from a fresh read-only public view.
            branches=self.model.branches(self.model.vector(current.belief),candidate)
            accepted=next((post for y,p,post in branches if y==self.model.config['actions']['accept_symbol']),None)
            p,allowed=(0.,False) if accepted is None else self.model.authorization_prediction(candidate.goal_id,accepted)
            self.decisions[-1]['post_compute_authorization_prediction']={'goal_probability':p,'allowed':allowed}
            if not allowed:self.stop('post_compute_confirmation_would_be_blocked');return None
        return action

    def submit(self):
        delay=self.model.gate['return_s'];start=self.now;self.now=round(self.now+delay,6)
        self.intervals.append({'phase':'return_authorization_dispatch','start_s':start,'end_s':self.now,'user_busy':False})
        try:
            self.session.authorize(now=stamp(self.now))
            self.session.dispatch('command-1',observation(self.now),now=stamp(self.now))
        except ProtocolError as exc:self.stop('authorization_or_dispatch_blocked:'+str(exc));return False
        return True
