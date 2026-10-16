"""Align hypothetical terminal branches with the unchanged step11 authorizer.

The candidate and independent O1 share the declared gate specification, but
implement its algebra separately. This is an engineering fix to standard VoI.
"""
import math
from copy import deepcopy

from ..protocols import freeze,require,number
from ..core.diagnostic_policy import DiagnosticPolicy
from ..core.diagnostic_actions import DiagnosticAction
from ..baselines.finite_model import FiniteModel

VERSION='authorization-aware-finite/2.0'


class ReturnAction(DiagnosticAction):
    def __init__(self,action,return_s):
        for key in action.__dataclass_fields__:object.__setattr__(self,key,getattr(action,key))
        object.__setattr__(self,'return_s',return_s)

    @property
    def elapsed_s(self):return self.setup_s+self.duration_s+self.return_s

    def record(self):return dict(super().record(),return_s=self.return_s)


class AuthorizationCatalog:
    def __init__(self,catalog,gate):self.base=catalog;self.gate=gate;self.config=catalog.config
    def candidates(self):
        return tuple(ReturnAction(a,self.gate['return_s'] if a.kind=='CONFIRM' else 0.) for a in self.base.candidates())
    def legality(self,a,state):
        reason=self.base.legality(a,state)
        if reason:return reason
        if a.kind=='CONFIRM' and a.return_s>self.gate['max_confirmation_age_s']:
            return 'confirmation_expires_before_authorization'
        return None


def checked_gate(config):
    c=deepcopy(config)
    require(set(c)=={'version','minimum_goal_probability','max_confirmation_age_s','return_s'},'invalid_gate_spec')
    require(c['version']==VERSION,'unsupported_gate_spec')
    number(c['minimum_goal_probability'],0,1);number(c['max_confirmation_age_s'],0);number(c['return_s'],0)
    return freeze(c)


class AuthorizationAwarePolicy(DiagnosticPolicy):
    def __init__(self,*args,gate,**kwargs):
        super().__init__(*args,**kwargs);self.gate=checked_gate(gate)
        self.catalog=AuthorizationCatalog(self.catalog,self.gate)

    def confirmation_value(self,target,posterior):
        # Candidate implementation: explicit CTMC matrix multiplication.
        dt=self.gate['return_s'];matrix=self.transition.matrix(dt)
        predicted=tuple(math.fsum(posterior[i]*matrix[i][j] for i in range(len(posterior))) for j in range(len(posterior)))
        p=math.fsum(mass for mass,(g,z) in zip(predicted,self.states) if g==target)
        allowed=dt<=self.gate['max_confirmation_age_s'] and p>=self.gate['minimum_goal_probability']
        value=math.fsum(mass*self._terminal_loss(target,g,z) for mass,(g,z) in zip(predicted,self.states)) if allowed else self.config['pause_loss']
        return value,{'authorized':allowed,'predicted_goal_probability':p,'prediction_gap_s':dt,
            'outcome':'execution_after_authorization' if allowed else 'pause_uncompleted_authorization_blocked'}

    def _evaluate(self,b,state,depth,*,diagnostics=True):
        # Adapted from the frozen09 Bellman evaluator; accepted branches now
        # pay pause loss when the actual authorization gate would reject them.
        candidates=[]
        for a in self.catalog.candidates():
            row=dict(a.record(),legal=False,reason=None,expected_loss=None,branches=[],
                immediate_time_cost=a.elapsed_s*self.catalog.config['time_weight_per_s'])
            reason=self.catalog.legality(a,state)
            if not reason and a.kind not in {'PAUSE','EXIT','CONFIRM'} and (depth==0 or not diagnostics):reason='diagnostic_horizon_exhausted'
            if reason:row['reason']=reason;candidates.append(row);continue
            if a.kind in {'PAUSE','EXIT'}:
                row.update(legal=True,expected_loss=self.config['pause_loss' if a.kind=='PAUSE' else 'exit_loss'])
                candidates.append(row);continue
            try:
                tv=self._tv(a)
                if tv is not None:
                    row['channel_total_variation']=tv
                    if tv<=1e-12:row['reason']='probe_cannot_distinguish_modes';candidates.append(row);continue
                branches=self._branches(b,a)
            except ValueError as exc:row['reason']=str(exc);candidates.append(row);continue
            after=state.advance(a.elapsed_s);expected=row['immediate_time_cost']
            for branch in branches:
                posterior=branch['posterior']
                if a.kind=='CONFIRM':
                    if branch['observation']==self.catalog.config['accept_symbol']:
                        value,gate=self.confirmation_value(a.goal_id,posterior);branch['authorization_gate']=gate
                        next_id='EXECUTION_REQUIRES_ACTUAL_BOUND_EVIDENCE' if gate['authorized'] else 'PAUSE_AUTHORIZATION_BLOCKED'
                    else:value,next_id=self.config['pause_loss'],'PAUSE_CONFIRMATION_NOT_ACCEPTED'
                elif branch['observation']=='NO_RESPONSE':value,next_id=self.config['pause_loss'],'PAUSE_NONRESPONSE'
                else:
                    best,_=self._evaluate(tuple(posterior),after,depth-1);value,next_id=best['expected_loss'],best['action_id']
                branch.update(continuation_loss=value,next_action=next_id,after_time_s=after.now_s,remaining_s=after.remaining_s)
                expected+=branch['probability']*value
            row.update(legal=True,expected_loss=expected,branches=branches,reason='authorization_aware_expected_loss')
            candidates.append(row)
        legal=[r for r in candidates if r['legal']]
        return min(legal,key=lambda r:(round(r['expected_loss'],12),r['kind']!='PAUSE',r['action_id'])),candidates


class AuthorizationAwareFiniteModel(FiniteModel):
    """Independent O1 algebra: does not call candidate confirmation_value."""
    def __init__(self,policy,gate):super().__init__(policy);self.gate=checked_gate(gate)

    def terminal_loss(self,target,b,*,no_task_consequence=False):
        predicted=self.predict(b,self.gate['return_s'])
        goals,_=self.marginals(predicted)
        if self.gate['return_s']>self.gate['max_confirmation_age_s'] or goals[target]<self.gate['minimum_goal_probability']:
            return self.config['pause_loss']
        return super().terminal_loss(target,predicted,no_task_consequence=no_task_consequence)

    def authorization_prediction(self,target,b):
        predicted=self.predict(b,self.gate['return_s']);goals,_=self.marginals(predicted)
        return goals[target],self.gate['return_s']<=self.gate['max_confirmation_age_s'] and goals[target]>=self.gate['minimum_goal_probability']
