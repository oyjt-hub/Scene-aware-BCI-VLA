"""Independent exhaustive finite tree, same information O1 and clairvoyant O2."""
from time import perf_counter_ns

from ..protocols import require

TERMINAL={'PAUSE','EXIT','CONFIRM'}


class ExactOracle:
    def __init__(self,model): self.model=model

    def _search(self,b,state,horizon,*,variant='joint',allowed=None,counts=None,reveal=False):
        m=self.model; rows=[]
        for a in m.catalog.candidates():
            reason=m.catalog.legality(a,state)
            if reason or (a.kind not in TERMINAL and horizon==0) or (allowed is not None and a.kind not in allowed): continue
            if a.kind in {'PAUSE','EXIT'}:
                value=m.config['pause_loss' if a.kind=='PAUSE' else 'exit_loss']; branches=[]
            else:
                try: branches=m.branches(b,a,variant,counts)
                except ValueError: continue
                after=state.advance(a.elapsed_s); value=m.cost(a,variant)
                for y,p,post in branches:
                    if a.kind=='CONFIRM':
                        continuation=(m.terminal_loss(a.goal_id,post,no_task_consequence=variant=='no_task_consequence')
                                      if y==m.config['actions']['accept_symbol'] else m.config['pause_loss'])
                    elif y=='NO_RESPONSE': continuation=m.config['pause_loss']
                    else:
                        next_counts=counts
                        if variant=='pseudo_labels':
                            next_counts=m.update_counts({'pseudo_counts':counts or {}},b,a,y).get('pseudo_counts')
                        if reveal:
                            continuation=sum(prob*self._search(tuple(float(i==j) for j in range(len(post))),after,horizon-1,
                                variant=variant,allowed=allowed,counts=next_counts,reveal=True)[0]['value']
                                for i,prob in enumerate(post) if prob)
                        else:
                            continuation=self._search(post,after,horizon-1,variant=variant,allowed=allowed,counts=next_counts)[0]['value']
                    value+=p*continuation
            rows.append({'action':a,'value':value,'observation_branches':len(branches)})
        require(bool(rows),'no_legal_terminal_action')
        return min(rows,key=lambda r:(r['value'],r['action'].kind!='PAUSE',r['action'].action_id)),rows

    def solve(self,state,*,horizon=2,variant='joint',allowed=None,counts=None):
        require(type(horizon) is int and 0<=horizon<=3,'bounded_oracle_horizon_required')
        start=perf_counter_ns(); reason=self.model.guard(state)
        if reason:
            a=next(a for a in self.model.catalog.candidates() if a.kind=='PAUSE')
            best,rows={'action':a,'value':self.model.config['pause_loss']},[]
        else: best,rows=self._search(self.model.vector(state.belief),state,horizon,variant=variant,allowed=allowed,counts=counts)
        return {'scope':'offline_exact_same_information','horizon':horizon,'selected_action':best['action'].record(),
                'value':best['value'],'candidates':[dict(r['action'].record(),value=r['value'],observation_branches=r['observation_branches']) for r in rows],
                'compute_ns':perf_counter_ns()-start,'reason':reason,'execution_allowed':False}

    def omniscient_reference(self,state,*,horizon=2):
        """O2 reveals g/z at every subsequent decision, not perfect confirmation.

        Returns a state-conditional policy, not a deployable single action. No
        human/online method accepts true g/z as an input argument.
        """
        require(type(horizon) is int and 0<=horizon<=3,'bounded_oracle_horizon_required')
        require(self.model.guard(state) is None,'invalid_omniscient_reference_state')
        start=perf_counter_ns(); b=self.model.vector(state.belief); rows=[]
        for i,p in enumerate(b):
            if not p: continue
            point=tuple(float(j==i) for j in range(len(b)))
            best,_=self._search(point,state,horizon,reveal=True)
            rows.append({'revealed_state':list(self.model.states[i]),'probability':p,
                         'conditional_action':best['action'].record(),'value':best['value']})
        return {'scope':'non_deployable_omniscient_reference','horizon':horizon,'conditional_actions':rows,
                'value':sum(r['probability']*r['value'] for r in rows),'compute_ns':perf_counter_ns()-start,
                'confirmation_model':'same_noisy_CONFIRM','execution_allowed':False}
