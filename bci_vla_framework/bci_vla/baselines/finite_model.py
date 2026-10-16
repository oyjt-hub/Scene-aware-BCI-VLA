"""Shared finite comparison problem and independent observation algebra.

This module never calls DiagnosticPolicy.plan/_evaluate/_branches. It shares the
validated channel, transition, consequence and legal-action models deliberately.
"""
from copy import deepcopy
import math

from ..protocols import JointBelief, thaw, require


def entropy(probabilities):
    return -math.fsum(p * math.log2(p) for p in probabilities if p > 0)


class FiniteModel:
    def __init__(self, policy):
        self.policy = policy
        self.config = policy.config
        self.channel = policy.channel
        self.transition = policy.transition
        self.catalog = policy.catalog
        self.goals, self.modes, self.states = policy.goals, policy.modes, policy.states
        self.symbols = tuple(self.channel.data['scope']['symbols'])

    def vector(self, belief):
        cells = {(r['goal_id'], r['mode_id']): r['probability'] for r in belief.data['mass']}
        require(all(not p or key in self.states for key, p in cells.items()), 'unmodeled_support_pause')
        return tuple(cells.get(s, 0) for s in self.states)

    def joint(self, b):
        return JointBelief({'goal_ids': list(self.goals), 'mode_ids': list(self.modes),
            'mass': [{'goal_id': g, 'mode_id': z, 'probability': p} for (g,z),p in zip(self.states,b)],
            'as_of_event_id': 'finite-hypothetical-observation-not-live-evidence'})

    def marginals(self, b):
        return ({g: math.fsum(p for (h,z),p in zip(self.states,b) if h == g) for g in self.goals},
                {z: math.fsum(p for (g,w),p in zip(self.states,b) if w == z) for z in self.modes})

    def factorize(self, b):
        g,z = self.marginals(b)
        return tuple(g[h]*z[w] for h,w in self.states)

    def predict(self, b, seconds, variant='joint'):
        t = self.transition.matrix(seconds)
        out = tuple(math.fsum(p*t[i][j] for i,p in enumerate(b)) for j in range(len(b)))
        if variant == 'fixed_channel':
            g,_ = self.marginals(out)
            return tuple(g[h] if z == self.modes[0] else 0.0 for h,z in self.states)
        if variant == 'inertia':
            # Independent channel monitoring, with task frozen over each gap.
            # Current goal prior can still change through incoming task evidence.
            g,_ = self.marginals(b); _, z = self.marginals(out)
            return tuple(g[h]*z[w] for h,w in self.states)
        return self.factorize(out) if variant == 'factorized' else out

    def likelihood(self, a, y, variant='joint', counts=None):
        mode = 'TASK_QUERY' if a.kind in {'RETRY','EXTEND'} else a.kind
        if variant == 'mix_sources' and mode == 'CHANNEL_PROBE': mode = 'TASK_QUERY'
        c = self.config['actions']; reject = next(s for s in self.symbols if s != c['accept_symbol'])
        values=[]
        for g,z in self.states:
            intended = (a.prompt if mode == 'CHANNEL_PROBE' else c['goal_symbols'][g] if mode == 'TASK_QUERY'
                        else c['accept_symbol'] if g == a.goal_id else reject)
            dist = self.channel.likelihoods(mode, intended, y, scope=thaw(c['scope']), duration_s=a.duration_s,
                compliance_id=c['compliance_id'] if mode == 'CHANNEL_PROBE' else None,
                assumption='state_constant_within_window')
            value=dist[z]
            if variant == 'pseudo_labels' and mode == 'TASK_QUERY' and counts:
                row=counts.get(z,{}).get(intended,{})
                n=sum(row.values())
                # Ten explicitly illustrative prior observations, never a fitted prior.
                value=(10*value+row.get(y,0))/(10+n)
            values.append(value)
        return tuple(values)

    def observe(self, b, a, y, variant='joint', counts=None):
        prior = self.predict(b,a.setup_s,variant)
        mass=tuple(p*l for p,l in zip(prior,self.likelihood(a,y,variant,counts)))
        p=math.fsum(mass)
        require(p > 0, 'zero_likelihood_pause')
        post=tuple(x/p for x in mass)
        if variant in {'factorized','inertia'}: post=self.factorize(post)
        return p,post

    def branches(self,b,a,variant='joint',counts=None):
        result=[]
        for y in (*self.symbols,'NO_RESPONSE'):
            prior=self.predict(b,a.setup_s,variant)
            likelihood=self.likelihood(a,y,variant,counts)
            mass=tuple(p*l for p,l in zip(prior,likelihood)); p=math.fsum(mass)
            if not p: continue
            post=tuple(x/p for x in mass)
            if variant in {'factorized','inertia'}: post=self.factorize(post)
            result.append((y,p,post))
        require(abs(math.fsum(x[1] for x in result)-1)<1e-9,'predictive_not_normalized')
        return result

    def terminal_loss(self,target,b,*,no_task_consequence=False):
        if no_task_consequence:
            return math.fsum(p*(g != target) for (g,z),p in zip(self.states,b))
        total=0.0
        for (g,z),p in zip(self.states,b):
            row=self.policy.outcomes.get(self.config['decision_by_goal'][target],g,z)
            total+=p*math.fsum(v*self.config['weights'][k] for k,v in row['components'].items() if k!='user_time_s')
        return total

    def cost(self,a,variant='joint'):
        if variant=='ignore_probe_cost' and a.kind=='CHANNEL_PROBE': return 0.0
        return a.elapsed_s*self.config['actions']['time_weight_per_s']

    def guard(self,state):
        c=self.config['context']
        require((state.scene_id,state.scene_revision,state.task_id,state.task_revision)==
                (c['scene_id'],c['scene_revision'],c['task_id'],c['task_revision']),'policy_context_mismatch')
        if state.stopped: return 'stop_latched'
        if state.pending: return 'collection_pending_no_menu_change'
        if not state.inference_ready: return 'filter_not_ready'
        if state.now_s>=state.scene_valid_until_s: return 'scene_expired'
        try: self.vector(state.belief)
        except ValueError: return 'unmodeled_support_pause'
        return None

    def information(self,b,a,axis):
        require(axis in {'goal','channel'},'invalid_information_axis')
        i=0 if axis=='goal' else 1
        before=self.marginals(self.predict(b,a.setup_s))[i]
        after=math.fsum(p*entropy(self.marginals(post)[i].values()) for _,p,post in self.branches(b,a))
        return max(0.0,entropy(before.values())-after)

    def update_counts(self,memory,b,a,y):
        result=deepcopy(memory)
        if a.kind not in {'TASK_QUERY','RETRY','EXTEND'}: return result
        # Deliberately flawed self-training ablation: own pre-observation estimate
        # is used as a label. No hidden state or evaluation label is accessible.
        g,z=self.marginals(b); target=max(g,key=g.get); channel=max(z,key=z.get)
        u=self.config['actions']['goal_symbols'][target]
        counts=result.setdefault('pseudo_counts',{}).setdefault(channel,{}).setdefault(u,{})
        counts[y]=counts.get(y,0)+1
        return result
