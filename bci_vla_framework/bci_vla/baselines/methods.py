"""Explicit adapted baselines over the shared finite diagnostic action space."""
from copy import deepcopy
from time import perf_counter_ns

from ..protocols import require
from .finite_model import entropy
from .oracle import ExactOracle, TERMINAL

IDS=tuple(['B'+str(i) for i in range(11)]+['B1P','M','M2'])
VARIANTS={'B2':'fixed_channel','B3':'inertia','A_factorized':'factorized',
          'A_mix_sources':'mix_sources','A_pseudo_labels':'pseudo_labels',
          'A_no_task_consequence':'no_task_consequence','A_ignore_probe_cost':'ignore_probe_cost'}
ABLATIONS=('A_no_probe','A_factorized','A_periodic','A_no_task_consequence',
           'A_mix_sources','A_pseudo_labels','A_ignore_probe_cost','A_fixed_duration')
MEMORY_FIELDS={'diagnostic_count','predicted_goal','prediction_checked','prediction_rejected',
               'correction_queries','last_confirmed_goal','quality_score','pseudo_counts'}


class ComparisonSuite:
    def __init__(self,model,parameters):
        self.model=model; self.oracle=ExactOracle(model); self.parameters=deepcopy(parameters)
        require(parameters['status']=='development_only','validation_freeze_not_available')
        require(parameters['deployment_allowed'] is False,'offline_comparisons_only')
        require(type(parameters['period']) is int and parameters['period']>0,'invalid_period')
        for k in ('confidence','conflict_threshold','quality_threshold','channel_entropy_stop'):
            require(isinstance(parameters[k],(int,float)) and not isinstance(parameters[k],bool) and 0<=parameters[k]<=1,'invalid_threshold')

    def variant(self,method): return VARIANTS.get(method,'joint')

    def inference_state(self,method,state):
        """B2/B3 inference restrictions, not alterations of the evaluation world."""
        b=self.model.vector(state.belief)
        if method=='B2': b=self.model.predict(b,0,'fixed_channel')
        if method in {'B3','A_factorized'}: b=self.model.factorize(b)
        clone=state.advance(0)
        object.__setattr__(clone,'retry_allowed',state.retry_allowed)
        object.__setattr__(clone,'extension_allowed',state.extension_allowed)
        object.__setattr__(clone,'belief',self.model.joint(b))
        return clone

    def observe(self,method,state,action_id,observation,*,memory=None):
        """Synthetic categorical evidence only; real ingest remains step05/08.

        No raw scores, hidden goal, hidden channel, or evaluation labels accepted.
        Returned inference belief uses this method's model; the world is separate.
        """
        require(method in IDS+ABLATIONS,'unknown_method')
        require(self.model.guard(state) is None,'invalid_observation_state')
        a=next((a for a in self.model.catalog.candidates() if a.action_id==action_id),None)
        require(a is not None and a.kind not in {'PAUSE','EXIT','EXECUTE_CONTRACT','RECODE','RECOMMEND_RECALIBRATION'},'invalid_observation_action')
        require(self.model.catalog.legality(a,state) is None,'illegal_observation_action')
        memory=deepcopy(memory or {})
        require(set(memory)<=MEMORY_FIELDS,'private_or_unsupported_method_input')
        b=self.model.vector(self.inference_state(method,state).belief)
        _,post=self.model.observe(b,a,observation,self.variant(method),memory.get('pseudo_counts'))
        after=state.advance(a.elapsed_s); object.__setattr__(after,'belief',self.model.joint(post))
        memory['diagnostic_count']=memory.get('diagnostic_count',0)+int(a.kind!='CONFIRM')
        if method=='A_pseudo_labels': memory=self.model.update_counts(memory,b,a,observation)
        if method=='B1P' and a.kind in {'TASK_QUERY','RETRY','EXTEND'}:
            g,_=self.model.marginals(post)
            predicted=memory.get('predicted_goal')
            memory['prediction_checked']=True
            memory['prediction_rejected']=predicted is not None and max(g,key=g.get)!=predicted
            if memory['prediction_rejected']: memory['correction_queries']=memory.get('correction_queries',0)+1
        return after,memory

    def choose(self,method,state,*,memory=None,execution_mode='offline',horizon_remaining=2):
        require(execution_mode=='offline','offline_comparisons_only')
        require(type(horizon_remaining) is int and 0<=horizon_remaining<=3,'bounded_comparison_horizon_required')
        require(method in IDS+ABLATIONS,'unknown_or_non_deployable_method')
        start=perf_counter_ns(); m=self.model; memory=deepcopy(memory or {})
        require(set(memory)<=MEMORY_FIELDS,'private_or_unsupported_method_input')
        reason=m.guard(state)
        if reason:
            action=next(a for a in m.catalog.candidates() if a.kind=='PAUSE')
            return self._result(method,action.record(),reason,start)
        state=self.inference_state(method,state); b=m.vector(state.belief)
        variant=self.variant(method); counts=memory.get('pseudo_counts')
        if method=='B0':
            # Preserve the frozen real LLM chain as an explicit external adapter.
            # No counterfeit proposal/timing is generated to fill the comparison.
            return {'method':method,'status':'unavailable_external_legacy_trace',
                    'selected_action':None,'required':['source-bound proposal trace','measured complete query/LLM timing','common confirmation wrapper'],
                    'compute_ns':perf_counter_ns()-start,'execution_allowed':False}
        if method in {'M','M2'}:
            r=m.policy.plan(state,depth=min(1 if method=='M' else 2,horizon_remaining))
            return self._result(method,r['selected_action'],r['reason'],start,planning_value=r['selected_action']['expected_loss'])
        allowed=set(TERMINAL)|{'TASK_QUERY','CHANNEL_PROBE','RETRY','EXTEND'}
        if method in {'B3','B4','A_no_probe','B6'}: allowed.discard('CHANNEL_PROBE')
        if method=='A_fixed_duration':
            # Registry's long EXTEND is disabled; durations still remain calibrated.
            allowed.discard('EXTEND')
        horizon=0 if method in {'B1','B1P'} else min(1,horizon_remaining)
        search=self.oracle.solve(state,horizon=horizon,variant=variant,allowed=allowed,counts=counts)
        rows=search['candidates']; best=search['selected_action']; chosen_reason='task_risk_with_shared_confirmation'
        if method=='A_fixed_duration':
            rows=[r for r in rows if r['duration_s'] in {0,self.parameters['fixed_duration_s']}]
            best=min(rows,key=lambda r:(r['value'],r['kind']!='PAUSE',r['action_id']))
        g,z=m.marginals(b); confidence=max(g.values())
        def ranked(kinds):
            pool=[r for r in rows if r['kind'] in kinds]
            return min(pool,key=lambda r:(r['value'],r['kind']!='PAUSE',r['action_id'])) if pool else None
        def choose_information(axis,kinds):
            pool=[]
            for row in rows:
                if row['kind'] not in kinds: continue
                a=next(a for a in m.catalog.candidates() if a.action_id==row['action_id'])
                try: info=m.information(b,a,axis)
                except ValueError: continue
                if info>1e-12: pool.append((info/a.elapsed_s,info,row))
            return max(pool,key=lambda x:(x[0],x[1],x[2]['action_id']))[2] if pool else None
        if method in {'B1','B1P'}:
            # Predictions are supplied by an external public predictor, never g/z.
            need_query=confidence<self.parameters['confidence']
            if method=='B1P':
                require(memory.get('predicted_goal') in m.goals,'public_prediction_required')
                conflict_with_prediction=max(g,key=g.get)!=memory['predicted_goal']
                need_query=confidence<self.parameters['confidence'] or (not memory.get('prediction_checked',False) and conflict_with_prediction)
                chosen_reason='prediction_verification_or_correction_query' if need_query else 'prediction_checked_noisy_final_confirmation'
            if need_query:
                if horizon_remaining==0:
                    pause=next(a for a in m.catalog.candidates() if a.kind=='PAUSE')
                    return self._result(method,pause.record(),'verification_budget_horizon_exhausted',start)
                # Evaluate legal question directly; this is a fixed flow, not VoI.
                probe=self.oracle.solve(state,horizon=1,allowed=set(TERMINAL)|{'TASK_QUERY'})
                queries=[r for r in probe['candidates'] if r['kind']=='TASK_QUERY']
                if queries: best=min(queries,key=lambda r:r['value'])
        elif method in {'B5','A_periodic'}:
            due=memory.get('diagnostic_count',0)%self.parameters['period']==0
            pick=ranked({'CHANNEL_PROBE'}) if due else ranked({'TASK_QUERY','CONFIRM','PAUSE','EXIT'})
            if pick: best=pick
            chosen_reason='fixed_period_probe_due' if due else 'fixed_period_task_phase'
        elif method in {'B6','B7'}:
            previous=memory.get('last_confirmed_goal')
            conflict=(1-g[previous]) if previous in g else 0.0
            quality=memory.get('quality_score')
            require(quality is None or (type(quality) in {float,int} and 0<=quality<=1),'invalid_quality_score')
            trigger=conflict>=self.parameters['conflict_threshold'] or (quality is not None and quality<=self.parameters['quality_threshold'])
            if trigger:
                pick=ranked({'TASK_QUERY','RETRY','EXTEND'} if method=='B6' else {'CHANNEL_PROBE'})
                if pick: best=pick
            else:
                pick=ranked({'TASK_QUERY','CONFIRM','PAUSE','EXIT'})
                if pick: best=pick
            chosen_reason='conflict_repeat_trigger' if method=='B6' else 'conflict_or_measured_quality_probe_trigger'
        elif method=='B8':
            pick=choose_information('channel',{'CHANNEL_PROBE'}) if entropy(z.values())>self.parameters['channel_entropy_stop'] else None
            if pick: best=pick; chosen_reason='channel_information_gain_per_second_then_semantics'
            else:
                pick=ranked({'TASK_QUERY','CONFIRM','PAUSE','EXIT'})
                if pick: best=pick
        elif method=='B10':
            # RBC-inspired recursive evidence and MI coding restricted to the one
            # verified binary codebook; stop on a FINAL disjoint acquisition.
            pick=choose_information('goal',{'TASK_QUERY','RETRY','EXTEND'})
            probe=ranked({'CHANNEL_PROBE'})
            if confidence>=self.parameters['confidence']:
                terminal=ranked(TERMINAL)
                if terminal: best=terminal
            elif pick is not None and pick['value']<m.config['pause_loss']:
                best=probe if probe is not None and probe['value']<pick['value'] else pick
            chosen_reason='restricted_recursive_bayes_MI_and_final_window_stopping_plus_semantic_VoI'
        return self._result(method,best,chosen_reason,start,planning_value=best.get('value',search['value']))

    @staticmethod
    def _result(method,action,reason,start,**extra):
        return dict(method=method,status='offline_ready',selected_action=action,reason=reason,
                    compute_ns=perf_counter_ns()-start,execution_allowed=False,**extra)

    def evaluate(self,method,state,*,horizon=2,memory=None):
        """Exact small fixed-policy audit under the ORIGINAL world loss/model.

        Used to check ablations are not scored by their own distorted objectives.
        This returns model expectations, never empirical performance.
        """
        require(type(horizon) is int and 0<=horizon<=3,'bounded_evaluation_required')
        m=self.model; world=m.vector(state.belief); memory=deepcopy(memory or {})
        def walk(public,truth,remaining,mem):
            decision=self.choose(method,public,memory=mem,horizon_remaining=remaining)
            require(decision['status']=='offline_ready','baseline_not_available')
            a=next(a for a in m.catalog.candidates() if a.action_id==decision['selected_action']['action_id'])
            if a.kind in {'PAUSE','EXIT'}: return m.config['pause_loss' if a.kind=='PAUSE' else 'exit_loss']
            if a.kind not in TERMINAL and remaining==0: return m.config['pause_loss']
            value=m.cost(a)  # Always actual shared cost, including cost ablations.
            for y,p,post in m.branches(truth,a):
                if a.kind=='CONFIRM':
                    child=m.terminal_loss(a.goal_id,post) if y==m.config['actions']['accept_symbol'] else m.config['pause_loss']
                elif y=='NO_RESPONSE': child=m.config['pause_loss']
                else:
                    try: after,next_mem=self.observe(method,public,a.action_id,y,memory=mem)
                    except ValueError: child=m.config['pause_loss']
                    else: child=walk(after,post,remaining-1,next_mem)
                value+=p*child
            return value
        initial=self.inference_state(method,state)
        return {'scope':'analytic_fixed_policy_audit_not_experiment','method':method,
                'horizon':horizon,'actual_model_expected_loss':walk(initial,world,horizon,memory),
                'execution_allowed':False}
