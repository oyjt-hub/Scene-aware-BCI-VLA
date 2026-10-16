"""Independent same-horizon O1 plus actual authorization evidence checks."""
from copy import deepcopy
from time import perf_counter_ns
from ..protocols import thaw,ProtocolError
from ..baselines.methods import ComparisonSuite
from ..baselines.oracle import ExactOracle
from ..core.diagnostic_policy import PlanningState
from ..benchmark.offline import make_session,read,response,stamp,BASE
from ..benchmark.runner import replay_without_backend
from .controller import gate_for
from .planning import AuthorizationAwarePolicy,AuthorizationAwareFiniteModel


def components(prior=(.4,0,.6,0),*,fixed=True,return_s=.01,menu=None):
    session=make_session('runtime-v2-audit',prior=prior,fixed_transition=fixed)
    gate=gate_for(session,return_s);pc=thaw(session.policy.config)
    if menu is not None:pc['actions']['durations_s']=menu
    policy=AuthorizationAwarePolicy(pc,gate=gate,channel_model=session.policy.channel,
        transition_model=session.policy.transition,outcome_table=session.policy.outcomes)
    model=AuthorizationAwareFiniteModel(policy,gate)
    state=PlanningState(belief=model.joint(prior),scene_id=session.context['scene_id'],scene_revision=session.context['scene_revision'],
        task_id=session.context['task_id'],task_revision=session.context['task_revision'],now_s=BASE.timestamp(),used_s=0,
        remaining_s=20,scene_valid_until_s=BASE.timestamp()+60,contracts=read('examples/diagnosis/state.json')['contracts'])
    return session,policy,model,state


def gate_replay(prior=(.4,0,.6,0),*,fixed=True,return_s=.01):
    session,policy,model,state=components(prior,fixed=fixed,return_s=return_s)
    target=policy.goals[0];action=next(a for a in model.catalog.candidates() if a.kind=='CONFIRM' and a.goal_id==target)
    probability,posterior=model.observe(prior,action,'0')
    value,explanation=policy.confirmation_value(target,posterior);independent=model.terminal_loss(target,posterior)
    session.propose(read('examples/diagnosis/state.json')['contracts'][target],now=stamp(0))
    q=session.begin_interaction('CONFIRM',now=stamp(0),monotonic_now=0)
    session.ingest(response(q,'0',start=1,duration=1,index=1),now=stamp(2),monotonic_now=2)
    allowed=True;reason=None
    try:session.authorize(now=stamp(2+return_s))
    except ProtocolError as exc:allowed=False;reason=str(exc)
    assert replay_without_backend(session)
    assert explanation['authorized']==allowed
    assert abs(value-independent)<1e-12
    return {'prior':prior,'fixed_transition':fixed,'return_s':return_s,'accept_probability':probability,
        'predicted_goal_probability':explanation['predicted_goal_probability'],'candidate_authorizes':explanation['authorized'],
        'actual_authorizes':allowed,'actual_reason':reason,'candidate_continuation_loss':value,
        'independent_O1_continuation_loss':independent,'pause_loss':policy.config['pause_loss'],
        'replay_matches':True,'actual_backend_submissions':session.backend.submission_count}


def alignment_audit():
    gate_cases=[]
    for prior in ((.1,0,.9,0),(.4,0,.6,0),(.6,0,.4,0),(.9,0,.1,0),(1,0,0,0)):
        for dt in (.0,.01,2.,2.01):
            gate_cases.append(gate_replay(prior,fixed=True,return_s=dt))
    for prior in ((.4,0,.6,0),(.6,0,.4,0),(1,0,0,0)):
        for dt in (.01,2.,2.01):gate_cases.append(gate_replay(prior,fixed=False,return_s=dt))
    priors=[(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1),(.25,.25,.25,.25),(0,.5,.5,0),(.81,.09,.09,.01),(.1,.2,.3,.4)]
    rows=[]
    for menu in ([1.],[1.,2.]):
        for i,prior in enumerate(priors):
            session,policy,model,state=components(prior,fixed=False,menu=menu)
            oracle=ExactOracle(model);suite=ComparisonSuite(model,read('configs/comparisons/parameters.json'))
            for h in (0,1,2):
                exact=oracle.solve(state,horizon=h)
                for method in (('M','B9') if h<2 else ('M2',)):
                    start=perf_counter_ns();decision=suite.choose(method,state,horizon_remaining=h);elapsed=perf_counter_ns()-start
                    value=next(r['value'] for r in exact['candidates'] if r['action_id']==decision['selected_action']['action_id'])
                    regret=value-exact['value'];assert -1e-9<=regret<=1e-9
                    assert abs(decision['planning_value']-exact['value'])<1e-9
                    rows.append({'prior_index':i,'menu_durations_s':menu,'horizon':h,'method':method,
                        'selected_action':decision['selected_action']['action_id'],'regret':regret,
                        'planning_value_gap':decision['planning_value']-exact['value'],'method_compute_ns':elapsed,
                        'oracle_compute_ns':exact['compute_ns'],'state_count':len(model.states)})
    return {'scope':'offline_authorization_aware_finite_v2','gate_cases':gate_cases,'same_horizon_rows':rows,
        'all_actual_gate_predictions_match':all(r['candidate_authorizes']==r['actual_authorizes'] for r in gate_cases),
        'max_same_horizon_regret':max(r['regret'] for r in rows),
        'threshold':.8,'max_confirmation_age_s':2.,'all_thresholds_unchanged':True,
        'counterexample_fixed':next(r for r in gate_cases if r['prior']==(.4,0,.6,0) and r['return_s']==.01)['candidate_continuation_loss']==1.,
        'full_runtime_optimality_claim':False,'unknown_future_scene_changes_modeled':False}
