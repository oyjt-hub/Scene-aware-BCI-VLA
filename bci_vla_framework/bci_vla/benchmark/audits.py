"""Explicit mechanism, finite-oracle and lifecycle fault acceptance fixtures."""
from copy import deepcopy
from time import perf_counter_ns

from ..protocols import JointBelief, ProtocolError, thaw
from ..baselines.finite_model import FiniteModel, entropy
from ..baselines.oracle import ExactOracle
from ..baselines.methods import ComparisonSuite
from ..core.diagnostic_policy import DiagnosticPolicy, PlanningState
from ..scene.state import GroundedScene
from .offline import make_session, OfflineController, observation, response, read, stamp, BASE
from .runner import replay_without_backend


def mechanisms():
    s=make_session('mechanism',prior=(0,.5,.5,0),fixed_transition=True);m=FiniteModel(s.policy)
    scope=thaw(s.policy.config['actions']['scope']);channel=s.policy.channel
    rows=[]
    for swap in (False,True):
        for y in ('0','1','NO_RESPONSE'):
            right_symbol='0' if swap else '1';left_symbol='1' if swap else '0'
            a=channel.likelihoods('TASK_QUERY',right_symbol,y,scope=scope,duration_s=1.,assumption='state_constant_within_window')['normal']
            b=channel.likelihoods('TASK_QUERY',left_symbol,y,scope=scope,duration_s=1.,assumption='state_constant_within_window')['flip']
            rows.append({'swap_code':swap,'output':y,'right_normal':a,'left_flip':b,'equal':abs(a-b)<1e-12})
    prior=(0,.5,.5,0);probe=next(a for a in m.catalog.candidates() if a.kind=='CHANNEL_PROBE' and a.prompt=='0' and a.duration_s==1.)
    _,post=m.observe(prior,probe,'0');before_g,before_z=m.marginals(prior);after_g,after_z=m.marginals(post)
    likelihood=m.likelihood(probe,'0')
    uninform=make_session('mechanism-negative',prior=prior,probe_compliance='uninformative',fixed_transition=True)
    negative=FiniteModel(uninform.policy);_,unchanged=negative.observe(prior,probe,'0')
    independent=(.25,.25,.25,.25);_,independent_post=m.observe(independent,probe,'0')
    independent_g=m.marginals(independent_post)[0]
    checks={'binary_reencoding_preserves_alias':all(r['equal'] for r in rows),
        'probe_is_goal_independent_given_channel':abs(likelihood[0]-likelihood[2])<1e-12 and abs(likelihood[1]-likelihood[3])<1e-12,
        'informative_probe_reduces_joint_entropy_for_this_observation':entropy(post)<entropy(prior),
        'correlated_goal_changes_without_overwrite':max(abs(after_g[g]-before_g[g]) for g in m.goals)>.1,
        'independent_goal_stays_half':all(abs(p-.5)<1e-12 for p in independent_g.values()),
        'uninformative_probe_preserves_joint':all(abs(a-b)<1e-12 for a,b in zip(prior,unchanged)),
        'uninformative_probe_has_zero_mutual_information':negative.information(prior,probe,'channel')<1e-12,
        'recode_unavailable_without_separate_calibration':s.policy.catalog.legality(next(a for a in m.catalog.candidates() if a.kind=='RECODE'),OfflineController(s,'M').planning_state())=='no_separately_calibrated_alternative_mapping'}
    assert all(checks.values()),checks
    return {'scope':'analytic_L1_fixed_transition_boundary_checks','checks':checks,'alias_rows':rows,
        'prior':prior,'informative_posterior':post,'uninformative_posterior':unchanged,
        'before_goal':before_g,'after_goal':after_g,'before_channel':before_z,'after_channel':after_z,
        'caution':'positive observation is a conditional example; not all random observations reduce entropy'}


def oracle_audit():
    priors=[(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1),(.25,.25,.25,.25),(0,.5,.5,0),(.81,.09,.09,.01),(.1,.2,.3,.4)]
    rows=[]
    for menu in ([1.],[1.,2.]):
        s=make_session('oracle-audit');pc=thaw(s.policy.config);pc['actions']['durations_s']=menu
        policy=DiagnosticPolicy(pc,channel_model=s.policy.channel,transition_model=s.policy.transition,outcome_table=s.policy.outcomes)
        model=FiniteModel(policy);oracle=ExactOracle(model);suite=ComparisonSuite(model,read('configs/comparisons/parameters.json'))
        for index,prior in enumerate(priors):
            state=PlanningState(belief=model.joint(prior),scene_id=s.context['scene_id'],scene_revision=s.context['scene_revision'],
                task_id=s.context['task_id'],task_revision=s.context['task_revision'],now_s=BASE.timestamp(),used_s=0,
                remaining_s=20,scene_valid_until_s=BASE.timestamp()+60,contracts=read('examples/diagnosis/state.json')['contracts'])
            for h in (0,1,2):
                exact=oracle.solve(state,horizon=h)
                for method in (('M','B9') if h<2 else ('M2',)):
                    start=perf_counter_ns();result=suite.choose(method,state,horizon_remaining=h);elapsed=perf_counter_ns()-start
                    q=next(row['value'] for row in exact['candidates'] if row['action_id']==result['selected_action']['action_id'])
                    regret=q-exact['value'];assert regret>=-1e-9
                    rows.append({'prior_index':index,'prior':prior,'menu_durations_s':menu,'hidden_states':len(model.states),
                        'candidate_count':len(exact['candidates']),'horizon':h,'method':method,
                        'selected_action':result['selected_action']['action_id'],'oracle_action':exact['selected_action']['action_id'],
                        'oracle_value':exact['value'],'exact_value_of_chosen_action':q,'same_horizon_regret':regret,
                        'planning_value_gap':result['planning_value']-exact['value'],'oracle_compute_ns':exact['compute_ns'],
                        'method_compute_ns':elapsed,'compute_budget_ns':5_000_000_000,'within_compute_budget':max(elapsed,exact['compute_ns'])<=5_000_000_000})
    return {'scope':'internal_finite_planning_model_only_not_runtime_oracle','rows':rows,
        'max_same_horizon_regret':max(r['same_horizon_regret'] for r in rows),
        'max_planning_value_gap':max(abs(r['planning_value_gap']) for r in rows),
        'max_oracle_compute_ns':max(r['oracle_compute_ns'] for r in rows),'all_within_compute_budget':all(r['within_compute_budget'] for r in rows),
        'scale_axes':['diagnostic_horizon_0_1_2','one_or_two_durations','eight_joint_priors'],
        'state_space_scaling':'unsupported beyond fixed 2 goals x 2 modes; not tested',
        'runtime_gap':'O1 and frozen09 terminal loss do not include step11 posterior>=0.8 authorization rejection; full episodes record those blocked branches separately',
        'optimal_pomdp_claim':False}


def _confirmed():
    s=make_session('fault',prior=(1,0,0,0),fixed_transition=True)
    goal=s.policy.goals[0];s.propose(read('examples/diagnosis/state.json')['contracts'][goal],now=stamp(0))
    q=s.begin_interaction('CONFIRM',now=stamp(.1),monotonic_now=.1)
    r=response(q,'0',start=1,duration=1,index=1);assert s.ingest(r,now=stamp(2),monotonic_now=2).accepted
    s.authorize(now=stamp(2.01));return s,q,r


def faults():
    rows=[]
    def record(name,s,ok,detail):
        assert ok,(name,s.snapshot());assert replay_without_backend(s)
        rows.append({'fault':name,'passed':True,'detail':detail,'final_state':s.state,
            'backend_submissions':s.backend.submission_count,'stop_feedback':s.stop_feedback,
            'physical_stop_verified':s.snapshot()['physical_stop_verified'],'replay_matches':True})
    def blocked(call):
        try:call()
        except ProtocolError:return True
        return False

    s=make_session('fault-no-response');q=s.begin_interaction('TASK_QUERY',now=stamp(.1),monotonic_now=.1)
    s.ingest(response(q,'NO_RESPONSE',start=1,duration=1,index=1),now=stamp(2),monotonic_now=2)
    record('no_response',s,s.state=='PAUSED' and s.authorization is None,'no default answer or authorization')

    s=make_session('fault-old-menu');q=s.begin_interaction('TASK_QUERY',now=stamp(.1),monotonic_now=.1)
    r=response(q,'0',start=1,duration=1,index=1);r['mapping_id']='old-menu'
    accepted=s.ingest(r,now=stamp(2),monotonic_now=2).accepted;s.request_stop(now=stamp(2.01))
    record('old_menu',s,not accepted and s.backend.submission_count==0,'stale mapping quarantined')

    s=make_session('fault-expired');s.propose(read('examples/diagnosis/state.json')['contracts'][s.policy.goals[0]],now=stamp(0))
    q=s.begin_interaction('CONFIRM',now=stamp(.1),monotonic_now=.1,expires_in_s=1)
    accepted=s.ingest(response(q,'0',start=1,duration=1,index=1),now=stamp(2),monotonic_now=2).accepted
    s.expire(now=stamp(2.01),monotonic_now=2.01)
    record('expired_confirmation',s,not accepted and s.authorization is None,'expired receipt cannot confirm')

    s,q,r=_confirmed();accepted=s.ingest(r,now=stamp(2.02),monotonic_now=2.02).accepted
    s.dispatch('command-1',observation(2.03),now=stamp(2.03));s.dispatch('command-1',observation(2.03),now=stamp(2.04))
    record('duplicate_confirmation_and_dispatch',s,not accepted and s.backend.submission_count==1,'same payload cached, actual confirmation not reused')

    s,_,_=_confirmed();scene=s.scene.to_dict();scene['revision']=2;scene['objects'][1]['visible']=False
    s.revise_context(dict(s.context,scene_revision=2),GroundedScene(scene),now=stamp(2.02))
    record('target_disappeared',s,blocked(lambda:s.dispatch('command-1',observation(2.03),now=stamp(2.03))) and s.backend.submission_count==0,'scene revision invalidates authorization')

    s=make_session('fault-policy-timeout');c=OfflineController(s,'M',compute_budget_ns=0);c.choose()
    record('policy_timeout',s,c.termination=='policy_compute_budget_exceeded' and s.backend.submission_count==0,'result discarded after measured computation; no hard preemption claim')

    s=make_session('fault-illegal-action');a=read('configs/execution/fake_vla_interface.json')['fixture_actions'];a['values'][0][0]=float('nan')
    record('illegal_action',s,blocked(lambda:s.backend.validate_actions(a)) and s.backend.submission_count==0,'nonfinite action rejected before use')

    s=make_session('fault-cancel');q=s.begin_interaction('TASK_QUERY',now=stamp(.1),monotonic_now=.1);s.cancel(now=stamp(.2))
    accepted=s.ingest(response(q,'0',start=1,duration=1,index=1),now=stamp(2),monotonic_now=2).accepted
    record('cancel_late_response',s,not accepted and s.state=='CANCELLED','cancel latch rejects late ordinary evidence')

    s,_,_=_confirmed();s.dispatch('command-1',observation(2.02),now=stamp(2.02))
    s.backend.record_feedback('command-1','STARTED',feedback_id='start');s.refresh_submission(now=stamp(2.1))
    s.request_stop(now=stamp(2.2));requested=s.stop_feedback=='REQUESTED' and s.state=='PAUSED'
    s.backend.record_feedback('command-1','STOPPED',feedback_id='stop');s.refresh_submission(now=stamp(2.3))
    record('stop_during_execution',s,requested and s.stop_feedback=='SIMULATED_STOPPED' and s.state=='CANCELLED','stop requested first; explicit fake feedback later; physical stop false')

    s,_,_=_confirmed()
    def lost(*args):raise ConnectionError('synthetic transport response lost')
    s.backend.submit=lost;s.dispatch('command-1',observation(2.02),now=stamp(2.02))
    refused=blocked(lambda:s.dispatch('retry',observation(2.03),now=stamp(2.03)))
    record('unknown_submission_no_retry',s,refused and s.state=='SUBMISSION_UNKNOWN','exception consumes authorization and prevents new submission')
    return {'scope':'software_injected_faults_not_EEG_or_physical_safety_validation','cases':rows,'all_passed':all(r['passed'] for r in rows)}
