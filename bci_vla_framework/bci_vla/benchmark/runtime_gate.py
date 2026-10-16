"""A reproducible counterexample to treating internal O1 as a runtime oracle."""
from ..protocols import ProtocolError
from .offline import make_session,read,response,stamp
from .runner import replay_without_backend


def authorization_gate_counterexample():
    s=make_session('gate-counterexample',prior=(.4,0,.6,0),fixed_transition=True)
    goal=s.policy.goals[0]
    s.propose(read('examples/diagnosis/state.json')['contracts'][goal],now=stamp(0))
    q=s.begin_interaction('CONFIRM',now=stamp(.1),monotonic_now=.1)
    s.ingest(response(q,'0',start=1,duration=1,index=1),now=stamp(2),monotonic_now=2)
    probability=s.filter.snapshot()['goal_marginals'][goal]
    reason=None
    try:s.authorize(now=stamp(2.01))
    except ProtocolError as exc:reason=str(exc)
    assert reason=='confirmation_posterior_below_declared_policy'
    assert replay_without_backend(s)
    return {'scope':'software_counterexample_not_real_data','initial_goal_probability':.4,
        'actual_observed_confirm_symbol':'0','posterior_after_noisy_confirm':probability,
        'analytic_posterior':(.4*.8)/(.4*.8+.6*.15),'authorization_threshold':s.config['minimum_goal_probability'],
        'observed_accept_is_not_authorization':True,'authorization_blocked_reason':reason,
        'backend_submissions':s.backend.submission_count,'session_state':s.state,
        'internal_terminal_model':'09/10 model assigns accepted branch a terminal execution loss',
        'runtime_terminal_behavior':'11 refuses authorization; uncompleted pause must replace execution branch',
        'requires_versioned_planner_and_O1_alignment':True,'gate_weakened':False,'replay_matches':True}
