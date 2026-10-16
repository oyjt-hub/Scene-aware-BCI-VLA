"""Evaluation-side orchestration. Truth joins occur after online outputs freeze."""
from collections import Counter
from statistics import mean
from copy import deepcopy
from unittest.mock import patch

from ..protocols import thaw, ProtocolError
from ..adapters.vla import FakeVLAAdapter
from ..contracts.session import replay_session_state
from ..monitoring.journal import EventJournal
from .offline import make_session, OfflineController, stamp, union_duration
from .synthetic_world import PrivateWorld


def replay_without_backend(session):
    journal=EventJournal.from_jsonl(session.journal.to_jsonl())
    with patch.object(FakeVLAAdapter,'__init__',side_effect=AssertionError('replay cannot construct backend')), \
         patch.object(FakeVLAAdapter,'submit',side_effect=AssertionError('replay cannot submit')), \
         patch.object(FakeVLAAdapter,'query',side_effect=AssertionError('replay cannot query')), \
         patch.object(FakeVLAAdapter,'stop',side_effect=AssertionError('replay cannot stop')):
        actual=replay_session_state(journal)
    assert actual==session.snapshot()
    return True


def run_episode(case,seed,method,public_id,*,diagnostic_limit=2):
    s=make_session(public_id,prior=case.get('prior'),probe_compliance=case.get('public_compliance','imperfect'))
    c=OfflineController(s,method,diagnostic_limit=diagnostic_limit)
    world=PrivateWorld(case,seed,s.policy.channel,s.policy.goals)
    initial={'kind':'TASK_QUERY','setup_s':1.,'duration_s':1.,'elapsed_s':2.}
    q,start,end=c.begin(initial);symbol=world.emit(q,start=start,duration=1.)
    c.receive(q,symbol,initial,start,end)
    dispatched_at=None
    for _ in range(diagnostic_limit+2):
        if c.termination is not None:break
        action=c.choose()
        if action is None:break
        q,start,end=c.begin(action);symbol=world.emit(q,start=start,duration=action['duration_s'])
        c.receive(q,symbol,action,start,end)
        if action['kind']=='CONFIRM':
            if s.state=='AWAITING_AUTHORIZATION' and c.submit():
                dispatched_at=c.now;execution_start=c.now
                c.now=round(c.now+.1,6);s.backend.record_feedback('command-1','STARTED',feedback_id='simulated-start')
                s.refresh_submission(now=stamp(c.now))
                c.now=round(c.now+1.,6);s.backend.record_feedback('command-1','COMPLETED',feedback_id='simulated-complete')
                s.refresh_submission(now=stamp(c.now))
                observed=round(c.now+.05,6);c.now=round(c.now+.1,6)
                s.verify_endpoint(world.endpoint(s,observed),now=stamp(c.now))
                c.intervals.append({'phase':'fake_execution_and_endpoint','start_s':execution_start,'end_s':c.now,'user_busy':False})
                c.termination='independent_virtual_endpoint_judged'
            break
    if c.termination is None:c.stop('diagnostic_limit_exhausted')
    assert replay_without_backend(s)
    commanded=None if s.submission_contract is None else s.submission_contract.data['goal_id']
    public={'episode_id':public_id,'method':method,'scope':'L1_synthetic_development_not_physical',
        'decisions':deepcopy(c.decisions),'intervals':deepcopy(c.intervals),'final_state':s.state,
        'termination':c.termination,'commanded_goal':commanded,'submission_time_s':dispatched_at,
        'backend_submissions':s.backend.submission_count,'final_public_filter':s.filter.snapshot(),
        'elapsed_s':c.now,'timeline_union_s':union_duration(c.intervals),
        'user_busy_s':union_duration([r for r in c.intervals if r['user_busy']]),
        'probe_s':union_duration([r for r in c.intervals if r['phase']=='CHANNEL_PROBE']),
        'compute_ns':sum(r['actual_call_compute_ns'] for r in c.decisions),
        'journal_records':len(s.journal.records),'journal_last_hash':s.journal.records[-1]['record_hash'],
        'replay_matches_each_checkpoint':True,'physical_execution_allowed':False}
    # Join privately only after the public controller has finished.
    truth_goal,truth_channel=world.truth_at(c.now if dispatched_at is None else dispatched_at)
    wrong=commanded is not None and commanded!=truth_goal
    faithful=commanded==truth_goal and s.state=='SUCCEEDED'
    marg=s.filter.snapshot()['channel_marginals'];mode=max(marg,key=marg.get)
    evaluation={'episode_id':public_id,'method':method,'case_id':case['id'],'seed':seed,
        'faithful_virtual_completion':faithful,'wrong_start':wrong,'started':commanded is not None,
        'unresolved':not faithful,'unstarted':commanded is None,'elapsed_s':c.now,
        'user_busy_s':public['user_busy_s'],'probe_s':public['probe_s'],'compute_ns':public['compute_ns'],
        'suppressed_new_goal_start':bool(case['goal_change'] and commanded==s.policy.goals[0]),
        'false_new_goal_start':bool(not case['goal_change'] and commanded==s.policy.goals[1]),
        'confident_wrong_channel_view':bool(marg[mode]>=.8 and mode!=truth_channel),
        'parameter_update_error_rate':None,'parameter_update_status':'not_applicable_no_parameter_updates',
        'termination':c.termination,'truth_goal_at_dispatch_or_end':truth_goal,'truth_channel_at_dispatch_or_end':truth_channel}
    private=dict(world.record(),episode_id=public_id)
    return public,evaluation,private,s.journal


def summarize(rows):
    output=[]
    for case,method in sorted({(r['case_id'],r['method']) for r in rows}):
        group=[r for r in rows if r['case_id']==case and r['method']==method];n=len(group)
        output.append({'case_id':case,'method':method,'episodes':n,
            **{k:sum(bool(r[k]) for r in group)/n for k in ['faithful_virtual_completion','wrong_start','unresolved','unstarted',
                'suppressed_new_goal_start','false_new_goal_start','confident_wrong_channel_view']},
            **{f'mean_{k}':mean(r[k] for r in group) for k in ['elapsed_s','user_busy_s','probe_s','compute_ns']},
            'termination_counts':dict(Counter(r['termination'] for r in group))})
    return output
