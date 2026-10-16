"""Frozen runtime proposer and authorizer with synthetic study instrumentation."""
from ..runtime_v2.controller import RuntimeController
from ..benchmark.offline import make_session,stamp,union_duration
from ..benchmark.runner import replay_without_backend
from .world import StudyWorld

METHODS=['B4','B5','B7','B8','B9','B10','M','M2','A_no_probe','A_periodic','A_fixed_duration','A_no_task_consequence','A_ignore_probe_cost','A_factorized']

def run_episode(case,seed,method,identifier):
 assert method in METHODS
 s=make_session(identifier,prior=case.get('prior'),probe_compliance=case.get('public_compliance','imperfect'))
 c=RuntimeController(s,'M',budget_s=20,diagnostic_limit=2);c.method=method
 w=StudyWorld(case,seed,s.policy.channel,s.policy.goals);error=None;dispatched=None;exec_filter_unchanged=None
 def acquire(action):
  # RETRY/EXTEND use a fresh, disjoint calibrated TASK window; no prefix reuse.
  wire=dict(action)
  if wire['kind'] in {'RETRY','EXTEND'}:wire['kind']='TASK_QUERY'
  q,start,end=c.begin(wire);symbol=w.emit(q,start=start,duration=wire['duration_s']);c.receive(q,symbol,wire,start,end)
 try:
  acquire(dict(kind='TASK_QUERY',setup_s=1.,duration_s=1.,elapsed_s=2.))
  for _ in range(4):
   if c.termination:break
   a=c.choose()
   if a is None:break
   acquire(a)
   if a['kind']=='CONFIRM':
    if s.state=='AWAITING_AUTHORIZATION' and c.submit():
     dispatched=c.now;before=s.filter.snapshot();start=c.now
     c.now+=.1;s.backend.record_feedback('command-1','STARTED',feedback_id='study-start');s.refresh_submission(now=stamp(c.now))
     c.now+=1.;s.backend.record_feedback('command-1','COMPLETED',feedback_id='study-complete');s.refresh_submission(now=stamp(c.now))
     observed=c.now+.05;c.now+=.1;s.verify_endpoint(w.endpoint(s,observed),now=stamp(c.now))
     exec_filter_unchanged=before==s.filter.snapshot();c.intervals.append(dict(phase='fake_execution_and_endpoint',start_s=start,end_s=c.now,user_busy=False))
     c.termination='independent_virtual_endpoint_judged'
    break
  if not c.termination:c.stop('diagnostic_limit_exhausted')
 except Exception as exc:
  error=type(exc).__name__+':'+str(exc)
  if not s.stop_latched:c.stop('recorded_runtime_error')
 assert replay_without_backend(s)
 command=None if s.submission_contract is None else s.submission_contract.data['goal_id']
 truth,channel=w.truth_at(c.now if dispatched is None else dispatched)
 wrong=command is not None and command!=truth;faithful=command==truth and s.state=='SUCCEEDED'
 marg=s.filter.snapshot()['channel_marginals'];mode=max(marg,key=marg.get)
 evaluation=dict(episode_id=identifier,case_id=case['id'],seed=seed,method=method,faithful_virtual_completion=faithful,
  wrong_start=wrong,unresolved=not faithful,unstarted=command is None,elapsed_s=c.now,
  user_busy_s=union_duration([x for x in c.intervals if x['user_busy']]),probe_s=union_duration([x for x in c.intervals if x['phase']=='CHANNEL_PROBE']),
  suppressed_new_goal_start=truth!=s.policy.goals[0] and command==s.policy.goals[0],false_new_goal_start=truth==s.policy.goals[0] and command==s.policy.goals[1],
  confident_wrong_channel_view=marg[mode]>=.8 and mode!=channel,execution_does_not_update_channel=exec_filter_unchanged,
  parameter_update_error_rate=None,parameter_update_status='not_applicable_no_channel_parameter_fit',
  runtime_error=error,termination=c.termination,submission_time_s=dispatched,truth_goal_at_dispatch_or_end=truth,
  truth_channel_at_dispatch_or_end=channel,final_state=s.state)
 public=dict(episode_id=identifier,method=method,scope='synthetic_registered_replication_not_LIBERO',decisions=c.decisions,
  intervals=c.intervals,final_state=s.state,commanded_goal=command,backend_submissions=s.backend.submission_count,
  final_public_filter=s.filter.snapshot(),journal_records=len(s.journal.records),journal_last_hash=s.journal.records[-1]['record_hash'],
  replay_matches_each_checkpoint=True,physical_execution_allowed=False,runtime_error=error)
 return public,evaluation,dict(w.record(),episode_id=identifier),s.journal
