"""Deterministic lifecycle fixture; no decoded EEG or participant performance."""
from ..benchmark.offline import make_session,stamp
from ..benchmark.runner import replay_without_backend
from ..benchmark.synthetic_world import PrivateWorld,CASES
from ..runtime_v2.controller import RuntimeController
from .software_ui import presentation


def run_software_flow():
    s=make_session('step13-software-fixture',prior=(1,0,0,0),fixed_transition=True)
    c=RuntimeController(s,'M');views=[];stages=[]
    for action in [dict(kind='TASK_QUERY',setup_s=1.,duration_s=1.,elapsed_s=2.),
                   dict(kind='CHANNEL_PROBE',prompt='0',setup_s=1.,duration_s=1.,elapsed_s=2.)]:
        q,start,end=c.begin(action)
        views.append(presentation(q,s.context,now=q.to_dict()['sent_at']))
        c.receive(q,'0',action,start,end);stages.append(s.state)
    action=c.choose()
    assert action is not None and action['kind']=='CONFIRM'
    q,start,end=c.begin(action)
    views.append(presentation(q,s.context,now=q.to_dict()['sent_at']))
    c.receive(q,'0',action,start,end);stages.append(s.state)
    assert c.submit()
    for status,dt in [('STARTED',.1),('COMPLETED',1.)]:
        c.now=round(c.now+dt,6)
        s.backend.record_feedback('command-1',status,feedback_id='software-'+status)
        s.refresh_submission(now=stamp(c.now));stages.append(s.state)
    world=PrivateWorld(CASES[0],0,s.policy.channel,s.policy.goals)
    observed=c.now+.05;c.now+=.1
    s.verify_endpoint(world.endpoint(s,observed),now=stamp(c.now));stages.append(s.state)
    assert s.state=='SUCCEEDED' and replay_without_backend(s)
    return {'scope':'deterministic_software_lifecycle_not_human_pilot','modes':[r['mode'] for r in views],
        'views':views,'stages':stages,'final_state':s.state,'fake_backend_submissions':s.backend.submission_count,
        'planner_ref':c.planner_ref,'replay_matches_each_checkpoint':True,
        'stimulus_presented':False,'hardware_calls':0,'cloud_calls':0,'robot_commands':0,
        'real_eeg_trials':0,'decoder_accuracy':None,'probe_compliance':None,'G3_passed':False},s.journal.to_jsonl()
