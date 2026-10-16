"""Handwritten offline lifecycle fixtures, not BCI/robot performance experiments."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from bci_vla.protocols import JointBelief, ProtocolError, thaw
from bci_vla.models.channel import ChannelModel, JointTransitionModel
from bci_vla.models.consequence import OutcomeTable
from bci_vla.scene.state import GroundedScene
from bci_vla.core.joint_belief import GoalVocabulary, JointBeliefFilter
from bci_vla.core.diagnostic_policy import DiagnosticPolicy
from bci_vla.adapters.vla import FakeVLAAdapter
from bci_vla.contracts import EndpointEvidence
from bci_vla.contracts.session import InteractionSession, replay_session_state
from bci_vla.monitoring.journal import EventJournal

BASE=datetime(2026,10,15,tzinfo=timezone.utc)
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def stamp(s):return (BASE+timedelta(seconds=s)).isoformat()
def write(p,v):(ROOT/p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def fixture(name,*,status='ACCEPTED'):
    f=read('configs/belief/simulation_binary_filter.json');f['current_context']['session_id']='step11-'+name
    vocabulary=GoalVocabulary(f['vocabulary']);prior=deepcopy(f['initial_joint'])
    # Declared known-state prior only tests lifecycle, never claims inference accuracy.
    prior['mass']=[{'goal_id':vocabulary.known_goal_ids[0],'mode_id':'normal','probability':1}]
    channel=ChannelModel(read('configs/channels/simulation_binary_v1.json'))
    transition=JointTransitionModel(read('examples/joint_belief/fixed_state_transition.json'))
    filt=JointBeliefFilter(JointBelief(prior),vocabulary=vocabulary,channel_model=channel,transition_model=transition,
        current_context=f['current_context'],layout_id=f['layout_id'],initial_at=BASE)
    pc=read('configs/diagnosis.json');pc['context']=dict(f['current_context'])
    policy=DiagnosticPolicy(pc,channel_model=channel,transition_model=transition,
        outcome_table=OutcomeTable(read('configs/consequences/simulation_canonical_outcomes.json')))
    scene=read('examples/scene_outcomes/scene.json');scene['identity']['session_id']=f['current_context']['session_id']
    session=InteractionSession(current_context=f['current_context'],scene=GroundedScene(scene),belief_filter=filt,policy=policy,
        backend=FakeVLAAdapter(submit_status=status),config=read('configs/execution/session.json'),started_at=BASE)
    proposal=read('examples/diagnosis/state.json')['contracts'][vocabulary.known_goal_ids[0]]
    session.propose(proposal,now=stamp(0))
    return session


def answer(session,mode,*,request_s,start_s,index,symbol='0'):
    kw={} if mode!='CHANNEL_PROBE' else {'prompt_symbol':symbol,'presentation_ack':True,
        'presented_at':{'host_id':'ui','at':stamp(request_s),'offset_ms':0,'uncertainty_ms':0}}
    q=session.begin_interaction(mode,now=stamp(request_s),monotonic_now=request_s,**kw).to_dict()
    r=read('examples/protocols/confirm_response.json')
    for key in ('session_id','task_id','task_revision','scene_id','scene_revision','mapping_id','stimulus_version',
                'channel_profile_id','mode','interaction_id','query_id','probe_id'):r[key]=q[key]
    r.update(event_id='fixture-response-'+str(index),response_to=q['event_id'],host_id='bci',sequence=index,
        sent_at=stamp(start_s+1.1),expires_at=q['expires_at'],clock_offset_ms=0,clock_uncertainty_ms=0)
    r['payload'].update(status='OBSERVATION',symbol=symbol)
    r['payload']['sample_window'].update(window_id='fixture-window-'+str(index),recording_id='step11-fixture',
        sample_start=index*1000,sample_end=index*1000+250,started_at=stamp(start_s),ended_at=stamp(start_s+1),
        source_kind='offline_replay',sample_rate_hz=250)
    result=session.ingest(r,now=stamp(start_s+1.2),monotonic_now=start_s+1.2)
    assert result.accepted,result.reason


def observation(s):
    v=read('configs/execution/fake_vla_interface.json')['fixture_observation'];v['captured_at']=stamp(s);return v


def rebuild(session):
    imported=EventJournal.from_jsonl(session.journal.to_jsonl())
    with patch.object(FakeVLAAdapter,'__init__',side_effect=AssertionError('no backend construction during replay')), \
         patch.object(FakeVLAAdapter,'submit',side_effect=AssertionError('no dispatch during replay')), \
         patch.object(FakeVLAAdapter,'query',side_effect=AssertionError('no query during replay')), \
         patch.object(FakeVLAAdapter,'stop',side_effect=AssertionError('no stop during replay')):
        restored=replay_session_state(imported)
    assert restored==session.snapshot()
    return {'session_id':session.context['session_id'],'records':len(imported.records),
        'final_state':restored['state'],'exact_state_match':True,'recomputed_each_checkpoint':True,'backend_calls':0}


def main():
    legal=fixture('legal')
    answer(legal,'TASK_QUERY',request_s=.1,start_s=.2,index=1)
    answer(legal,'CHANNEL_PROBE',request_s=1.5,start_s=1.6,index=2,symbol='1')
    assert legal.backend.submission_count==0 and legal.authorization is None
    decision=legal.request_policy(remaining_s=20,scene_valid_until_s=BASE.timestamp()+60,now=stamp(2.9))
    answer(legal,'CONFIRM',request_s=3,start_s=3.1,index=3)
    legal.authorize(now=stamp(4.4))
    accepted=legal.dispatch('legal-command',observation(4.5),now=stamp(4.5));ack_state=legal.state
    assert accepted['status']=='ACCEPTED' and ack_state=='SUBMITTED'
    legal.backend.record_feedback('legal-command','STARTED',feedback_id='fixture-start')
    legal.refresh_submission(now=stamp(5))
    legal.backend.record_feedback('legal-command','COMPLETED',feedback_id='fixture-complete')
    legal.refresh_submission(now=stamp(6));completed_state=legal.state
    tr=read('examples/scene_outcomes/trajectory.json');tr.update(split='test',source_ref='step11-handwritten-independent-endpoint',trajectory_id='step11-endpoint')
    p=legal.submission_contract.data['proposal']
    evidence=EndpointEvidence({'evidence_id':'step11-evidence','session_id':legal.context['session_id'],
        'submission_id':'legal-command','contract_id':p['contract_id'],'contract_revision':p['revision'],
        'scene_id':p['scene_id'],'scene_revision':p['scene_revision'],'source':'simulation_fixture_annotation',
        'observed_at':{'host_id':'fixture-annotator','at':stamp(6.1),'offset_ms':0,'uncertainty_ms':0},'trajectory':tr})
    legal.verify_endpoint(evidence,now=stamp(6.2));assert legal.state=='SUCCEEDED'

    unknown=fixture('unknown',status='UNKNOWN')
    answer(unknown,'CONFIRM',request_s=.1,start_s=1,index=1)
    unknown.authorize(now=stamp(2.3));obs=observation(2.4)
    unknown.dispatch('unknown-command',obs,now=stamp(2.4));unknown.refresh_submission(now=stamp(2.5))
    unknown.dispatch('unknown-command',obs,now=stamp(2.6))
    rejected=False
    try:unknown.dispatch('new-command',obs,now=stamp(2.65))
    except ProtocolError:rejected=True
    assert rejected and unknown.backend.submission_count==1
    unknown.request_stop(now=stamp(2.7));before_feedback=unknown.snapshot()
    unknown.backend.record_feedback('unknown-command','STOPPED',feedback_id='fixture-stopped')
    unknown.refresh_submission(now=stamp(3))
    assert before_feedback['stop_feedback']=='REQUESTED' and unknown.stop_feedback=='SIMULATED_STOPPED'
    assert not unknown.snapshot()['physical_stop_verified']

    (ROOT/'execution/11_session_events.jsonl').write_text(legal.journal.to_jsonl(),encoding='utf-8')
    (ROOT/'execution/11_unknown_stop_events.jsonl').write_text(unknown.journal.to_jsonl(),encoding='utf-8')
    trace={'date':'2026-10-16','scope':'handwritten_offline_lifecycle_fixtures',
        'prior':'declared_known_left_normal_with_fixed_transition_not_an_accuracy_experiment',
        'legal':{'state_after_ack':ack_state,'state_after_completed_feedback':completed_state,
            'final_state':legal.state,'policy_decision':decision,'backend_submissions':legal.backend.submission_count,
            'authorization_consumed':legal.authorization_consumed,'instruction':legal.contract.data['instruction']},
        'unknown_stop':{'new_id_resubmit_rejected':rejected,'backend_submissions':unknown.backend.submission_count,
            'state_before_stop_feedback':before_feedback['state'],'stop_before_feedback':before_feedback['stop_feedback'],
            'final_state':unknown.state,'stop_after_feedback':unknown.stop_feedback},
        'physical_execution_allowed':False,'physical_stop_verified':False,'hardware_calls':0,'cloud_calls':0,
        'G2_passed':False,'performance_claim':False}
    replay={'result':'PASS','cases':[rebuild(legal),rebuild(unknown)],'backend_calls':0,
        'scope':'pure_event_reconstruction_and_checkpoint_comparison','authenticates_physical_events':False}
    write('execution/11_lifecycle_trace.json',trace);write('execution/11_replay_result.json',replay)
    print(json.dumps({'result':'PASS','legal':legal.state,'unknown_stop':unknown.state,'replay':replay},indent=2))


if __name__=='__main__':main()
