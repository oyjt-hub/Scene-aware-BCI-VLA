"""Actual bound replay confirmation, fake lifecycle and fault transaction tests."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import unittest

from bci_vla.protocols import JointBelief, TaskContract, Event, ProtocolError, thaw
from bci_vla.models.channel import ChannelModel, JointTransitionModel
from bci_vla.models.consequence import OutcomeTable, ObservedTrajectory, evaluate_trajectory
from bci_vla.scene.state import GroundedScene
from bci_vla.core.joint_belief import GoalVocabulary, JointBeliefFilter
from bci_vla.core.diagnostic_policy import DiagnosticPolicy
from bci_vla.adapters.vla import FakeVLAAdapter, DryRunVLAAdapter
from bci_vla.contracts.session import InteractionSession, replay_session_state
from bci_vla.contracts import EndpointEvidence
from bci_vla.monitoring.journal import EventJournal

ROOT=Path(__file__).resolve().parents[1]; BASE=datetime(2026,10,15,tzinfo=timezone.utc)
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def stamp(s):return (BASE+timedelta(seconds=s)).isoformat()

def make_session(*,submit_status='ACCEPTED',initial=None,dynamic=False,dry=False):
    f=read('configs/belief/simulation_binary_filter.json'); v=GoalVocabulary(f['vocabulary'])
    if initial is None:
        initial=deepcopy(f['initial_joint']); initial['mass']=[{'goal_id':v.known_goal_ids[0],'mode_id':'normal','probability':1}]
    channel=ChannelModel(read('configs/channels/simulation_binary_v1.json'))
    trans=JointTransitionModel(read('configs/belief/simulation_canonical_transition.json' if dynamic else 'examples/joint_belief/fixed_state_transition.json'))
    filt=JointBeliefFilter(JointBelief(initial),vocabulary=v,channel_model=channel,transition_model=trans,
        current_context=f['current_context'],layout_id=f['layout_id'],initial_at=BASE)
    p=DiagnosticPolicy(read('configs/diagnosis.json'),channel_model=channel,transition_model=trans,
        outcome_table=OutcomeTable(read('configs/consequences/simulation_canonical_outcomes.json')))
    backend=DryRunVLAAdapter() if dry else FakeVLAAdapter(submit_status=submit_status)
    return InteractionSession(current_context=f['current_context'],scene=GroundedScene(read('examples/scene_outcomes/scene.json')),
        belief_filter=filt,policy=p,backend=backend,config=read('configs/execution/session.json'),started_at=BASE)

def propose(s,*,identifier=None):
    g=s.filter.vocabulary.known_goal_ids[0]
    p=read('examples/diagnosis/state.json')['contracts'][g]
    if identifier:p['contract_id']=identifier
    return s.propose(p,now=stamp(0))

def response(q,*,symbol='0',status='OBSERVATION',start=1,index=1,change=None):
    q=q.to_dict() if isinstance(q,Event) else q
    r=read('examples/protocols/confirm_response.json')
    for k in ('session_id','task_id','task_revision','scene_id','scene_revision','mapping_id','stimulus_version','channel_profile_id',
              'mode','interaction_id','query_id','probe_id'):r[k]=q[k]
    r.update(event_id='response-'+str(index),response_to=q['event_id'],sequence=index,host_id='bci',
        sent_at=stamp(start+1.1),expires_at=q['expires_at'],clock_offset_ms=0,clock_uncertainty_ms=0)
    r['payload'].update(status=status,symbol=symbol if status=='OBSERVATION' else None)
    if status=='OBSERVATION':
        r['payload']['sample_window'].update(window_id='session-window-'+str(index),recording_id='fixture-recording',
            sample_start=index*1000,sample_end=index*1000+250,started_at=stamp(start),ended_at=stamp(start+1),
            source_kind='offline_replay',sample_rate_hz=250)
    else:r['payload'].update(sample_window=None,raw_scores=None,score_kind=None,calibration_id=None)
    if change:change(r)
    return r

def confirm(s,*,symbol='0',status='OBSERVATION',change=None):
    propose(s)
    q=s.begin_interaction('CONFIRM',now=stamp(.1),monotonic_now=.1)
    r=response(q,symbol=symbol,status=status,change=change)
    result=s.ingest(r,now=stamp(2.2),monotonic_now=2.2)
    return q,r,result

def authorized(**kw):
    s=make_session(**kw);confirm(s);s.authorize(now=stamp(2.3));return s

def observation(t=2.4):
    v=read('configs/execution/fake_vla_interface.json')['fixture_observation'];v['captured_at']=stamp(t)
    return v

def trajectory(*,match=True):
    v=read('examples/scene_outcomes/trajectory.json'); v['split']='test';v['final_predicates'][0]['value']=match
    return ObservedTrajectory(v)

def completed():
    s=authorized();s.dispatch('submission-1',observation(),now=stamp(2.4))
    s.backend.record_feedback('submission-1','STARTED',feedback_id='start-1');s.refresh_submission(now=stamp(3))
    s.backend.record_feedback('submission-1','COMPLETED',feedback_id='complete-1');s.refresh_submission(now=stamp(4))
    return s


def endpoint(s,*,match=True):
    p=s.submission_contract.data['proposal']
    return EndpointEvidence({'evidence_id':'endpoint-1','session_id':s.context['session_id'],
        'submission_id':s.submission['submission_id'],'contract_id':p['contract_id'],'contract_revision':p['revision'],
        'scene_id':p['scene_id'],'scene_revision':p['scene_revision'],
        'observed_at':{'host_id':'fixture-annotator','at':stamp(4.05),'offset_ms':0,'uncertainty_ms':0},
        'source':'simulation_fixture_annotation','trajectory':thaw(trajectory(match=match).data)})


class SessionTests(unittest.TestCase):
    def test_contract_has_explicit_completion_abort_conditions_and_immutable_ids(self):
        s=make_session();c=propose(s)
        self.assertEqual(c.data['completion']['name'],'inside')
        self.assertEqual(c.data['preconditions']['visible_instance_ids'],('grape','bowl-left'))
        with self.assertRaises(TypeError):c.data['proposal']['intent']['source_id']='other'

    def test_proposal_or_task_clarification_is_not_authorization(self):
        s=make_session();propose(s)
        with self.assertRaises(ProtocolError):s.dispatch('x',observation(),now=stamp(.1))
        self.assertEqual(s.backend.submission_count,0)

    def test_observed_accept_then_separate_fake_authorization(self):
        s=make_session();q,r,result=confirm(s)
        self.assertTrue(result.accepted);self.assertEqual(s.state,'AWAITING_AUTHORIZATION')
        self.assertIsNone(s.authorization);self.assertFalse(s.snapshot()['execution_allowed'])
        a=s.authorize(now=stamp(2.3))
        self.assertEqual(a.data['request']['event_id'],q.data['event_id'])
        self.assertEqual(a.data['response']['event_id'],r['event_id'])
        self.assertTrue(a.data['confirmation_is_noisy']);self.assertEqual(s.state,'AUTHORIZED')
        self.assertTrue(a.data['model_refs']['confirmation_sha256'])

    def test_confirmation_semantics_comes_from_explicit_request_mapping(self):
        s=make_session();q,r,_=confirm(s)
        self.assertEqual({o['semantic_ref']:o['symbol'] for o in q.data['payload']['options']}['confirmation:accept'],'0')
        self.assertEqual(r['payload']['symbol'],'0')

    def test_rejection_cannot_be_reasked_even_under_new_contract_id(self):
        s=make_session();confirm(s,symbol='1');self.assertEqual(s.state,'PAUSED')
        with self.assertRaises(ProtocolError):s.authorize(now=stamp(2.3))
        with self.assertRaises(ProtocolError):s.begin_interaction('CONFIRM',now=stamp(3),monotonic_now=3)
        p=s.contract.to_dict()['proposal'];p['contract_id']='another-id'
        s.propose(p,now=stamp(3))
        with self.assertRaises(ProtocolError):s.begin_interaction('CONFIRM',now=stamp(3.1),monotonic_now=3.1)

    def test_omission_invalid_signal_and_timeout_never_default_accept(self):
        for status in ('NO_RESPONSE','INVALID_SIGNAL','NO_CONTROL','TRANSPORT_FAILURE','TIMEOUT'):
            with self.subTest(status=status):
                s=make_session();confirm(s,status=status)
                self.assertEqual(s.state,'PAUSED')
                with self.assertRaises(ProtocolError):s.authorize(now=stamp(2.3))

    def test_score_only_confirm_cannot_authorize(self):
        s=make_session();confirm(s,change=lambda r:r['payload'].update(symbol=None))
        self.assertEqual(s.state,'PAUSED')
        with self.assertRaises(ProtocolError):s.authorize(now=stamp(2.3))

    def test_probe_freezes_dispatch_and_never_changes_contract_into_prompt(self):
        s=make_session();c=propose(s).to_dict()
        q=s.begin_interaction('CHANNEL_PROBE',now=stamp(.1),monotonic_now=.1,prompt_symbol='1',presentation_ack=True,
            presented_at={'host_id':'ui','at':stamp(.1),'offset_ms':0,'uncertainty_ms':0})
        self.assertEqual(s.state,'PROBING');self.assertIsNone(q.data['payload']['contract'])
        with self.assertRaises(ProtocolError):s.dispatch('x',observation(),now=stamp(.2))
        r=response(q,symbol='1');self.assertTrue(s.ingest(r,now=stamp(2.2),monotonic_now=2.2).accepted)
        self.assertEqual(s.contract.to_dict(),c);self.assertIsNone(s.authorization)

    def test_probe_without_actual_presentation_ack_is_not_evidence(self):
        s=make_session();q=s.begin_interaction('CHANNEL_PROBE',now=stamp(.1),monotonic_now=.1,prompt_symbol='0')
        result=s.ingest(response(q),now=stamp(2.2),monotonic_now=2.2)
        self.assertFalse(result.accepted);self.assertIsNone(s.authorization)

    def test_duplicate_confirmation_is_not_consumed_twice(self):
        s=make_session();q,r,_=confirm(s);a=s.authorize(now=stamp(2.3))
        self.assertFalse(s.ingest(r,now=stamp(2.4),monotonic_now=2.4).accepted)
        self.assertEqual(a.to_dict(),s.authorization.to_dict())
        with self.assertRaises(ProtocolError):s.authorize(now=stamp(2.5))

    def test_old_confirmation_after_task_revision_rejected(self):
        s=make_session();propose(s);q=s.begin_interaction('CONFIRM',now=stamp(.1),monotonic_now=.1)
        fresh=dict(s.context,task_revision=2)
        s.revise_context(fresh,s.scene,now=stamp(.2))
        self.assertFalse(s.ingest(response(q),now=stamp(2.2),monotonic_now=2.2).accepted)
        with self.assertRaises(ProtocolError):s.authorize(now=stamp(2.3))
        self.assertFalse(s.snapshot()['generation_valid'])

    def test_scene_change_revokes_authorization_and_missing_target_never_substituted(self):
        s=authorized();scene=s.scene.to_dict();scene['revision']=2;scene['objects'][1]['visible']=False
        s.revise_context(dict(s.context,scene_revision=2),GroundedScene(scene),now=stamp(2.4))
        self.assertIsNone(s.authorization)
        with self.assertRaises(ProtocolError):s.dispatch('x',observation(),now=stamp(2.5))
        self.assertEqual(s.contract.data['proposal']['intent']['destination_id'],'bowl-left')

    def test_declared_model_change_invalidates(self):
        s=authorized();s.invalidate_models(now=stamp(2.4))
        with self.assertRaises(ProtocolError):s.dispatch('x',observation(),now=stamp(2.5))

    def test_external_filter_update_after_confirmation_breaks_freeze(self):
        s=authorized();s.filter._status='paused'
        with self.assertRaises(ProtocolError):s.dispatch('x',observation(),now=stamp(2.4))

    def test_confirmation_and_contract_freshness_enforced(self):
        s=make_session();confirm(s)
        with self.assertRaises(ProtocolError):s.authorize(now=stamp(5))
        s=authorized()
        with self.assertRaises(ProtocolError):s.dispatch('x',observation(10.1),now=stamp(10.1))
        self.assertFalse(s.authorization_consumed)

    def test_unknown_or_low_posterior_does_not_authorize(self):
        initial=read('configs/belief/simulation_binary_filter.json')['initial_joint']
        s=make_session(initial=initial);confirm(s)
        with self.assertRaises(ProtocolError):s.authorize(now=stamp(2.3))

    def test_bad_observation_rejected_before_authorization_consumption(self):
        s=authorized();obs=observation();obs['scene_revision']=2
        with self.assertRaises(ProtocolError):s.dispatch('x',obs,now=stamp(2.4))
        self.assertFalse(s.authorization_consumed);self.assertEqual(s.backend.submission_count,0)

    def test_future_or_stale_observation_refused(self):
        for captured in (0,3):
            s=authorized()
            with self.assertRaises(ProtocolError):s.dispatch('x',observation(captured),now=stamp(2.4))

    def test_transport_acceptance_is_not_execution_or_success(self):
        s=authorized();r=s.dispatch('x',observation(),now=stamp(2.4))
        self.assertEqual(r['status'],'ACCEPTED');self.assertEqual(s.state,'SUBMITTED')
        self.assertTrue(s.authorization_consumed);self.assertFalse(r['actual_execution'])
        with self.assertRaises(ProtocolError):s.verify_endpoint(trajectory(),now=stamp(2.5))

    def test_duplicate_same_submission_is_cached_not_reissued(self):
        s=authorized();obs=observation();s.dispatch('x',obs,now=stamp(2.4));s.dispatch('x',obs,now=stamp(2.5))
        self.assertEqual(s.backend.submission_count,1)
        self.assertEqual(sum(r['kind']=='SUBMIT' for r in s.backend.audit_log),1)
        with self.assertRaises(ProtocolError):s.dispatch('different',obs,now=stamp(2.6))

    def test_same_id_different_payload_rejected(self):
        s=authorized();s.dispatch('x',observation(),now=stamp(2.4))
        with self.assertRaises(ProtocolError):s.dispatch('x',observation(2.5),now=stamp(2.5))

    def test_unknown_submission_no_new_or_automatic_retry(self):
        s=authorized(submit_status='UNKNOWN');s.dispatch('x',observation(),now=stamp(2.4))
        self.assertEqual(s.state,'SUBMISSION_UNKNOWN');s.refresh_submission(now=stamp(2.5))
        with self.assertRaises(ProtocolError):s.dispatch('retry',observation(),now=stamp(2.6))
        s.dispatch('x',observation(),now=stamp(2.7));self.assertEqual(s.backend.submission_count,1)

    def test_submit_exception_remains_unknown_and_not_retryable(self):
        s=authorized()
        def lose(*args):raise ConnectionError('fixture response lost')
        s.backend.submit=lose
        self.assertEqual(s.dispatch('x',observation(),now=stamp(2.4))['status'],'UNKNOWN')
        with self.assertRaises(ProtocolError):s.dispatch('retry',observation(),now=stamp(2.5))

    def test_separate_fake_feedback_and_independent_endpoint_close_legal_flow(self):
        s=completed();self.assertEqual(s.state,'VERIFYING')
        result=s.verify_endpoint(endpoint(s),now=stamp(4.1))
        self.assertTrue(result['endpoint_matches_goal']);self.assertEqual(s.state,'SUCCEEDED')
        self.assertFalse(s.snapshot()['execution_allowed'])

    def test_unknown_or_failed_endpoint_not_success(self):
        for match in (None,False):
            s=completed();s.verify_endpoint(endpoint(s,match=match),now=stamp(4.1))
            self.assertEqual(s.state,'PAUSED' if match is None else 'FAILED')

    def test_backend_failure_does_not_adapt_decoder_or_redefine_goal(self):
        s=authorized();before=s.filter.snapshot();s.dispatch('x',observation(),now=stamp(2.4))
        s.backend.record_feedback('x','FAILED',feedback_id='failed');s.refresh_submission(now=stamp(3))
        self.assertEqual(s.state,'FAILED');self.assertEqual(s.filter.snapshot(),before)

    def test_stop_request_does_not_equal_stopped_feedback(self):
        s=authorized();s.dispatch('x',observation(),now=stamp(2.4));s.request_stop(now=stamp(2.5))
        self.assertEqual(s.stop_feedback,'REQUESTED');self.assertEqual(s.state,'PAUSED')
        self.assertFalse(s.snapshot()['physical_stop_verified'])
        s.backend.record_feedback('x','STOPPED',feedback_id='stopped');s.refresh_submission(now=stamp(3))
        self.assertEqual(s.stop_feedback,'SIMULATED_STOPPED');self.assertEqual(s.state,'CANCELLED')
        self.assertFalse(s.snapshot()['physical_stop_verified'])

    def test_stop_is_latched_and_late_accept_cannot_authorize(self):
        s=make_session();propose(s);q=s.begin_interaction('CONFIRM',now=stamp(.1),monotonic_now=.1)
        s.request_stop(now=stamp(.2))
        self.assertFalse(s.ingest(response(q),now=stamp(2.2),monotonic_now=2.2).accepted)
        with self.assertRaises(ProtocolError):s.authorize(now=stamp(2.3))

    def test_trusted_STOP_bypasses_old_versions_and_expiry(self):
        s=make_session();v=read('examples/protocols/stop.json');v.update(s.context)
        v.update(task_revision=0,scene_revision=0,sent_at=stamp(-5),expires_at=stamp(-4))
        self.assertTrue(s.ingest(v,now=stamp(.1),monotonic_now=.1).accepted)
        self.assertTrue(s.stop_latched)

    def test_pending_expiry_is_explicit_and_never_defaults(self):
        s=make_session();propose(s);s.begin_interaction('CONFIRM',now=stamp(.1),monotonic_now=.1,expires_in_s=1)
        s.expire(now=stamp(1.1),monotonic_now=1.1)
        self.assertEqual(s.state,'PAUSED');self.assertIsNone(s.authorization)

    def test_dry_run_validates_without_backend_submission(self):
        s=authorized(dry=True);r=s.dispatch('x',observation(),now=stamp(2.4))
        self.assertEqual(r['reason'],'dry_run_validation_only');self.assertEqual(s.backend.submission_count,0)

    def test_live_mode_refused(self):
        s=authorized()
        with self.assertRaises(ProtocolError):s.dispatch('x',observation(),now=stamp(2.4),execution_mode='live')

    def test_policy_planning_copy_predicts_latency_without_mutating_filter(self):
        s=make_session(dynamic=True);before=s.filter.snapshot()
        r=s.request_policy(remaining_s=10,scene_valid_until_s=BASE.timestamp()+60,now=stamp(.5))
        self.assertEqual(s.filter.snapshot(),before);self.assertFalse(r['execution_allowed'])
        self.assertEqual(s.journal.records[-1]['payload']['event']['planning_prediction_gap_s'],.5)

    def test_replay_exact_snapshot_without_backend_calls(self):
        s=completed();s.verify_endpoint(endpoint(s),now=stamp(4.1))
        recovered=EventJournal.from_jsonl(s.journal.to_jsonl())
        def forbidden(*args):raise AssertionError('replay must not submit')
        s.backend.submit=forbidden
        self.assertEqual(replay_session_state(recovered),s.snapshot())
        kinds={r['kind'] for r in recovered.records}
        self.assertTrue({'INPUT_RECEIVED','SUBMISSION_INTENT','SUBMISSION_RETURNED','SIMULATED_EXECUTION_FEEDBACK','ENDPOINT_JUDGED'}<=kinds)

    def test_error_confirmation_can_be_counted_as_wrong_start_independently(self):
        s=completed();truth_goal=read('configs/belief/simulation_binary_filter.json')['vocabulary']['goals'][1]
        before=s.filter.snapshot();result=evaluate_trajectory(trajectory(),[truth_goal])
        self.assertEqual(result[truth_goal['goal_id']]['components']['wrong_start'],1)
        self.assertEqual(s.filter.snapshot(),before)


if __name__=='__main__':unittest.main()
