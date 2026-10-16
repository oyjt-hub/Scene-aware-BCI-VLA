"""Independent regressions for cross-episode evidence and authorization aging.

All cases use declared finite fixtures. They do not contact a backend or provide
evidence that an EEG decoder, policy checkpoint, or physical robot is working.
"""
from copy import deepcopy
import math
import unittest

from bci_vla.adapters.vla import FakeVLAAdapter
from bci_vla.contracts.records import AuthorizationRecord, EndpointEvidence
from bci_vla.contracts.session import InteractionSession
from bci_vla.core.diagnostic_policy import DiagnosticPolicy
from bci_vla.core.joint_belief import GoalVocabulary, JointBeliefFilter
from bci_vla.models.channel import ChannelModel, JointTransitionModel
from bci_vla.models.consequence import ObservedTrajectory, OutcomeTable
from bci_vla.protocols import JointBelief, ProtocolError
from bci_vla.scene.state import GroundedScene
from tests.test_execution_session import (
    BASE, authorized, completed, confirm, observation, read, stamp,
)


def raw_endpoint_trajectory():
    value = read('examples/scene_outcomes/trajectory.json')
    value['split'] = 'test'
    return value


def endpoint_payload(session, **changes):
    proposal = session.submission_contract.data['proposal']
    value = {
        'evidence_id': 'bound-endpoint-1',
        'session_id': session.context['session_id'],
        'submission_id': session.submission['submission_id'],
        'contract_id': proposal['contract_id'],
        'contract_revision': proposal['revision'],
        'scene_id': proposal['scene_id'],
        'scene_revision': proposal['scene_revision'],
        'observed_at': {
            'host_id': 'fixture-annotator', 'at': stamp(4.05),
            'offset_ms': 0, 'uncertainty_ms': 0,
        },
        'source': 'simulation_fixture_annotation',
        'trajectory': raw_endpoint_trajectory(),
    }
    value.update(changes)
    return value


def changing_goal_session():
    """A declared model in which a recent confirmation can age below threshold."""
    value = read('configs/belief/simulation_binary_filter.json')
    vocabulary = GoalVocabulary(value['vocabulary'])
    initial = deepcopy(value['initial_joint'])
    initial['mass'] = [{'goal_id': vocabulary.known_goal_ids[0],
                        'mode_id': 'normal', 'probability': 1.0}]
    channel = ChannelModel(read('configs/channels/simulation_binary_v1.json'))
    transition_config = read('examples/joint_belief/fixed_state_transition.json')
    transition_config['goal_rate_per_s'] = 0.3
    transition = JointTransitionModel(transition_config)
    belief_filter = JointBeliefFilter(
        JointBelief(initial), vocabulary=vocabulary, channel_model=channel,
        transition_model=transition, current_context=value['current_context'],
        layout_id=value['layout_id'], initial_at=BASE,
    )
    policy = DiagnosticPolicy(
        read('configs/diagnosis.json'), channel_model=channel,
        transition_model=transition,
        outcome_table=OutcomeTable(read('configs/consequences/simulation_canonical_outcomes.json')),
    )
    return InteractionSession(
        current_context=value['current_context'],
        scene=GroundedScene(read('examples/scene_outcomes/scene.json')),
        belief_filter=belief_filter, policy=policy, backend=FakeVLAAdapter(),
        config=read('configs/execution/session.json'), started_at=BASE,
    )


class EndpointBindingRegressions(unittest.TestCase):
    def assert_refused_without_success(self, session, value, now=4.1):
        before = session.filter.snapshot()
        with self.assertRaises(ProtocolError):
            session.verify_endpoint(value, now=stamp(now))
        self.assertEqual(session.state, 'VERIFYING')
        self.assertEqual(session.filter.snapshot(), before)
        self.assertEqual(session.backend.submission_count, 1)

    def test_old_unbound_trajectory_cannot_close_current_submission(self):
        session = completed()
        self.assert_refused_without_success(
            session, ObservedTrajectory(raw_endpoint_trajectory()),
        )

    def test_other_session_submission_or_contract_is_not_current_result(self):
        for change in ({'session_id': 'earlier-session'},
                       {'submission_id': 'earlier-submission'},
                       {'contract_id': 'earlier-contract'},
                       {'contract_revision': 2}):
            with self.subTest(change=change):
                session = completed()
                evidence = EndpointEvidence(endpoint_payload(session, **change))
                self.assert_refused_without_success(session, evidence)

    def test_scene_identity_or_revision_mismatch_cannot_close_task(self):
        for change in ({'scene_id': 'different-scene'}, {'scene_revision': 2}):
            with self.subTest(change=change):
                session = completed()
                evidence = EndpointEvidence(endpoint_payload(session, **change))
                self.assert_refused_without_success(session, evidence)

    def test_pre_completion_or_future_observation_is_rejected(self):
        for observed in (3.99, 4.2):
            with self.subTest(observed=observed):
                session = completed()
                value = endpoint_payload(session)
                value['observed_at']['at'] = stamp(observed)
                self.assert_refused_without_success(session, EndpointEvidence(value))

    def test_clock_uncertainty_must_fit_entire_completed_observation_interval(self):
        for center in (4.01, 4.09):
            with self.subTest(center=center):
                session = completed()
                value = endpoint_payload(session)
                value['observed_at'].update(at=stamp(center), uncertainty_ms=20)
                self.assert_refused_without_success(session, EndpointEvidence(value))

    def test_calibration_trajectory_cannot_be_relabeled_current_endpoint_evidence(self):
        session = completed()
        value = endpoint_payload(session)
        value['trajectory']['split'] = 'calibration'
        with self.assertRaises(ProtocolError):
            EndpointEvidence(value)
        self.assertEqual(session.state, 'VERIFYING')

    def test_current_bound_independent_observation_closes_once(self):
        session = completed()
        evidence = EndpointEvidence(endpoint_payload(session))
        result = session.verify_endpoint(evidence, now=stamp(4.1))
        self.assertTrue(result['endpoint_matches_goal'])
        self.assertEqual(session.state, 'SUCCEEDED')
        self.assertFalse(session.snapshot()['execution_allowed'])
        with self.assertRaises(ProtocolError):
            session.verify_endpoint(evidence, now=stamp(4.2))


class AuthorizationAgingRegressions(unittest.TestCase):
    def test_authorization_uses_predicted_goal_probability_after_confirmation_window(self):
        session = changing_goal_session()
        confirm(session)
        before = session.filter.snapshot()
        goal = session.contract.data['goal_id']
        confirmation_probability = before['goal_marginals'][goal]
        predicted = 0.5 + (confirmation_probability - 0.5) * math.exp(-2 * 0.3 * 1.99)
        self.assertGreater(confirmation_probability, 0.8)
        self.assertLess(predicted, 0.8)
        with self.assertRaises(ProtocolError):
            session.authorize(now=stamp(3.99))
        self.assertIsNone(session.authorization)
        self.assertEqual(session.filter.snapshot(), before)
        self.assertEqual(session.backend.submission_count, 0)

    def test_dispatch_rechecks_prediction_without_consuming_expired_probability(self):
        session = changing_goal_session()
        confirm(session)
        session.authorize(now=stamp(2.3))
        before = session.filter.snapshot()
        self.assertFalse(session.authorization_consumed)
        with self.assertRaises(ProtocolError):
            session.dispatch('late-dispatch', observation(3.99), now=stamp(3.99))
        self.assertFalse(session.authorization_consumed)
        self.assertEqual(session.backend.submission_count, 0)
        self.assertEqual(session.filter.snapshot(), before)

    def test_authorization_preserves_evidence_and_records_separate_predicted_belief(self):
        session = changing_goal_session()
        confirm(session)
        before = session.filter.snapshot()
        authorization = session.authorize(now=stamp(2.3)).to_dict()
        goal = session.contract.data['goal_id']
        expected = 0.5 + (before['goal_marginals'][goal] - 0.5) * math.exp(-2 * 0.3 * 0.3)
        predicted = JointBelief(authorization['belief_at_authorization']).to_dict()
        marginal = sum(row['probability'] for row in predicted['mass'] if row['goal_id'] == goal)
        self.assertAlmostEqual(authorization['authorization_prediction_gap_s'], 0.3)
        self.assertAlmostEqual(authorization['goal_probability'], expected)
        self.assertAlmostEqual(marginal, expected)
        self.assertEqual(authorization['belief_at_confirmation'], before['belief'])
        self.assertEqual(session.filter.snapshot(), before)


class AuthorizationRecordBindingRegressions(unittest.TestCase):
    def test_response_header_cannot_mix_task_scene_or_mapping_versions(self):
        for key, bad in (('session_id', 'other-session'), ('task_revision', 2),
                         ('scene_revision', 2), ('mapping_id', 'other-mapping')):
            with self.subTest(key=key):
                value = authorized().authorization.to_dict()
                value['response'][key] = bad
                with self.assertRaises(ProtocolError):
                    AuthorizationRecord(value)

    def test_top_level_versions_must_match_evidence_and_contract(self):
        for key, bad in (('task_id', 'other-task'), ('task_revision', 2),
                         ('scene_id', 'other-scene'), ('scene_revision', 2)):
            with self.subTest(key=key):
                value = authorized().authorization.to_dict()
                value[key] = bad
                with self.assertRaises(ProtocolError):
                    AuthorizationRecord(value)

    def test_reject_symbol_cannot_be_imported_as_accept_authorization(self):
        value = authorized().authorization.to_dict()
        value['response']['payload']['symbol'] = '1'
        with self.assertRaises(ProtocolError):
            AuthorizationRecord(value)

    def test_authorization_cannot_predate_confirmation_or_outlive_contract(self):
        for key, bad in (('issued_at', stamp(1.9)), ('valid_until', stamp(61))):
            with self.subTest(key=key):
                value = authorized().authorization.to_dict()
                value[key] = bad
                with self.assertRaises(ProtocolError):
                    AuthorizationRecord(value)

    def test_record_probability_must_match_authorization_belief(self):
        value = authorized().authorization.to_dict()
        value['goal_probability'] = 0.01
        with self.assertRaises(ProtocolError):
            AuthorizationRecord(value)


if __name__ == '__main__':
    unittest.main()
