"""Validate immutable authorization and endpoint evidence, without a session."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from bci_vla.contracts.records import AuthorizationRecord, EndpointEvidence
from bci_vla.protocols import ProtocolError
from bci_vla.scene.tasks import goal_id

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))


def authorization():
    q=read('examples/protocols/confirm_request.json')
    r=read('examples/protocols/confirm_response.json')
    q['channel_profile_id']=r['channel_profile_id']='simulation-binary-v1'
    p=q['payload']['contract']; canonical=goal_id(p['intent'])
    belief={'goal_ids':[canonical,'unknown'],'mode_ids':['normal','flip'],
            'mass':[{'goal_id':canonical,'mode_id':'normal','probability':.8},
                    {'goal_id':'unknown','mode_id':'normal','probability':.2}],
            'as_of_event_id':r['event_id']}
    predicted=deepcopy(belief)
    predicted['as_of_event_id']='authorization-prediction'
    predicted['mass'][0]['probability']=.7
    predicted['mass'][1]['probability']=.3
    return {'authorization_id':'fixture-authorization','scope':'offline_fake_only',
            'contract_sha256':'a'*64,'contract_id':p['contract_id'],'task_id':p['task_id'],
            'task_revision':p['revision'],'scene_id':p['scene_id'],'scene_revision':p['scene_revision'],
            'request':q,'response':r,
            'model_refs':dict(channel_profile='simulation-binary-v1',channel_sha256='b'*64,
                              transition_sha256='c'*64,confirmation_sha256='d'*64,
                              policy_sha256='e'*64,backend_spec_sha256='f'*64),
            'issued_at':'2026-10-15T00:00:03.200000Z','valid_until':'2026-10-15T00:00:08Z',
            'belief_at_confirmation':belief,'belief_at_authorization':predicted,
            'authorization_prediction_gap_s':1.2,'goal_probability':.7,
            'confirmation_is_noisy':True,'frozen_until_dispatch':True,'consume_once':True,
            'physical_execution_allowed':False}


def endpoint():
    trajectory=read('examples/scene_outcomes/trajectory.json');trajectory['split']='test'
    return {'evidence_id':'endpoint-1','session_id':'fixture-session','submission_id':'submission-1',
            'contract_id':'fixture-contract','contract_revision':1,'scene_id':'fixture-scene','scene_revision':1,
            'observed_at':{'host_id':'annotation','at':'2026-10-15T00:00:05Z',
                           'offset_ms':0,'uncertainty_ms':0},
            'source':'simulation_fixture_annotation','trajectory':trajectory}


class AuthorizationRecordTests(unittest.TestCase):
    def test_valid_noisy_accept_retains_confirmation_and_predicted_belief(self):
        value=authorization();record=AuthorizationRecord(value)
        self.assertEqual(record.to_dict(),value)
        self.assertEqual(record.data['goal_probability'],.7)
        self.assertEqual(record.data['belief_at_confirmation']['mass'][0]['probability'],.8)
        with self.assertRaises(TypeError):record.data['goal_probability']=1

    def test_every_response_binding_field_is_checked(self):
        for key in ('session_id','task_id','task_revision','scene_id','scene_revision','mapping_id',
                    'stimulus_version','channel_profile_id','interaction_id','query_id'):
            value=authorization()
            value['response'][key]=2 if key.endswith('revision') else 'different'
            with self.subTest(key=key),self.assertRaises(ProtocolError):AuthorizationRecord(value)

    def test_top_level_task_scene_and_contract_must_match_confirmation(self):
        for key in ('contract_id','task_id','task_revision','scene_id','scene_revision'):
            value=authorization();value[key]=2 if key.endswith('revision') else 'different'
            with self.subTest(key=key),self.assertRaises(ProtocolError):AuthorizationRecord(value)

    def test_raw_only_live_or_reject_selection_cannot_authorize(self):
        for change in ('raw_only','live','reject','no_response'):
            value=authorization();payload=value['response']['payload']
            if change=='raw_only':payload['symbol']=None
            elif change=='live':payload['sample_window']['source_kind']='live_bci'
            elif change=='reject':payload['symbol']='0'
            else:payload.update(status='NO_RESPONSE',symbol=None)
            with self.subTest(change=change),self.assertRaises(ProtocolError):AuthorizationRecord(value)

    def test_accept_semantic_must_be_unique(self):
        for semantic in ('confirmation:accept','confirmation:reject'):
            value=authorization()
            for option in value['request']['payload']['options']:option['semantic_ref']=semantic
            with self.subTest(semantic=semantic),self.assertRaisesRegex(ProtocolError,'actual_accept_selection_required'):
                AuthorizationRecord(value)

    def test_authorization_cannot_outlive_any_evidence_or_contract_expiry(self):
        for target in ('request','response','contract'):
            value=authorization()
            if target=='contract':value['request']['payload']['contract']['valid_until']='2026-10-15T00:00:07Z'
            else:value[target]['expires_at']='2026-10-15T00:00:07Z'
            with self.subTest(target=target),self.assertRaisesRegex(ProtocolError,'authorization_time_outside_confirmation'):
                AuthorizationRecord(value)

    def test_issue_time_cannot_precede_response_or_follow_expiry(self):
        for at in ('2026-10-15T00:00:02.5Z','2026-10-15T00:00:09Z'):
            value=authorization();value['issued_at']=at
            with self.subTest(at=at),self.assertRaises(ProtocolError):AuthorizationRecord(value)

    def test_sample_window_must_be_after_request_and_before_response(self):
        for key,at in (('sent_at','2026-10-15T00:00:01.5Z'),('sent_at','2026-10-15T00:00:02.5Z')):
            value=authorization()
            target='request' if at.endswith('01.5Z') else 'response'
            value[target][key]=at
            # For the response variant, move the window beyond its send time.
            if target=='response':
                value['response']['payload']['sample_window'].update(started_at='2026-10-15T00:00:02Z',
                                                                     ended_at='2026-10-15T00:00:03Z')
            with self.subTest(target=target),self.assertRaisesRegex(ProtocolError,'authorization_time_outside_confirmation'):
                AuthorizationRecord(value)

    def test_nonzero_response_clock_uncertainty_is_not_an_exact_prediction_origin(self):
        value=authorization();value['response']['clock_uncertainty_ms']=1
        with self.assertRaisesRegex(ProtocolError,'authorization_requires_exact_reference_window'):
            AuthorizationRecord(value)

    def test_prediction_gap_is_measured_from_reference_window_end(self):
        value=authorization();value['authorization_prediction_gap_s']=0
        with self.assertRaisesRegex(ProtocolError,'authorization_prediction_gap_mismatch'):AuthorizationRecord(value)

    def test_goal_probability_uses_predicted_belief_not_confirmation_belief(self):
        value=authorization();value['goal_probability']=.8
        with self.assertRaisesRegex(ProtocolError,'authorization_goal_probability_mismatch'):AuthorizationRecord(value)

    def test_confirmation_belief_must_bind_actual_response(self):
        value=authorization();value['belief_at_confirmation']['as_of_event_id']='different-event'
        with self.assertRaisesRegex(ProtocolError,'confirmation_belief_evidence_mismatch'):AuthorizationRecord(value)

    def test_canonical_goal_and_axes_are_checked(self):
        for change in ('axes','canonical'):
            value=authorization()
            if change=='axes':value['belief_at_authorization']['mode_ids'].append('unknown_channel')
            else:
                for field in ('belief_at_confirmation','belief_at_authorization'):
                    value[field]['goal_ids'][0]='unbound-goal';value[field]['mass'][0]['goal_id']='unbound-goal'
            with self.subTest(change=change),self.assertRaises(ProtocolError):AuthorizationRecord(value)

    def test_model_refs_are_specific_fingerprints_and_bind_channel_profile(self):
        for change in ('missing','profile','hash','extra'):
            value=authorization()
            if change=='missing':del value['model_refs']['confirmation_sha256']
            elif change=='profile':value['model_refs']['channel_profile']='different'
            elif change=='hash':value['model_refs']['confirmation_sha256']='not-a-version'
            else:value['model_refs']['new_unbound_model']='value'
            with self.subTest(change=change),self.assertRaises(ProtocolError):AuthorizationRecord(value)

    def test_missing_prediction_and_unfrozen_or_live_authority_rejected(self):
        for key in ('belief_at_authorization','authorization_prediction_gap_s','consume_once','physical_execution_allowed'):
            value=authorization()
            if key in {'belief_at_authorization','authorization_prediction_gap_s'}:del value[key]
            else:value[key]=key=='physical_execution_allowed'
            with self.subTest(key=key),self.assertRaises(ProtocolError):AuthorizationRecord(value)


class EndpointEvidenceTests(unittest.TestCase):
    def test_test_or_calibration_holdout_fixture_is_immutable(self):
        for split in ('test','calibration_holdout'):
            value=endpoint();value['trajectory']['split']=split
            record=EndpointEvidence(value)
            self.assertEqual(record.to_dict(),value)
            with self.assertRaises(TypeError):record.data['source']='different'

    def test_training_calibration_fixture_cannot_validate_its_own_endpoint(self):
        value=endpoint();value['trajectory']['split']='calibration'
        with self.assertRaisesRegex(ProtocolError,'endpoint_validation_split_required'):EndpointEvidence(value)

    def test_source_and_trajectory_provenance_must_be_offline_annotation(self):
        for change in ('source','provenance'):
            value=endpoint()
            if change=='source':value['source']='model_self_report'
            else:value['trajectory'].update(provenance='independent_annotation',checkpoint_ref='real-unverified')
            with self.subTest(change=change),self.assertRaises(ProtocolError):EndpointEvidence(value)

    def test_observation_clock_requires_aware_timestamp_and_declared_uncertainty(self):
        for change in ('naive','negative_uncertainty','missing_offset'):
            value=endpoint()
            if change=='naive':value['observed_at']['at']='2026-10-15T00:00:05'
            elif change=='negative_uncertainty':value['observed_at']['uncertainty_ms']=-1
            else:del value['observed_at']['offset_ms']
            with self.subTest(change=change),self.assertRaises(ProtocolError):EndpointEvidence(value)

    def test_binding_fields_and_trajectory_structure_required(self):
        for change in ('submission','revision','trajectory'):
            value=endpoint()
            if change=='submission':value['submission_id']=''
            elif change=='revision':value['contract_revision']=0
            else:del value['trajectory']['final_predicates']
            with self.subTest(change=change),self.assertRaises(ProtocolError):EndpointEvidence(value)


if __name__=='__main__':unittest.main()
