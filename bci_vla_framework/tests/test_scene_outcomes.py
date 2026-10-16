"""Identity, intent preservation and outcome-evidence regressions; no robot trials."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from bci_vla.protocols import ProtocolError, TaskContract, ExecutionEvent
from bci_vla.scene.state import GroundedScene, IdentityLedger, from_detections
from bci_vla.scene.tasks import BackendCapabilities, task_candidates, clarify_slots, revalidate_contract, TaskHistory
from bci_vla.models.consequence import ObservedTrajectory, evaluate_trajectory, estimate_components, OutcomeTable

ROOT=Path(__file__).resolve().parents[1]
def load(p): return json.loads((ROOT/p).read_text(encoding='utf-8'))
def scene(): return GroundedScene(load('examples/scene_outcomes/scene.json'))
def backend(): return BackendCapabilities(load('configs/consequences/simulation_backend.json'))
def goals(): return load('examples/scene_outcomes/goal_hypotheses.json')
def trajectory(): return ObservedTrajectory(load('examples/scene_outcomes/trajectory.json'))
def contract(status='proposed'):
    return TaskContract({'contract_id':'fixture-contract','task_id':'fixture-task','revision':1,'scene_id':'fixture-scene','scene_revision':1,'intent':goals()[0]['intent'],'status':status,'valid_until':'2026-10-15T00:01:00Z','confirmation_event_id':'external-authorization' if status=='confirmed' else None})


class SceneTests(unittest.TestCase):
    def test_pixel_coordinates_and_stable_ids_survive_reordering(self):
        v=scene().to_dict(); detections=[]
        for o in reversed(v['objects']):
            obj=deepcopy(o); obj['box_xyxy_px']=[x*1000 for x in obj.pop('box_xyxy')]; detections.append(obj)
        s=from_detections(scene_id=v['scene_id'],revision=2,observed_at=v['observed_at'],session_id=v['identity']['session_id'],tracker_ref=v['identity']['tracker_ref'],frame_size=[1000,1000],detections=detections)
        self.assertEqual({o['instance_id']:list(o['box_xyxy']) for o in s.data['objects']},{o['instance_id']:o['box_xyxy'] for o in v['objects']})
        self.assertTrue(all(e['predicate'] in {'left_of','right_of','above','below'} for e in s.data['relations']))
        self.assertFalse(any(e['predicate']=='inside' for e in s.data['relations']))
        ledger=IdentityLedger();ledger.observe(scene());ledger.observe(s)

    def test_duplicate_ids_missing_identity_and_bad_coordinates_rejected(self):
        for case in ['duplicate','missing','outside','units']:
            v=scene().to_dict()
            if case=='duplicate':v['objects'][1]['instance_id']='grape'
            if case=='missing':del v['objects'][0]['track_token']
            if case=='outside':v['objects'][0]['box_xyxy'][0]=-1
            if case=='units':v['coordinate_frame']['unit']='meters'
            with self.subTest(case=case):
                with self.assertRaises(ProtocolError):GroundedScene(v)

    def test_identity_reassignment_across_snapshots_is_rejected(self):
        a=scene();v=a.to_dict();v['revision']=2;v['objects'][0]['track_token']='another-object'
        ledger=IdentityLedger();ledger.observe(a)
        with self.assertRaisesRegex(ProtocolError,'identity_reassigned'):ledger.observe(GroundedScene(v))
        with self.assertRaisesRegex(ProtocolError,'identity_reassigned'):revalidate_contract(contract(),a,GroundedScene(v),backend())

    def test_occlusion_is_not_silently_a_missing_or_new_instance(self):
        v=scene().to_dict();v['revision']=2;v['objects'][0].update(visible=False,box_xyxy=None)
        s=GroundedScene(v)
        self.assertEqual(s.data['objects'][0]['instance_id'],'grape')
        self.assertFalse(any(c['kind']=='task' and c['intent']['source_id']=='grape' for c in task_candidates(s,backend())))
        self.assertEqual(revalidate_contract(contract(),scene(),s,backend())['status'],'needs_requery')

    def test_none_unknown_and_all_destinations_are_retained(self):
        b=BackendCapabilities(load('configs/consequences/real_backend_unverified.json'))
        candidates=task_candidates(scene(),b)
        self.assertEqual({c['kind'] for c in candidates},{'unknown','none','task'})
        place=[c for c in candidates if c['kind']=='task' and c['intent']['action']=='place']
        self.assertEqual({c['intent']['destination_id'] for c in place},{'bowl-left','bowl-right'})
        self.assertTrue(all(c['capability']['eligibility']=='unknown' for c in place))
        reordered=scene().to_dict();reordered['objects'].reverse()
        self.assertEqual({c['goal_id'] for c in candidates},{c['goal_id'] for c in task_candidates(GroundedScene(reordered),b)})

    def test_selection_requires_explicit_action_and_destination(self):
        a=clarify_slots(scene(),{'action':None,'source_id':'grape','destination_id':None})
        self.assertEqual((a['status'],a['slot']),('clarify','action'))
        d=clarify_slots(scene(),{'action':'place','source_id':'grape','destination_id':None})
        self.assertEqual(d['slot'],'destination')
        p=clarify_slots(scene(),goals()[0]['intent'])
        self.assertEqual(p['status'],'proposal');self.assertIn('leftmost bowl',p['instruction'])

    def test_coincident_same_class_objects_are_not_described_by_list_order(self):
        v=scene().to_dict();v['objects'][2]['box_xyxy']=list(v['objects'][1]['box_xyxy'])
        result=clarify_slots(GroundedScene(v),goals()[0]['intent'])
        self.assertEqual(result['reason'],'language_reference_ambiguous')

    def test_contract_scene_changes_invalidate_without_rewriting_intent(self):
        a=scene();v=a.to_dict();v['revision']=2;v['objects'][1]['box_xyxy']=[0.2,0.2,0.4,0.4]
        c=contract('confirmed');before=c.to_dict()
        result=revalidate_contract(c,a,GroundedScene(v),backend())
        self.assertEqual(result['status'],'needs_reconfirmation');self.assertFalse(result['execute_allowed'])
        self.assertEqual(c.to_dict(),before)
        v['revision']=1
        with self.assertRaisesRegex(ProtocolError,'scene_version_reused'):revalidate_contract(c,a,GroundedScene(v),backend())

    def test_folding_remains_extension_and_live_capability_not_claimed(self):
        intent={'action':'fold','source_id':'cloth','destination_id':None}
        self.assertEqual(backend().assess(intent)['eligibility'],'unknown')
        self.assertEqual(backend().assess(goals()[0]['intent'],execution_mode='live')['eligibility'],'unknown')

    def test_history_proposals_confirmations_and_progress_are_separate(self):
        h=TaskHistory();h.record_proposal(contract());self.assertEqual(len(h.confirmed),0)
        with self.assertRaisesRegex(ProtocolError,'validated_contract_required'):h.record_proposal(contract().to_dict())
        confirmed=contract('confirmed')
        with self.assertRaisesRegex(ProtocolError,'external_confirmation_required'):h.record_confirmed(confirmed,authorization_ref='wrong')
        h.record_confirmed(confirmed,authorization_ref='external-authorization')
        event=ExecutionEvent({'execution_id':'e','contract_id':'fixture-contract','contract_revision':1,'stage':'command_dispatched','at':scene().to_dict()['observed_at'],'evidence_ref':None})
        h.record_progress(event)
        self.assertEqual((len(h.proposals),len(h.confirmed),len(h.progress)),(1,1,1))
        self.assertEqual(h.progress[0].data['stage'],'command_dispatched')
        with self.assertRaisesRegex(ProtocolError,'duplicate_confirmation'):h.record_confirmed(confirmed,authorization_ref='external-authorization')


class OutcomeTests(unittest.TestCase):
    def test_one_observed_trajectory_evaluates_multiple_hypotheses(self):
        v=evaluate_trajectory(trajectory(),goals())
        self.assertTrue(v['old']['endpoint_matches_goal']);self.assertFalse(v['new']['endpoint_matches_goal'])
        self.assertEqual(v['new']['components']['wrong_start'],1)
        self.assertIsNone(v['unknown']['endpoint_matches_goal']);self.assertIsNone(v['unknown']['components']['wrong_start'])
        self.assertEqual(v['none']['components']['wrong_start'],1)

    def test_missing_predicate_and_other_goals_recovery_remain_unknown(self):
        raw=load('examples/scene_outcomes/trajectory.json');raw['final_predicates']=raw['final_predicates'][:1]
        v=evaluate_trajectory(ObservedTrajectory(raw),goals())
        self.assertIsNone(v['new']['components']['task_failure']);self.assertIsNone(v['new']['components']['recovery_s'])
        self.assertEqual(v['old']['components']['recovery_s'],0)

    def test_calibration_is_conditioned_on_backend_and_command(self):
        kwargs={'decision_intent':goals()[0]['intent'],'executor_id':'simulation-pi05-interface','checkpoint_ref':None,'predicate_version':'endpoint-v1'}
        v=estimate_components([trajectory()],goals()[0],**kwargs)
        self.assertEqual(v['means']['wrong_start'],0);self.assertEqual(v['component_counts']['recovery_s'],1)
        self.assertFalse(v['deployment_allowed']);self.assertEqual(v['status'],'simulation_only')
        empty=estimate_components([],goals()[0],**kwargs)
        self.assertEqual(empty['status'],'unavailable');self.assertTrue(all(v is None for v in empty['means'].values()))
        with self.assertRaisesRegex(ProtocolError,'duplicate_trajectory'):estimate_components([trajectory(),trajectory()],goals()[0],**kwargs)
        wrong=deepcopy(kwargs);wrong['decision_intent']=goals()[1]['intent']
        with self.assertRaisesRegex(ProtocolError,'action_calibration_mismatch'):estimate_components([trajectory()],goals()[0],**wrong)
        wrong=deepcopy(kwargs);wrong['executor_id']='other-backend'
        with self.assertRaisesRegex(ProtocolError,'backend_calibration_mismatch'):estimate_components([trajectory()],goals()[0],**wrong)

    def test_test_trajectories_and_llm_success_cannot_calibrate(self):
        raw=load('examples/scene_outcomes/trajectory.json');raw['split']='test'
        with self.assertRaisesRegex(ProtocolError,'test_trajectory_cannot_calibrate'):
            estimate_components([ObservedTrajectory(raw)],goals()[0],decision_intent=goals()[0]['intent'],executor_id='simulation-pi05-interface',checkpoint_ref=None,predicate_version='endpoint-v1')
        raw['provenance']='vla_self_confidence'
        with self.assertRaisesRegex(ProtocolError,'independent_endpoint_source_required'):ObservedTrajectory(raw)

    def test_real_trajectories_need_identified_checkpoint(self):
        raw=load('examples/scene_outcomes/trajectory.json');raw['provenance']='independent_annotation'
        with self.assertRaisesRegex(ProtocolError,'checkpoint_required'):ObservedTrajectory(raw)

    def test_simulation_cost_keeps_components_and_explicit_weights(self):
        t=OutcomeTable(load('configs/consequences/simulation_outcome_table.json'));weights=load('configs/consequences/simulation_weight_sets.json')['weight_sets']
        a=t.score('KEEP_OLD','old','normal',weights['base'])
        self.assertAlmostEqual(a['scalar_total'],0.24);self.assertEqual(a['components']['task_failure'],0.2)
        self.assertEqual(len(t.sensitivity('ACCEPT_NEW','old','normal',weights)),3)
        with self.assertRaisesRegex(ProtocolError,'unknown_outcome_state'):t.get('ACCEPT_NEW','unknown','normal')
        with self.assertRaisesRegex(ProtocolError,'cannot_run_live'):t.get('ACCEPT_NEW','old','normal',execution_mode='live')

    def test_wait_or_check_cost_does_not_imply_zero_future_failure(self):
        t=OutcomeTable(load('configs/consequences/simulation_outcome_table.json'));w=load('configs/consequences/simulation_weight_sets.json')['weight_sets']['base']
        for action in ['WAIT','CHECK_CHANNEL','REQUEST_RECALIBRATION']:
            v=t.score(action,'old','normal',w)
            self.assertIsNone(v['scalar_total']);self.assertIsNone(v['components']['task_failure'])
            self.assertIsNotNone(v['known_immediate_user_cost'])
        raw=load('configs/consequences/simulation_outcome_table.json')
        next(row for row in raw['rows'] if row['stage']=='continuation')['components']['task_failure']=0
        with self.assertRaisesRegex(ProtocolError,'continuation_future_outcome_unknown'):OutcomeTable(raw)

    def test_real_cost_table_and_unknown_goal_never_default_to_zero(self):
        with self.assertRaisesRegex(ProtocolError,'real_consequence_calibration_unavailable'):
            OutcomeTable(load('configs/consequences/real_outcome_uncalibrated.template.json'))
        raw=load('configs/consequences/simulation_outcome_table.json');raw['rows'][0]['components']['task_failure']=None
        t=OutcomeTable(raw);w=load('configs/consequences/simulation_weight_sets.json')['weight_sets']['base']
        self.assertIsNone(t.score('ACCEPT_NEW','old','normal',w)['scalar_total'])


class ImportTests(unittest.TestCase):
    def test_scene_and_consequence_import_without_device_or_network(self):
        script='''
import builtins,socket
original=builtins.__import__
def guard(name,*a,**kw):
 if name.split('.')[0] in {'cv2','torch','jax','mne','pygame','piper_sdk','requests','paramiko','numpy'}: raise AssertionError('device/model dependency')
 return original(name,*a,**kw)
builtins.__import__=guard
socket.socket=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('network forbidden'))
import bci_vla.scene.state,bci_vla.scene.tasks,bci_vla.models.consequence
'''
        v=subprocess.run([sys.executable,'-B','-c',script],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(v.returncode,0,v.stderr)
