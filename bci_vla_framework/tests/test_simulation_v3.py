"""Explicit toy engine tests; actual LIBERO evidence is produced separately."""
import unittest
from copy import deepcopy
from unittest.mock import patch
from bci_vla.protocols import ProtocolError,thaw
from bci_vla.benchmark.offline import stamp
from bci_vla.simulation_v3.backend import SimulationVLAAdapter
from bci_vla.simulation_v3.factory import make_runtime,empirical_outcomes,OBJECTS
from bci_vla.simulation_v3.interaction import resolve,dispatch,endpoint
from bci_vla.simulation_v3.replay import replay_session_state

class ToyEngine:
    checkpoint='c'*64
    finished=False
    def observation_digests(self):return ['a'*64,'b'*64],'d'*64
    def observation(self,context,at):
        images,state=self.observation_digests()
        return dict(interface_version='libero-model/3.0',scene_id=context['scene_id'],scene_revision=context['scene_revision'],
            captured_at=at,image_sha256=images,sim_state_sha256=state,scene_source='privileged_simulator_state')
    def clear_queue(self):pass
    def advance(self,instruction,steps):self.finished=True;return 1
    def held(self):return {OBJECTS[0][0]:True,OBJECTS[1][0]:False}

def fixtures():
    engine=ToyEngine()
    spec=dict(scope='LIBERO_simulation',physical_execution_allowed=False,interface_version='libero-model/3.0',checkpoint_sha256=engine.checkpoint,
        instruction_to_object={'Pick up the '+label+'.':obj for obj,label in OBJECTS})
    backend=SimulationVLAAdapter(spec,engine)
    scene=dict(schema_version='grounded-scene/1.0',scene_id='unit-scene',revision=1,
        observed_at=dict(host_id='simulator-state',at=stamp(0),offset_ms=0,uncertainty_ms=0),
        coordinate_frame=dict(name='image_xy',unit='normalized',width_px=256,height_px=256),
        identity=dict(session_id='unit',scope='session',tracker_ref='toy-only'),
        objects=[dict(instance_id=obj,track_token=obj,label=label,box_xyxy=[.1+i*.4,.2,.3+i*.4,.5],visible=True,
            affordances=['pickable'],attributes=[],detection_confidence=None) for i,(obj,label) in enumerate(OBJECTS)],relations=[])
    records=[dict(record_id='unit-cal-'+str(i),target_index=i,split='calibration',held={o:j==i for j,(o,_) in enumerate(OBJECTS)},simulator_reset_wall_s=.01) for i in range(2)]
    return engine,backend,scene,empirical_outcomes(records,engine.checkpoint)

class SimulationSessionTests(unittest.TestCase):
    def build(self):
        e,b,scene,out=fixtures();s,c=make_runtime('unit',scene,b,out,prior=(.9,0,.1,0));return e,b,s,c

    def test_complete_original_authorization_algebra_to_simulator_endpoint_and_replay(self):
        e,b,s,c=self.build();modes=resolve(c,0,'normal');self.assertEqual(s.state,'AWAITING_AUTHORIZATION')
        dispatch(s,c,e,'unit-command');self.assertEqual(s.state,'SUBMITTED')
        b.advance();c.now+=.1;s.refresh_submission(now=stamp(c.now));self.assertEqual(s.state,'EXECUTING')
        b.finish();c.now+=.1;s.refresh_submission(now=stamp(c.now));self.assertEqual(s.state,'VERIFYING')
        endpoint(s,c,e,'unit');self.assertEqual(s.state,'SUCCEEDED')
        with patch.object(SimulationVLAAdapter,'__init__',side_effect=AssertionError('no model')),patch.object(SimulationVLAAdapter,'submit',side_effect=AssertionError('no execution')):
            self.assertEqual(replay_session_state(s.journal),s.snapshot())

    def test_dispatch_without_actual_confirmation_cannot_call_backend(self):
        e,b,s,c=self.build()
        with self.assertRaises(ProtocolError):s.dispatch('unconfirmed',e.observation(s.context,stamp(0)),now=stamp(0))
        self.assertFalse(b._submissions)

    def test_real_checkpoint_spec_is_bound_and_physical_mode_forbidden(self):
        e,b,s,c=self.build();resolve(c,0,'normal')
        with self.assertRaises(TypeError):b.spec.data['checkpoint_sha256']='x'*64
        spec=thaw(b.spec.data);spec['physical_execution_allowed']=True
        with self.assertRaises(ProtocolError):SimulationVLAAdapter(spec,e)

    def test_delayed_confirmation_cannot_authorize(self):
        e,b,s,c=self.build();resolve(c,0,'normal')
        with self.assertRaises(ProtocolError):s.authorize(now=stamp(c.now+2.01))
        self.assertFalse(b._submissions)

    def test_backend_stop_clears_and_cannot_advance_or_resubmit(self):
        e,b,s,c=self.build();resolve(c,0,'normal');dispatch(s,c,e,'unit-command');b.advance()
        s.request_stop(now=stamp(c.now+.1));s.refresh_submission(now=stamp(c.now+.2))
        self.assertTrue(s.stop_latched);self.assertEqual(s.state,'CANCELLED')
        with self.assertRaises(ProtocolError):b.advance()

    def test_holdout_cannot_fit_outcomes(self):
        e,b,scene,out=fixtures()
        with self.assertRaises(AssertionError):empirical_outcomes([dict(split='test',target_index=0)],e.checkpoint)

    def test_actual_observation_digest_mismatch_blocks_before_consumption(self):
        e,b,s,c=self.build();resolve(c,0,'normal');c.now+=.01;s.authorize(now=stamp(c.now))
        obs=e.observation(s.context,stamp(c.now));obs['sim_state_sha256']='e'*64
        with self.assertRaises(ProtocolError):s.dispatch('bad-frame',obs,now=stamp(c.now))
        self.assertFalse(s.authorization_consumed);self.assertFalse(b._submissions)

    def test_failed_physical_predicate_never_changes_channel_belief(self):
        e,b,s,c=self.build();resolve(c,0,'normal');dispatch(s,c,e,'unit-command');b.advance()
        c.now+=.1;s.refresh_submission(now=stamp(c.now));b.finish();c.now+=.1;s.refresh_submission(now=stamp(c.now))
        before=s.filter.snapshot();e.held=lambda:{obj:False for obj,_ in OBJECTS}
        endpoint(s,c,e,'failed-endpoint')
        self.assertEqual(s.state,'FAILED');self.assertEqual(before,s.filter.snapshot())
        self.assertEqual(replay_session_state(s.journal),s.snapshot())

    def test_context_change_stops_backend_and_invalidates_old_authorization(self):
        from bci_vla.scene.state import GroundedScene
        e,b,s,c=self.build();resolve(c,0,'normal');dispatch(s,c,e,'unit-command');b.advance()
        c.now+=.1;s.refresh_submission(now=stamp(c.now))
        context=dict(s.context,task_revision=2,scene_revision=2);scene=s.scene.to_dict();scene['revision']=2
        s.revise_context(context,GroundedScene(scene),now=stamp(c.now+.01));s.refresh_submission(now=stamp(c.now+.02))
        self.assertTrue(s.stop_latched);self.assertIsNone(s.authorization)
        with self.assertRaises(ProtocolError):b.advance()
        self.assertEqual(replay_session_state(s.journal),s.snapshot())

    def test_multiple_actual_chunks_do_not_repeat_started_transition(self):
        e,b,s,c=self.build();resolve(c,0,'normal');dispatch(s,c,e,'unit-command')
        e.advance=lambda instruction,steps:8
        for _ in range(3):
            b.advance();c.now+=.1;s.refresh_submission(now=stamp(c.now))
        self.assertEqual(s.state,'EXECUTING')
        self.assertEqual(replay_session_state(s.journal),s.snapshot())

    def test_known_probe_repair_restores_flipped_input_without_resetting_goal_truth(self):
        from bci_vla.simulation_v3.recovery import repair_replay_mapping
        e,b,scene,out=fixtures();s,c=make_runtime('unit-flip',scene,b,out,prior=(.81,.09,.09,.01))
        resolve(c,0,'flip')
        def fresh_scene(identifier,revision=1):
            value=deepcopy(scene);value['revision']=revision;value['identity']['session_id']=identifier;return value
        e.scene=fresh_scene
        s2,c2,b2,sessions,modes,proof=repair_replay_mapping(s,c,e,thaw(b.spec.data),out,'repair',emit_goal=0,emit_channel='flip')
        self.assertTrue(proof['inversion']);self.assertEqual(proof['goal_marginals_before'],proof['initial_repaired_goal_marginals'])
        self.assertEqual(s2.state,'AWAITING_AUTHORIZATION');dispatch(s2,c2,e,'new-command')
        self.assertEqual(s2.state,'SUBMITTED')
        for item in [s,*sessions]:self.assertEqual(replay_session_state(item.journal),item.snapshot())

    def test_goal_change_repair_uses_fresh_evidence_and_fresh_confirmation(self):
        from bci_vla.simulation_v3.recovery import repair_replay_mapping
        e,b,scene,out=fixtures();s,c=make_runtime('unit-goal-change',scene,b,out,prior=(.81,.09,.09,.01))
        resolve(c,1,'normal')
        def fresh_scene(identifier,revision=1):
            value=deepcopy(scene);value['revision']=revision;value['identity']['session_id']=identifier;return value
        e.scene=fresh_scene
        s2,c2,b2,sessions,modes,proof=repair_replay_mapping(s,c,e,thaw(b.spec.data),out,'goal-repair',emit_goal=1,emit_channel='normal')
        self.assertFalse(proof['inversion']);self.assertEqual(s2.state,'AWAITING_AUTHORIZATION')
        self.assertEqual(s2.contract.data['proposal']['intent']['source_id'],OBJECTS[1][0])

    def test_no_response_in_full_session_pauses_without_simulator_submission(self):
        from bci_vla.benchmark.offline import response
        e,b,s,c=self.build();q=s.begin_interaction('TASK_QUERY',now=stamp(0),monotonic_now=0)
        result=s.ingest(response(q,'NO_RESPONSE',start=1,duration=1,index=1),now=stamp(2),monotonic_now=2)
        self.assertTrue(result.accepted);self.assertEqual(s.state,'PAUSED');self.assertFalse(b._submissions)
        self.assertEqual(replay_session_state(s.journal),s.snapshot())
