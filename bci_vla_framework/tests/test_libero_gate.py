import unittest
from bci_vla.pilot.libero_gate import LiberoSimulationGate
from bci_vla.protocols import ProtocolError

class LiberoGateTests(unittest.TestCase):
    def ready(self):
        g=LiberoSimulationGate();c=g.bind(suite='libero_goal',task_id=0,language='fixture task',scene_revision=1)
        g.set_mode('CONFIRM');g.confirm(c,accepted=True,input_source='scripted_simulation_fixture');return g,c

    def test_without_confirmation_no_chunk(self):
        g=LiberoSimulationGate();c=g.bind(suite='libero_goal',task_id=0,language='a',scene_revision=1)
        with self.assertRaises(ProtocolError):g.enqueue([[0.]*7],c)

    def test_valid_7d_chunk_only_applies_in_bound_context(self):
        g,c=self.ready();g.enqueue([[.1]*7],c);self.assertEqual(g.pop(c),[.1]*7)
        self.assertEqual(g.snapshot()['applied_actions'],1);self.assertFalse(g.snapshot()['physical_execution_allowed'])

    def test_probe_flushes_chunk_and_requires_new_confirmation(self):
        g,c=self.ready();g.enqueue([[0.]*7]*3,c);g.set_mode('CHANNEL_PROBE')
        self.assertEqual(g.snapshot()['discarded_actions'],3)
        with self.assertRaises(ProtocolError):g.pop(c)
        g.set_mode('CONFIRM')
        with self.assertRaises(ProtocolError):g.enqueue([[0.]*7],c)

    def test_task_or_scene_change_invalidates_old_actions(self):
        for updates in [dict(task_id=1),dict(scene_revision=2),dict(language='changed')]:
            g,c=self.ready();g.enqueue([[0.]*7],c)
            kw=dict(suite='libero_goal',task_id=0,language='fixture task',scene_revision=1);kw.update(updates)
            new=g.bind(**kw);self.assertIsNone(g.confirmed);self.assertEqual(len(g.queue),0)
            with self.assertRaises(ProtocolError):g.confirm(c,accepted=True,input_source='scripted_simulation_fixture')
            with self.assertRaises(ProtocolError):g.enqueue([[0.]*7],new)

    def test_stop_latches_and_does_not_claim_physical_stop(self):
        g,c=self.ready();g.enqueue([[0.]*7]*2,c);g.stop();g.stop()
        self.assertTrue(g.snapshot()['stop_latched']);self.assertFalse(g.snapshot()['physical_stop_verified'])
        with self.assertRaises(ProtocolError):g.confirm(c,accepted=True,input_source='scripted_simulation_fixture')

    def test_bad_dimensions_nonfinite_and_out_of_bounds_are_atomic(self):
        for action in [[0.]*14,[float('nan')]*7,[1.01]*7,[True]*7]:
            g,c=self.ready()
            with self.assertRaises(ProtocolError):g.enqueue([[0.]*7,action],c)
            self.assertEqual(len(g.queue),0)

    def test_real_BCI_claim_rejected_and_reject_does_not_authorize(self):
        for source,accepted in [('live_bci',True),('scripted_simulation_fixture',False)]:
            g,c=self.ready();g.invalidate()
            with self.assertRaises(ProtocolError):g.confirm(c,accepted=accepted,input_source=source)
            self.assertIsNone(g.confirmed)

    def test_rejection_revokes_previous_simulated_confirmation(self):
        g,c=self.ready();g.enqueue([[0.]*7],c)
        with self.assertRaises(ProtocolError):g.confirm(c,accepted=False,input_source='scripted_simulation_fixture')
        self.assertFalse(g.queue);self.assertIsNone(g.confirmed)


class LiberoConversionTests(unittest.TestCase):
    def stats(self):return {'min':[-.9]*6+[0.],'max':[.9]*6+[1.],'mask':[True]*6+[False]}

    def test_gripper_sign_matches_checked_repository_evaluator(self):
        from bci_vla.pilot.libero_actions import decode_starvla_chunk
        import numpy as np
        raw=np.zeros((1,4,7));raw[0,:,6]=[0,.49,.5,1]
        result=decode_starvla_chunk(raw,self.stats())
        self.assertEqual(result[:,6].tolist(),[1,1,-1,-1])

    def test_translation_rotation_min_max_mapping_and_clipping(self):
        from bci_vla.pilot.libero_actions import decode_starvla_chunk
        result=decode_starvla_chunk([[[-2,0,2,-1,1,0,0]]],self.stats())
        self.assertEqual(result[0,:6].tolist(),[-.9,0,.9,-.9,.9,0])

    def test_invalid_shape_nonfinite_and_mask_rejected(self):
        from bci_vla.pilot.libero_actions import decode_starvla_chunk
        import numpy as np
        for raw in [np.zeros((1,8,14)),np.full((1,8,7),np.nan)]:
            with self.assertRaises(ProtocolError):decode_starvla_chunk(raw,self.stats())
        stats=self.stats();stats['mask'][6]=True
        with self.assertRaises(ProtocolError):decode_starvla_chunk(np.zeros((1,8,7)),stats)
