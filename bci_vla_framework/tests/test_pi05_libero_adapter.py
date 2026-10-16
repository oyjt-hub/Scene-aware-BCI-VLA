import unittest
import numpy as np
from bci_vla.adapters.vla.libero_pi05 import controller_actions
from bci_vla.protocols import ProtocolError

class Pi05LiberoAdapterTests(unittest.TestCase):
    def test_controller_bounds_clip_and_count_without_gripper_flip(self):
        raw=np.zeros((1,10,7));raw[0,0]=[-2,-1,0,1,2,.5,-.3]
        actions,report=controller_actions(raw,lower=[-1]*7,upper=[1]*7)
        self.assertEqual(actions.shape,(8,7));self.assertEqual(actions[0].tolist(),[-1,-1,0,1,1,.5,-.3])
        self.assertEqual(report['clipped_values'],2);self.assertEqual(raw[0,0,0],-2)

    def test_unknown_action_limits_cannot_be_guessed(self):
        with self.assertRaises(ProtocolError):controller_actions(np.zeros((1,8,7)),lower=[-2]*7,upper=[2]*7)

    def test_nonfinite_bad_dimensions_short_horizon_rejected(self):
        for raw in [np.zeros((1,8,14)),np.zeros((1,4,7)),np.full((1,8,7),np.nan)]:
            with self.assertRaises(ProtocolError):controller_actions(raw,lower=[-1]*7,upper=[1]*7)
