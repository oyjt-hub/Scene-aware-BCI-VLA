import copy
import unittest

from scene_context.baselines import RulePredictor
from scene_context.integration import ContextIntentAdapter, observation_from_detections


def observation():
    return observation_from_detections(["grapes","bowl"],[[10,10,30,30],[70,70,90,90]],[100,100],
        capabilities={"grapes":["pickable"],"bowl":["container"]},selected_label="grapes",context={"goal":"put_away"})


class IntegrationTests(unittest.TestCase):
    def test_proposal_cannot_be_consumed_twice(self):
        obs=observation()
        adapter=ContextIntentAdapter(RulePredictor())
        decision=adapter.propose(obs)
        self.assertEqual(decision["status"],"ready")
        self.assertEqual(adapter.confirm(decision["proposal_id"],decision["intent"],obs),"Put the grapes into the bowl.")
        with self.assertRaises(ValueError):
            adapter.confirm(decision["proposal_id"],decision["intent"],obs)

    def test_stale_scene_needs_fresh_confirmation(self):
        obs=observation()
        adapter=ContextIntentAdapter(RulePredictor())
        decision=adapter.propose(obs)
        changed=copy.deepcopy(obs)
        changed["objects"][1]["position"]=[.1,.9]
        with self.assertRaises(ValueError):
            adapter.confirm(decision["proposal_id"],decision["intent"],changed)

    def test_cancel_and_new_proposal_invalidate_old_ids(self):
        obs=observation()
        adapter=ContextIntentAdapter(RulePredictor())
        old=adapter.propose(obs)
        adapter.propose(obs)
        with self.assertRaises(ValueError):
            adapter.confirm(old["proposal_id"],old["intent"],obs)
        current=adapter.propose(obs)
        adapter.cancel()
        with self.assertRaises(ValueError):
            adapter.confirm(current["proposal_id"],current["intent"],obs)

    def test_duplicate_labels_do_not_select_first_instance(self):
        obs=observation_from_detections(["grapes","grapes"],[[0,0,10,10],[80,80,90,90]],[100,100],
            capabilities={"grapes":["pickable"]},selected_label="grapes",context={"goal":"pick"})
        self.assertIsNone(obs["selection"]["object_id"])
        decision=RulePredictor().predict(obs)
        self.assertEqual(decision["status"],"clarify")
        self.assertEqual(decision["clarification"]["slot"],"source")

    def test_detector_geometry_is_not_invented(self):
        with self.assertRaises(ValueError):
            observation_from_detections(["grapes"],[[10,10,200,200]],[100,100],capabilities={})

    def test_unknown_capability_is_not_assumed(self):
        obs=observation_from_detections(["unknown"],[[0,0,10,10]],[100,100],capabilities={},selected_label="unknown")
        self.assertEqual(RulePredictor().predict(obs)["status"],"reject")
