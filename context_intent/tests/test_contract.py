import copy
import json
from pathlib import Path
import tempfile
import unittest

from scene_context.baselines import RulePredictor
from scene_context.planning import candidate_key, finalize_prediction, generate_candidates, structural_clarification
from scene_context.rendering import render_instruction
from scene_context.schema import assert_disjoint_scenes, load_cases, public_observation, validate_case, validate_observation
from tests.fixtures import case, intent, observation, ready_proposal


class ObservationContractTests(unittest.TestCase):
    def test_duplicate_instance_ids_are_invalid(self):
        obs = observation()
        obs["objects"].append(copy.deepcopy(obs["objects"][0]))
        with self.assertRaises(ValueError):
            validate_observation(obs)

    def test_nonfinite_positions_and_confidence_are_invalid(self):
        for field, value in (("position", [float("nan"), 0.2]),
                             ("position", [1.1, 0.5]), ("detection_confidence", float("inf"))):
            with self.subTest(field=field, value=value):
                obs = observation()
                obs["objects"][0][field] = value
                with self.assertRaises(ValueError):
                    validate_observation(obs)

    def test_unknown_selection_relation_and_probability_ids_are_invalid(self):
        variants = []
        first = observation(); first["selection"]["object_id"] = "missing"; variants.append(first)
        second = observation(); second["relations"][0]["object"] = "missing"; variants.append(second)
        third = observation(); third["selection"]["probabilities"]["missing"] = 0.8; variants.append(third)
        for obs in variants:
            with self.subTest(obs=obs), self.assertRaises(ValueError):
                validate_observation(obs)

    def test_hidden_labels_cannot_be_smuggled_into_public_fields(self):
        for top, key in ((None, "annotation"), ("context", "true_intent"), ("selection", "ground_truth")):
            obs = observation()
            (obs if top is None else obs[top])[key] = "hidden_answer"
            with self.subTest(field=key), self.assertRaises(ValueError):
                validate_observation(obs)

    def test_posterior_normalization_does_not_mutate_input(self):
        obs = observation()
        obs["selection"]["probabilities"] = {"opaque_grape_739": 8, "opaque_bowl_284": 2}
        # Individual scores outside [0,1] must not be accepted as probabilities.
        with self.assertRaises(ValueError):
            validate_observation(obs)
        obs["selection"]["probabilities"] = {"opaque_grape_739": 0.8, "opaque_bowl_284": 0.1}
        original = copy.deepcopy(obs)
        normalized = validate_observation(obs)
        self.assertAlmostEqual(sum(normalized["selection"]["probabilities"].values()), 1)
        self.assertEqual(obs, original)

    def test_public_observation_omits_case_metadata_and_is_independent(self):
        item = case()
        item["hidden_request"] = "SECRET_REQUEST"
        public = public_observation(item)
        self.assertEqual(set(public), {"objects", "relations", "selection", "context"})
        public["objects"][0]["label"] = "changed"
        self.assertEqual(item["observation"]["objects"][0]["label"], "grapes")

    def test_clarify_label_must_not_contain_hidden_intent(self):
        item = case(ambiguous=True)
        item["annotation"]["intent"] = intent()
        with self.assertRaises(ValueError):
            validate_case(item)

    def test_scene_variants_cannot_cross_splits(self):
        train = [case("train_variant", "same_scene")]
        test = [case("test_variant", "same_scene", ambiguous=True)]
        with self.assertRaisesRegex(ValueError, "leakage"):
            assert_disjoint_scenes(train, [], test)
        assert_disjoint_scenes(train, [], [case("new", "independent_scene")])

    def test_duplicate_case_ids_in_jsonl_are_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text(json.dumps(case()) + "\n" + json.dumps(case()) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cases(path)


class IntentGuardTests(unittest.TestCase):
    def test_unique_task_is_proposal_requiring_confirmation(self):
        result = RulePredictor().predict(observation())
        self.assertEqual(result["status"], "ready")
        self.assertTrue(result["requires_confirmation"])
        self.assertEqual(result["intent"], intent())

    def test_multi_destination_ready_proposal_is_not_released(self):
        result = finalize_prediction(observation(True), ready_proposal(), confidence=1.0)
        self.assertEqual(result["raw_proposal"]["status"], "ready")
        self.assertEqual(result["status"], "clarify")
        self.assertEqual(result["clarification"]["slot"], "destination")
        self.assertIsNone(result["instruction"])
        self.assertFalse(result["requires_confirmation"])

    def test_unknown_action_requires_clarification_not_object_prior(self):
        obs = observation()
        obs["context"]["goal"] = "unspecified"
        result = RulePredictor().predict(obs)
        self.assertEqual(result["status"], "clarify")
        self.assertEqual(result["clarification"]["slot"], "action")

    def test_low_bci_confidence_requires_source_clarification(self):
        obs = observation()
        obs["selection"]["probabilities"] = {"opaque_grape_739": 0.55, "opaque_bowl_284": 0.45}
        result = finalize_prediction(obs, ready_proposal(), confidence=1.0)
        self.assertEqual(result["status"], "clarify")
        self.assertEqual(result["clarification"]["slot"], "source")

    def test_foreign_instance_cannot_be_ready(self):
        result = finalize_prediction(observation(), ready_proposal("hallucinated_bowl"))
        self.assertEqual(result["status"], "reject")
        self.assertIsNone(result["instruction"])

    def test_failed_predictor_is_not_made_successful_by_guard(self):
        result = finalize_prediction(observation(True), {"status": "reject", "intent": None})
        self.assertEqual(result["status"], "reject")

    def test_candidate_enumerator_does_not_leak_bci_or_goal(self):
        first = observation(True)
        first["objects"].append({"id": "cloth", "label": "cloth", "position": [0.4, 0.2],
                                  "affordances": ["pickable", "foldable"]})
        second = copy.deepcopy(first)
        second["selection"] = {"object_id": "cloth", "probabilities": {"cloth": 1.0}}
        second["context"]["goal"] = "fold"
        self.assertEqual({candidate_key(x) for x in generate_candidates(first)},
                         {candidate_key(x) for x in generate_candidates(second)})

    def test_explicit_destination_resolves_ambiguity(self):
        obs = observation(True)
        obs["context"]["confirmed_destination_id"] = "opaque_bowl_284"
        self.assertIsNone(structural_clarification(obs))
        self.assertEqual(RulePredictor().predict(obs)["status"], "ready")


class RenderingTests(unittest.TestCase):
    def test_instruction_contains_human_grounding_not_opaque_ids(self):
        text = render_instruction(observation(True), intent())
        self.assertIn("leftmost bowl", text)
        for obj in observation(True)["objects"]:
            self.assertNotIn(obj["id"], text)

    def test_indistinguishable_instances_cannot_be_rendered(self):
        obs = observation(True)
        obs["objects"][2]["position"] = [0.71, 0.5]
        obs["context"]["confirmed_destination_id"] = "opaque_bowl_284"
        with self.assertRaises(ValueError):
            render_instruction(obs, intent())
        self.assertEqual(finalize_prediction(obs, ready_proposal())["status"], "clarify")

    def test_unique_attribute_can_ground_instance(self):
        obs = observation(True)
        obs["objects"][1]["attributes"] = ["red"]
        obs["objects"][2]["attributes"] = ["blue"]
        self.assertIn("red bowl", render_instruction(obs, intent()))

    def test_attribute_subset_does_not_uniquely_identify_instance(self):
        obs = observation(True)
        obs["objects"][1]["attributes"] = ["red"]
        obs["objects"][2]["attributes"] = ["red", "small"]
        obs["objects"][2]["position"] = obs["objects"][1]["position"]
        with self.assertRaises(ValueError):
            render_instruction(obs, intent())


if __name__ == "__main__":
    unittest.main()
