import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from scene_context.planning import candidate_key
from tests.fixtures import case, observation, rename_instances

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from scene_context.model import IntentModel, ModelConfig
    from scene_context.training import train_model


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is optional locally; run this suite on the CPU server with torch")
class ModelInvarianceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)

    def model(self, variant="full"):
        return IntentModel(ModelConfig(variant=variant, hidden_dim=32, num_heads=4,
                                       num_layers=1, dropout=0)).eval()

    def assert_outputs_equal(self, left, right, rename=None):
        torch.testing.assert_close(left["status_logits"], right["status_logits"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(left["slot_logits"], right["slot_logits"], rtol=1e-5, atol=1e-6)
        lvalues = {candidate_key(c): float(v) for c, v in zip(left["candidates"][0], left["candidate_logits"][0])}
        rvalues = {}
        inverse = {v: k for k, v in (rename or {}).items()}
        for candidate, value in zip(right["candidates"][0], right["candidate_logits"][0]):
            original = copy.deepcopy(candidate)
            for key in ("source_id", "destination_id"):
                original[key] = inverse.get(original.get(key), original.get(key))
            rvalues[candidate_key(original)] = float(value)
        self.assertEqual(set(lvalues), set(rvalues))
        for key, value in lvalues.items():
            self.assertAlmostEqual(value, rvalues[key], places=5)

    def test_object_and_edge_order_do_not_change_semantics(self):
        model = self.model()
        obs = observation(True)
        obs["context"]["history"] = [{"action": "place", "source_id": "opaque_grape_739",
                                       "destination_id": "opaque_bowl_284"}]
        permuted = copy.deepcopy(obs)
        permuted["objects"] = list(reversed(permuted["objects"]))
        permuted["relations"] = list(reversed(permuted["relations"]))
        with torch.inference_mode():
            self.assert_outputs_equal(model(obs), model(permuted))

    def test_arbitrary_instance_id_spelling_is_not_a_learned_feature(self):
        model = self.model()
        obs = observation(True)
        obs["context"]["history"] = [{"action": "place", "source_id": "opaque_grape_739",
                                       "destination_id": "opaque_bowl_284"}]
        renamed, mapping = rename_instances(obs)
        with torch.inference_mode():
            self.assert_outputs_equal(model(obs), model(renamed), mapping)

    def test_no_relations_really_removes_edge_evidence(self):
        model = self.model("no_relations")
        obs = observation(True)
        changed = copy.deepcopy(obs)
        changed["relations"][0]["predicate"] = "inside"
        changed["relations"][0]["confidence"] = 0.1
        with torch.inference_mode():
            self.assert_outputs_equal(model(obs), model(changed))

    def test_no_bci_really_removes_initial_selection_evidence(self):
        model = self.model("no_bci")
        obs = observation(True)
        changed = copy.deepcopy(obs)
        changed["selection"] = {"object_id": "opaque_bowl_284", "probabilities": {"opaque_bowl_284": 1.0}}
        with torch.inference_mode():
            self.assert_outputs_equal(model(obs), model(changed))

    def test_no_history_really_removes_past_event_evidence(self):
        model = self.model("no_history")
        obs = observation(True)
        changed = copy.deepcopy(obs)
        changed["context"]["history"] = [{"action": "pick", "source_id": "opaque_grape_739", "destination_id": None}]
        with torch.inference_mode():
            self.assert_outputs_equal(model(obs), model(changed))

    def test_full_relation_network_is_sensitive_to_edge_evidence(self):
        model = self.model("full")
        obs = observation(True)
        changed = copy.deepcopy(obs)
        changed["relations"][0]["predicate"] = "inside"
        with torch.inference_mode():
            first, second = model(obs), model(changed)
        delta = torch.max(torch.abs(first["candidate_logits"][0] - second["candidate_logits"][0])).item()
        self.assertGreater(delta, 1e-7)

    def test_empty_scene_and_variable_object_batch_are_finite(self):
        model = self.model()
        empty = {"objects": [], "relations": [], "selection": {}, "context": {}}
        with torch.inference_mode():
            output = model([empty, observation(True)])
        self.assertEqual(tuple(output["status_logits"].shape), (2, 3))
        self.assertTrue(torch.isfinite(output["status_logits"]).all().item())
        self.assertEqual(output["candidate_logits"][0].numel(), 0)
        self.assertGreater(output["candidate_logits"][1].numel(), 0)

    def test_training_rejects_shared_scene_before_optimization(self):
        with tempfile.TemporaryDirectory() as tmp:
            train, val = Path(tmp) / "train.jsonl", Path(tmp) / "val.jsonl"
            train.write_text(json.dumps(case("train", "same")) + "\n", encoding="utf-8")
            val.write_text(json.dumps(case("val", "same")) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "leakage"):
                train_model(train, val, Path(tmp) / "run", epochs=1)

    def test_training_api_rejects_test_data_option(self):
        with self.assertRaises(TypeError):
            train_model("unused", "unused", "unused", test_path="sealed_test.jsonl")


if __name__ == "__main__":
    unittest.main()
