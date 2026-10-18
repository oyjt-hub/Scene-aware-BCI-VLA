"""B2 information-boundary, binding and token-budget tests on LIBERO fixtures."""
import copy
import json
from pathlib import Path
import unittest

import torch

from bci_vla.neural_interaction_v4.config import ModuleConfig
from bci_vla.neural_interaction_v4.model import clean_batch, FLOAT_KEYS
from bci_vla.neural_interaction_v4.tensorizer import collate_public
from bci_vla.training_v4.text_baseline import compact_model_view, build_b2_prompt, baseline_evidence_text


ROOT = Path(__file__).resolve().parents[1]


class TextBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = json.loads((ROOT / "execution/v4/06_fixtures/public.json").read_text(encoding="utf-8"))
        cls.public = next(p for p in cls.rows if p["public_state"]["diagnostics"]["status"] == "SUFFICIENT")

    def prompt(self, public):
        return "answer en Objects: " + "; ".join(o["alias"] for o in public["scene"]["objects"]) + ". Use public interaction evidence.\n"

    def test_lossless_roundtrip_every_model_tensor(self):
        # Distinct history/candidate configurations, not just a single fixture.
        for public in self.rows[::8]:
            expected, _ = collate_public([public])
            expected = clean_batch(expected, ModuleConfig())
            # Test the serialized decimal numbers, not only the pre-JSON view.
            view = json.loads(baseline_evidence_text(public).split(": ", 1)[1])
            self.assertEqual(set(view["tensors"]), set(expected))
            for key, tensor in expected.items():
                encoded = view["tensors"][key]
                restored = torch.zeros(encoded["shape"], dtype=tensor.dtype)
                flat = restored.reshape(-1)
                if key in FLOAT_KEYS:
                    for i, value in encoded["values"]:
                        flat[i] = value
                else:
                    flat[encoded["values"]] = True
                self.assertTrue(torch.equal(restored, tensor[0]), key)

    def test_no_private_or_provenance_identifiers_in_model_text(self):
        public = self.public
        result = build_b2_prompt(public, self.prompt(public))
        for marker in [public["sample_id"], public["context"]["session_id"], public["context"]["scene_id"], public["scene"]["image_ref"]]:
            self.assertNotIn(marker, result["prompt"])
        for obj in public["scene"]["objects"]:
            self.assertNotIn(obj["instance_id"], result["prompt"])
        self.assertNotIn("Selected object:", result["prompt"])
        self.assertEqual(result["prefix_tokens"], 0)
        self.assertFalse(result["truncated"])

    def test_out_of_window_probe_object_semantics_not_added(self):
        # Raw diagnostic references are deliberately deidentified by tensorizer.
        changed = copy.deepcopy(self.public)
        ids = [o["instance_id"] for o in changed["scene"]["objects"]]
        for collection in [changed["history"], changed["public_state"]["diagnostics"]["receipts"]]:
            for event in collection:
                if not event.get("question") or event["question"]["role"] != "PROBE":
                    continue
                for opt in event["question"]["options"]:
                    if opt["reference"] in ids:
                        opt["reference"] = ids[(ids.index(opt["reference"]) + 1) % len(ids)]
        self.assertEqual(compact_model_view(self.public), compact_model_view(changed))

    def test_out_of_window_receipt_detail_cannot_add_model_information(self):
        public = json.loads((ROOT / "examples/v4/confirmed_task_survives_nine_probes.json").read_text(encoding="utf-8"))
        changed = copy.deepcopy(public)
        ids_in_window = {e["event_id"] for e in changed["history"]}
        old = next(e for e in changed["public_state"]["diagnostics"]["receipts"] if e["event_id"] not in ids_in_window)
        # Score detail outside H is schema-valid but absent from every tensor;
        # aggregate attempts/correct/age remain unchanged.
        old["evidence"]["values"][0]["value"] += .1234567
        config = ModuleConfig(roi_dim=4, semantic_dim=3)
        self.assertEqual(compact_model_view(public, config=config), compact_model_view(changed, config=config))

    def test_event_and_sample_ids_do_not_change_text(self):
        changed = copy.deepcopy(self.public)
        changed["sample_id"] = "anonymous-alternate-sample"
        # sample_id is a binding key, not a feature (schema-valid length).
        a = build_b2_prompt(self.public, self.prompt(self.public))
        b = build_b2_prompt(changed, self.prompt(changed))
        self.assertEqual(a["prompt"], b["prompt"])
        self.assertEqual(a["view_sha256"], b["view_sha256"])
        self.assertNotEqual(a["public_input_sha256"], b["public_input_sha256"])

    def test_observed_score_change_is_not_dropped(self):
        public = next(p for p in self.rows if any(e.get("evidence") and e["evidence"]["values"] for e in p["history"]))
        changed = copy.deepcopy(public)
        event = next(e for e in changed["history"] if e.get("evidence") and e["evidence"]["values"])
        event["evidence"]["values"][0]["value"] += .01234567
        self.assertNotEqual(compact_model_view(public), compact_model_view(changed))

    def test_public_alias_row_binding_tracks_object_reorder(self):
        public = copy.deepcopy(self.public)
        public["scene"]["objects"].reverse()
        rows = compact_model_view(public)["object_rows"]
        self.assertEqual(rows, [[i, o["alias"]] for i, o in enumerate(public["scene"]["objects"]) if o["object_valid"]])
        with self.assertRaisesRegex(ValueError, "lacks public object alias"):
            build_b2_prompt(public, "answer en Objects: none.")

    def test_float32_signed_zero_is_not_silently_rounded(self):
        public = copy.deepcopy(self.public)
        public["scene"]["objects"][0]["roi_feature"][0] = -0.0
        serialized = json.loads(baseline_evidence_text(public).split(": ", 1)[1])
        values = dict(serialized["tensors"]["objects"]["values"])
        self.assertIn(0, values)
        self.assertTrue(torch.signbit(torch.tensor(values[0], dtype=torch.float32)))

    def test_complete_prompt_is_measured_and_never_truncated(self):
        seen = []
        def tokenizer(text):
            seen.append(text)
            return list(range(37))
        result = build_b2_prompt(self.public, self.prompt(self.public), tokenizer=tokenizer, max_tokens=37)
        self.assertEqual(seen, [result["prompt"]])
        self.assertEqual(result["text_tokens"], 37)
        with self.assertRaisesRegex(ValueError, "needs 37.*no truncation"):
            build_b2_prompt(self.public, self.prompt(self.public), tokenizer=tokenizer, max_tokens=36)
        with self.assertRaisesRegex(ValueError, "real tokenizer required"):
            build_b2_prompt(self.public, self.prompt(self.public), max_tokens=1600)
        self.assertEqual(build_b2_prompt(self.public, self.prompt(self.public))["text_tokens"], None)

    def test_label_injection_and_overlong_history_fail_closed(self):
        public = copy.deepcopy(self.public)
        public["private_labels"] = {"goal": "oracle"}
        with self.assertRaises(Exception):
            compact_model_view(public)
        public = copy.deepcopy(self.public)
        public["history"] *= 9
        with self.assertRaises(Exception):
            compact_model_view(public)


if __name__ == "__main__":
    unittest.main()
