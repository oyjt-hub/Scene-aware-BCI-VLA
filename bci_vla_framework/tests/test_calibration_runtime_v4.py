"""Public-only row assembly with mock logits; no learned-model calibration run."""
import copy
import json
import math
from pathlib import Path
import unittest

from bci_vla.calibration_v4 import build_calibration_rows, fit_temperature, evaluate_calibration
from execution.v4.build_step04_data import generate

ROOT = Path(__file__).resolve().parents[1]


class RuntimeRowsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Metadata only. No image, simulator state, test record or model opened.
        manifest = json.loads((ROOT / "data/v4/captures/capture_manifest.json").read_text(encoding="utf-8"))
        cls.capture = next(c for c in manifest["episodes"] if c["partition"] == "calibration_fit")

    def setUp(self):
        self.binding = {"model_sha256": "a" * 64, "data_sha256": "b" * 64,
                        "input_schema_sha256": "c" * 64, "policy_sha256": "d" * 64}
        self.calls = []

    def predictor(self, public):
        serialized = json.dumps(public)
        for forbidden in ("hidden_truth", "private_future", "later_goal", "initial_goal", "channel_injected_mode"):
            self.assertNotIn(forbidden, serialized)
        self.assertNotIn(public["history"][-1]["role"], ("STOP", "CANCEL"))
        self.calls.append(copy.deepcopy(public))
        ids = [o["instance_id"] for o in public["scene"]["objects"] if o["object_valid"]]
        support = public["public_state"]["diagnostics"]["status"] == "SUFFICIENT"
        return {"candidate_ids": ids, "intent_logits": [0.] * (len(ids) + 1),
                "channel_logit": .2 if support else None, "channel_supported": support}

    def build(self, variants=("ambiguous",), captures=None, predictor=None, **kwargs):
        return build_calibration_rows([self.capture] if captures is None else captures,
                                     self.predictor if predictor is None else predictor,
                                     split="calibration_fit", binding=self.binding, variants=variants, **kwargs)

    def test_latent_twins_share_prediction_and_probability_target(self):
        result = self.build()
        self.assertEqual(len(self.calls), 8)
        self.assertEqual(result["counts"]["latent_twin_pairs"], 8)
        self.assertEqual(result["counts"]["private_outcomes"], 16)
        self.assertEqual(len(result["intent_rows"]), 8)
        for row in result["intent_rows"]:
            self.assertEqual(row["target_mode"], "PROBABILITY")
            self.assertEqual(sorted(v for v in row["target"] if v), [.5, .5])
            self.assertEqual(row["metadata"]["latent_outcomes"], 2)
        self.assertFalse(result["identified_POINT_supervision"])
        self.assertFalse(result["policy_closed_loop"])

    def test_same_model_visible_history_across_variants_is_not_repredicted(self):
        result = self.build(variants=("stable", "ambiguous"))
        self.assertEqual(result["counts"]["generated_public_records"], 16)
        self.assertEqual(result["counts"]["predictor_calls"], 8)
        self.assertEqual(result["counts"]["deduplicated_public_records"], 8)
        self.assertTrue(all(sorted(v for v in row["target"] if v) == [.25, .75] for row in result["intent_rows"]))

    def test_filtering_precedes_generator_and_callback_test_frames_unread(self):
        excluded = [{"partition": "test_sealed"}, {"partition": "train"}, {"partition": "validation"},
                    {"partition": "calibration_audit"}]
        seen = []
        def generator(capture, variant):
            self.assertEqual(capture["partition"], "calibration_fit")
            seen.append(capture["base_episode_id"])
            return generate(capture, variant)
        result = self.build(captures={"episodes": excluded + [self.capture]}, generator=generator)
        self.assertEqual(seen, [self.capture["base_episode_id"]])
        self.assertEqual(result["counts"]["base_episodes"], 1)
        self.assertFalse(result["formal_test_accessed"])
        for split in ("train", "test", "test_sealed"):
            with self.assertRaisesRegex(ValueError, "test stays sealed"):
                build_calibration_rows(excluded, self.predictor, split=split, binding=self.binding)

    def test_controls_bypass_prediction_and_future_support_is_independent(self):
        result = self.build(variants=("stop", "cancel"))
        self.assertEqual(result["counts"]["control_records_bypassed"], 2)
        self.assertEqual(result["counts"]["generated_public_records"], 16)
        self.assertTrue(any(r["supported"] for r in result["channel_rows"]))
        for row in result["channel_rows"]:
            if not row["supported"]:
                self.assertFalse(row["mask"])
                self.assertIsNone(row["logits"])
                self.assertIsNone(row["target"])
            else:
                self.assertTrue(0 <= row["target"] <= 1)

    def test_hidden_goal_change_cannot_change_public_model_inputs_for_twin(self):
        before = copy.deepcopy(self.capture)
        def mutating_callback(public):
            prediction = self.predictor(public)
            public["history"].clear()
            public["scene"]["objects"].clear()
            return prediction
        result = self.build(predictor=mutating_callback)
        self.assertEqual(self.capture, before)
        self.assertEqual(len(result["intent_rows"]), 8)

    def test_unsupported_channel_cannot_be_claimed_or_future_invented(self):
        def oracle_support(public):
            prediction = self.predictor(public)
            prediction.update(channel_supported=True, channel_logit=1.)
            return prediction
        with self.assertRaisesRegex(ValueError, "without sufficient"):
            self.build(predictor=oracle_support)
        def missing_future(capture, variant):
            records, world = generate(capture, variant)
            return [(public, private, None) for public, private, future in records], world
        with self.assertRaisesRegex(ValueError, "missing its private future"):
            self.build(generator=missing_future)

    def test_predictor_axis_and_duplicate_capture_rejected(self):
        def bad_axis(public):
            prediction = self.predictor(public)
            prediction["candidate_ids"].reverse()
            return prediction
        with self.assertRaisesRegex(ValueError, "candidate axis"):
            self.build(predictor=bad_axis)
        with self.assertRaisesRegex(ValueError, "duplicate base capture"):
            self.build(captures=[self.capture, self.capture])

    def test_future_probe_old_mapping_or_wrong_horizon_is_rejected(self):
        for corruption in ("mapping", "time"):
            def wrong_future(capture, variant):
                records, world = generate(capture, variant)
                for public, private, future in records:
                    if future:
                        if corruption == "mapping":
                            future["question"]["context"]["mapping_revision"] += 1
                        else:
                            future["question"]["issued_at_s"] += 100
                return records, world
            with self.assertRaisesRegex(ValueError, "mapping/layout/time"):
                self.build(generator=wrong_future)


class SoftOutcomeCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.binding = {"model_sha256": "a" * 64, "data_sha256": "b" * 64,
                        "input_schema_sha256": "c" * 64, "policy_sha256": "d" * 64}

    def rows(self, split="calibration_fit", task="intent"):
        return [{"split": split, "group_id": split + "-scene", "episode_id": "ep" + str(i),
                 "logits": [8., -8.] if task == "intent" else 16.,
                 "target_mode": "PROBABILITY", "target": [.5, .5] if task == "intent" else .5,
                 "mask": True, "supported": True, "metadata": {"binding": self.binding}} for i in range(2)]

    def test_soft_outcomes_fit_without_forcing_a_hidden_class(self):
        for task in ("intent", "channel"):
            artifact = fit_temperature(self.rows(task=task), task=task, binding=self.binding)
            self.assertEqual(artifact["target_modes"], ["PROBABILITY"])
            self.assertTrue(artifact["outcome_calibration_not_identifiability_supervision"])
            self.assertGreater(artifact["temperature"], 1)
            audit = evaluate_calibration(self.rows("calibration_audit", task), artifact, task=task, binding=self.binding)
            self.assertLess(audit["after"]["nll"], audit["before"]["nll"])
            self.assertLess(audit["after"]["ece"], audit["before"]["ece"])
            # Outcome Brier retains uncertainty instead of claiming zero error
            # when the target distribution is intrinsically ambiguous.
            expected_uncertainty = .5 if task == "intent" else .25
            self.assertAlmostEqual(audit["after"]["brier"] - audit["after"]["distribution_brier"], expected_uncertainty)

    def test_invalid_soft_probability_cannot_enter_fit(self):
        for target in ([.8, .8], [1., -1.], [float("nan"), .5], [.5]):
            rows = self.rows(); rows[0]["target"] = target
            with self.assertRaisesRegex(ValueError, "probability target"):
                fit_temperature(rows, task="intent", binding=self.binding)


if __name__ == "__main__":
    unittest.main()
