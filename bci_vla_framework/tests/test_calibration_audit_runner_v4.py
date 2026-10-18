"""Synthetic CPU runner fixtures; no real calibration data/model/dynamics run."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from bci_vla.evaluation_v4.runner import run_episode, run_calibration_audit_episode, make_question
from execution.v4.step09_calibrate_and_policy import (aggregate_base_summaries, online_audit_acceptance,
                                                      online_audit_rows, candidates, SCENARIOS)

ROOT = Path(__file__).resolve().parents[1]


class WaitPolicy:
    name = "unit-fixed-policy"
    def __init__(self): self.inputs = []
    def decide(self, public):
        self.inputs.append(copy.deepcopy(public))
        assert "hidden_truth" not in public and "world" not in public
        return {"action": "WAIT"}


class AuditRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        objects = [{"instance_id": key, "alias": "OBJ_" + str(i), "label": key, "object_valid": True,
                    "roi_feature": [0.] * 6, "roi_valid": True, "semantic_feature": None, "semantic_valid": False,
                    "bbox_xyxy": [.1 + .4*i, .1, .3 + .4*i, .3]} for i, key in enumerate(("object-a", "object-b"))]
        frames = []
        for i in range(2):
            payload = b"synthetic runner hash fixture " + bytes([i])
            path = self.root / f"fixture{i}.bin"; path.write_bytes(payload)
            frames.append({"image_path": path.name, "image_sha256": hashlib.sha256(payload).hexdigest(), "objects": objects})
        self.capture = {"partition": "calibration_audit", "base_episode_id": "unit-base", "group_id": "unit-scene", "frames": frames}
        self.world = {"episode_id": "unit-base", "initial_goal": "object-a", "later_goal": "object-a",
                      "initial_success": .9, "later_success": .9, "change_at_s": 8.}

    def tearDown(self): self.temp.cleanup()

    def test_development_runner_keeps_calibration_and_test_closed(self):
        for split in ("calibration_fit", "calibration_audit", "test", "test_sealed"):
            capture = {**self.capture, "partition": split}
            policy = WaitPolicy()
            with self.assertRaisesRegex(ValueError, "not calibration or test"):
                run_episode(capture, self.world, policy, self.root, horizon=4)
            self.assertEqual(policy.inputs, [])

    def test_audit_entry_preserves_true_split_and_public_only_policy(self):
        policy, original = WaitPolicy(), copy.deepcopy(self.capture)
        result = run_calibration_audit_episode(self.capture, self.world, policy, self.root, horizon=4)
        self.assertEqual(result["partition"], "calibration_audit")
        self.assertTrue(result["calibration_audit"])
        self.assertFalse(result["VLA_called"])
        self.assertFalse(result["real_dynamics_advanced"])
        self.assertEqual(len(policy.inputs), 2)
        self.assertEqual(len(result["journal"]), 2)
        self.assertEqual(result["submissions"], [])
        self.assertEqual(self.capture, original)

    def test_audit_entry_refuses_fit_train_validation_and_test(self):
        for split in ("calibration_fit", "train", "validation", "test", "test_sealed"):
            policy = WaitPolicy()
            with self.assertRaisesRegex(ValueError, "audit only"):
                run_calibration_audit_episode({**self.capture, "partition": split}, self.world, policy, self.root)
            self.assertEqual(policy.inputs, [])

    def test_audit_stop_cancel_bypass_policy_before_any_prediction(self):
        for control in ("STOP", "CANCEL"):
            policy = WaitPolicy()
            result = run_calibration_audit_episode(self.capture, self.world, policy, self.root, horizon=4, control_at=(0., control))
            self.assertEqual(policy.inputs, [])
            self.assertEqual(result["journal"][0]["kind"], "CONTROL")
            self.assertEqual(result["journal"][0]["event"]["role"], control)
            self.assertFalse(result["submissions"])

    def test_declared_pick_is_public_for_all_candidates_without_changing_object_questions(self):
        declared, legacy = WaitPolicy(), WaitPolicy()
        result = run_calibration_audit_episode(self.capture, self.world, declared, self.root, horizon=2, declared_action="pick")
        run_calibration_audit_episode(self.capture, self.world, legacy, self.root, horizon=2)
        public = declared.inputs[0]
        self.assertEqual(legacy.inputs[0]["task_options"], [])
        self.assertEqual(result["declared_action"], "pick")
        self.assertEqual({t["source_id"] for t in public["task_options"]}, {"object-a", "object-b"})
        self.assertTrue(all(t["action"] == "pick" and t["destination_id"] is None for t in public["task_options"]))
        question = make_question(public, {"action": "TASK"}, 1)
        self.assertTrue(all(o["kind"] == "OBJECT" for o in question["options"]))
        self.assertEqual(question, make_question(legacy.inputs[0], {"action": "TASK"}, 1))
        with self.assertRaisesRegex(ValueError, "public pick protocol"):
            run_calibration_audit_episode(self.capture, self.world, WaitPolicy(), self.root, declared_action="place")

    def test_online_outcomes_use_policy_histories_and_pool_hidden_worlds(self):
        first = run_calibration_audit_episode(self.capture, self.world, WaitPolicy(), self.root, horizon=4)
        other_world = {**self.world, "initial_goal": "object-b", "later_goal": "object-b"}
        second = run_calibration_audit_episode(self.capture, other_world, WaitPolicy(), self.root, horizon=4)
        calls = []
        def predictor(public):
            self.assertNotIn("hidden_truth", public)
            calls.append(public)
            return {"candidate_ids": [o["instance_id"] for o in public["scene"]["objects"]],
                    "intent_logits": [1., 1., 0.], "channel_supported": False, "channel_logit": None}
        binding = {"model_sha256": "a"*64, "data_sha256": "b"*64, "input_schema_sha256": "c"*64, "policy_sha256": "d"*64}
        rows = online_audit_rows([(first, self.world, "unit-scene"), (second, other_world, "unit-scene")], predictor, binding)
        self.assertEqual(len(calls), 2)  # Two unique public timestamps, not four latent rows.
        self.assertTrue(all(r["target"] == [.5, .5, 0.] for r in rows["intent_rows"]))
        self.assertTrue(all(r["split"] == "calibration_audit" for r in rows["intent_rows"]))
        self.assertTrue(all(not r["mask"] and r["target"] is None for r in rows["channel_rows"]))
        first["partition"] = "validation"
        with self.assertRaisesRegex(ValueError, "audit only"):
            online_audit_rows([(first, self.world, "unit-scene")], predictor, binding)


class OnlineAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.criteria = {"max_wrong_episode_rate": .1, "min_resolved_fraction": 0.,
                         "min_probability_samples": 8, "min_probability_groups": 2, "nll_tolerance": 1e-8}
        self.metrics = {task: {"status": "AUDITED", "samples": 12, "groups": 2,
                              "before": {"nll": 1.}, "after": {"nll": .8}} for task in ("intent", "channel")}
        self.rows = [{"split": "calibration_audit", "group_id": "g" + str(i % 2), "episode_id": str(i),
                      "resolved_correctly": True, "wrong_submission": False} for i in range(8)]

    def test_always_wait_and_low_coverage_fail_while_nll_degradation_is_visible(self):
        self.assertTrue(online_audit_acceptance(self.rows, self.metrics, self.criteria)["passed"])
        waiting = [{**r, "resolved_correctly": False} for r in self.rows]
        self.assertFalse(online_audit_acceptance(waiting, self.metrics, self.criteria)["passed"])
        self.metrics["channel"]["samples"] = 0
        self.assertFalse(online_audit_acceptance(self.rows, self.metrics, self.criteria)["passed"])
        self.metrics["channel"]["samples"] = 12
        self.metrics["intent"]["after"]["nll"] = 2.
        report = online_audit_acceptance(self.rows, self.metrics, self.criteria)
        self.assertTrue(report["passed"])
        self.assertFalse(report["diagnostics"]["intent_nll_not_worse"])

    def test_wrong_episode_and_split_constraints_do_not_get_averaged_away(self):
        self.rows[0]["wrong_submission"] = True
        self.assertFalse(online_audit_acceptance(self.rows, self.metrics, self.criteria)["passed"])
        self.rows[0]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "audit episodes"):
            online_audit_acceptance(self.rows, self.metrics, self.criteria)

    def test_world_conditions_are_not_four_independent_base_episodes(self):
        conditions = [{"split": "calibration_audit", "policy_id": "fixed", "group_id": "g", "episode_id": "e",
                       "scenario": scenario, "resolved_correctly": scenario != "both_change", "wrong_submission": scenario == "goal_change",
                       "interaction_cost": 10.} for scenario in SCENARIOS]
        rows = aggregate_base_summaries(conditions, "calibration_audit")
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["resolved_correctly"])
        self.assertTrue(rows[0]["wrong_submission"])
        self.assertEqual(rows[0]["interaction_cost"], 40.)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            aggregate_base_summaries(conditions[:-1], "calibration_audit")

    def test_frozen_configs_have_no_dynamic_run_flags(self):
        template = json.loads((ROOT / "configs/v4/calibration_and_policy.json").read_text(encoding="utf-8"))
        for config in candidates(template).values():
            self.assertEqual(config["status"], "FROZEN_DEVELOPMENT_POLICY")
            self.assertTrue(config["policy_selection"]["require_positive_resolution"])
            for field in ("scope", "formal_test_access", "calibration_run", "policy_validation_run"):
                self.assertNotIn(field, config)


if __name__ == "__main__":
    unittest.main()
