"""Meaningful CPU calibration/policy boundaries; no model or real EEG claims."""
import copy
import json
import math
from pathlib import Path
import unittest

from bci_vla.calibration_v4 import (fit_temperature, evaluate_calibration, calibrate_prediction,
                                    decide_interaction, policy_hash, select_policy)
from bci_vla.interaction_contract_v4.validation import canonical_digest
from bci_vla.data_v4.channel import diagnostic_summary

ROOT = Path(__file__).resolve().parents[1]


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.binding = {"model_sha256": "a" * 64, "data_sha256": "b" * 64,
                        "input_schema_sha256": "c" * 64, "policy_sha256": "d" * 64}

    def rows(self, task="intent", split="calibration_fit", group="fit-scene"):
        # All positive high-confidence logits, 50% correct: known overconfidence.
        return [{"split": split, "logits": [5., -5.] if task == "intent" else 10.,
                 "target": i % 2, "target_mode": "POINT", "mask": True, "supported": True,
                 "group_id": group, "episode_id": "episode-" + str(i // 2),
                 "metadata": {"binding": self.binding.copy()}} for i in range(8)]

    def test_temperature_reduces_nll_on_overconfident_predictions(self):
        for task in ("intent", "channel"):
            artifact = fit_temperature(self.rows(task), task=task, binding=self.binding)
            self.assertEqual(artifact["status"], "FITTED")
            self.assertGreater(artifact["temperature"], 1.)
            self.assertLess(artifact["nll_after"], artifact["nll_before"])
            audit = evaluate_calibration(self.rows(task, "calibration_audit", "audit-scene"), artifact,
                                         task=task, binding=self.binding)
            self.assertEqual(audit["status"], "AUDITED")
            for metric in ("brier", "nll", "ece"):
                self.assertLess(audit["after"][metric], audit["before"][metric])
            self.assertFalse(audit["fit_performed"])

    def test_partition_leakage_rejected_including_masked_rows(self):
        artifact = fit_temperature(self.rows(), task="intent", binding=self.binding)
        for split in ("train", "validation", "calibration_audit", "test", "test_sealed"):
            rows = self.rows(split=split)
            rows[0]["mask"] = False
            with self.assertRaisesRegex(ValueError, "partition leak"):
                fit_temperature(rows, task="intent", binding=self.binding)
        for split in ("train", "validation", "calibration_fit", "test", "test_sealed"):
            with self.assertRaisesRegex(ValueError, "partition leak"):
                evaluate_calibration(self.rows(split=split), artifact, task="intent", binding=self.binding)
        with self.assertRaisesRegex(ValueError, "scene groups overlap"):
            evaluate_calibration(self.rows(split="calibration_audit"), artifact, task="intent", binding=self.binding)

    def test_all_four_version_hash_changes_refuse_old_artifact(self):
        artifact = fit_temperature(self.rows(), task="intent", binding=self.binding)
        for key in self.binding:
            changed = self.binding.copy(); changed[key] = "e" * 64
            with self.assertRaisesRegex(ValueError, "stale"):
                calibrate_prediction([1., 0.], artifact, task="intent", binding=changed)
            with self.assertRaises(ValueError):
                fit_temperature(self.rows(), task="intent", binding=changed)

    def test_masks_nonpoint_and_unsupported_never_force_targets(self):
        rows = self.rows()
        rows[0].update(mask=False, logits=None, target=None)
        rows[1].update(supported=False, logits=None, target=None)
        rows[2].update(target_mode="SET", logits=None, target=None)
        artifact = fit_temperature(rows, task="intent", binding=self.binding)
        self.assertEqual(artifact["samples"], 5)
        self.assertEqual(artifact["skipped"], {"masked": 1, "unsupported": 1, "nonpoint": 1})
        prediction = calibrate_prediction(None, artifact, task="intent", binding=self.binding, supported=False)
        self.assertIsNone(prediction["probabilities"])
        self.assertIsNone(prediction["confidence"])
        self.assertFalse(prediction["calibrated"])
        for row in rows:
            row["mask"] = False
        unfitted = fit_temperature(rows, task="intent", binding=self.binding)
        self.assertIsNone(unfitted["temperature"])
        self.assertIsNone(calibrate_prediction([1., 2.], unfitted, task="intent", binding=self.binding)["confidence"])

    def test_channel_masks_are_independent_from_intent_modes(self):
        rows = self.rows("channel")
        for row in rows:
            row["target_mode"] = "SET"
        artifact = fit_temperature(rows, task="channel", binding=self.binding)
        self.assertEqual(artifact["samples"], 8)
        self.assertEqual(artifact["skipped"]["nonpoint"], 0)
        with self.assertRaises(ValueError):
            fit_temperature([{**rows[0], "target": None}] * 2, task="channel", binding=self.binding)

    def test_episode_duplication_does_not_change_macro_calibration(self):
        rows = self.rows()
        a = fit_temperature(rows, task="intent", binding=self.binding)
        duplicated = rows + [copy.deepcopy(r) for r in rows if r["episode_id"] == "episode-0"] * 9
        b = fit_temperature(duplicated, task="intent", binding=self.binding)
        self.assertAlmostEqual(a["temperature"], b["temperature"], places=6)
        self.assertAlmostEqual(a["nll_after"], b["nll_after"], places=10)


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.publics = json.loads((ROOT / "execution/v4/06_fixtures/public.json").read_text(encoding="utf-8"))

    def setUp(self):
        self.public = copy.deepcopy(next(p for p in self.publics if p["public_state"]["diagnostics"]["status"] == "SUFFICIENT"))
        self.config = json.loads((ROOT / "configs/v4/calibration_and_policy_template.json").read_text(encoding="utf-8"))
        self.config["selection_freshness_s"] = 100.
        self.binding = {"model_sha256": "a" * 64, "data_sha256": "b" * 64,
                        "input_schema_sha256": "c" * 64, "policy_sha256": policy_hash(self.config)}
        self.state = {"questions": 1, "probes": 0, "intent_queries": 0, "retries": 0}

    def prediction(self, public=None):
        public = self.public if public is None else public
        ids = [o["instance_id"] for o in public["scene"]["objects"] if o["object_valid"]]
        return {"candidate_ids": ids, "probabilities": [.95] + [.05 / len(ids)] * len(ids),
                "intent_calibrated": True, "channel_probability": .9, "channel_supported": True,
                "channel_calibrated": True, "input_sha256": canonical_digest(public),
                "context": copy.deepcopy(public["context"]), "as_of_s": public["as_of_s"],
                "binding": self.binding.copy(), "auxiliary_action": "PROPOSE_TASK"}

    def decide(self, pred=None, public=None, variant="fixed_threshold", state=None):
        config, binding = copy.deepcopy(self.config), self.binding.copy()
        config["variant"] = variant
        if variant != "fixed_threshold":
            binding["policy_sha256"] = policy_hash(config)
        pred = self.prediction() if pred is None else copy.deepcopy(pred)
        if variant != "fixed_threshold":
            pred["binding"]["policy_sha256"] = binding["policy_sha256"]
        return decide_interaction(self.public if public is None else public, pred,
                                  config, variant=variant, binding=binding, state=self.state if state is None else state)

    def test_high_confidence_still_only_requests_exact_confirmation(self):
        result = self.decide()
        self.assertEqual(result["action"], "CONFIRM")
        self.assertTrue(result["requires_user_confirmation"])
        self.assertFalse(result["execution_authorized"])

    def test_evidence_gate_overrides_confidence_and_respects_question_budgets(self):
        required = 1 + sum(e['role'] in ('TASK','INTENT_QUERY') for e in self.public['history'])
        self.config['confirmation_gate'] = {'mode':'consistent_recheck','consistent_selections':max(2,required),
                                            'correct_probes':2,'selection_max_age_s':10.}
        self.binding['policy_sha256'] = policy_hash(self.config)
        result = self.decide()
        self.assertEqual(result['action'],'INTENT_QUERY')
        self.assertFalse(result['execution_authorized'])
        exhausted = {**self.state,'intent_queries':self.config['budgets']['intent_queries']}
        self.assertEqual(self.decide(state=exhausted)['action'],'WAIT')

    def test_rejected_proposal_needs_new_selection_despite_high_confidence(self):
        public = copy.deepcopy(self.public)
        fixture = json.loads((ROOT / "examples/v4/reject_clears_task.json").read_text(encoding="utf-8"))
        rejected = copy.deepcopy(fixture["history"][-1])
        now = public["as_of_s"] + 2.
        public["as_of_s"] = now
        rejected.update(event_id="new-rejected-proposal", sequence=max(e["sequence"] for e in public["history"]) + 1,
                        started_at_s=now - .75, ended_at_s=now - .25)
        question = rejected["question"]
        question.update(question_id="new-rejected-question", context=copy.deepcopy(public["context"]),
                        issued_at_s=now - 1., expires_at_s=now + .5)
        question["proposal"].update(context=copy.deepcopy(public["context"]),
                                   source_id=public["scene"]["objects"][0]["instance_id"], valid_until_s=now + 30.)
        public["history"] = public["history"][-7:] + [rejected]
        public["current_event_id"] = rejected["event_id"]
        public["public_state"]["confirmed_task"] = None
        result = self.decide(self.prediction(public), public)
        self.assertEqual(result["action"], "INTENT_QUERY")
        self.assertEqual(result["reason"], "REJECTED_PROPOSAL_REQUIRES_NEW_SELECTION")
        self.assertFalse(result["execution_authorized"])

    def test_changed_implementation_refuses_hash_matching_old_policy_config(self):
        import hashlib
        paths = ("bci_vla/calibration_v4/policy.py", "bci_vla/evaluation_v4/runner.py",
                 "bci_vla/calibration_v4/evidence_gate.py",
                 "bci_vla/data_v4/channel.py", "bci_vla/calibration_v4/calibration.py",
                 "execution/v4/step09_calibrate_and_policy.py")
        self.config["implementation_binding"] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
        self.binding["policy_sha256"] = policy_hash(self.config)
        self.assertEqual(self.decide()["action"], "CONFIRM")
        self.config["implementation_binding"][paths[0]] = "0" * 64
        self.binding["policy_sha256"] = policy_hash(self.config)
        with self.assertRaisesRegex(ValueError, "implementation changed"):
            self.decide()

    def test_stop_cancel_bypass_even_missing_invalid_predictions_and_config(self):
        for name in ("stop_clears_task.json", "cancel_clears_task.json"):
            public = json.loads((ROOT / "examples/v4" / name).read_text(encoding="utf-8"))
            result = decide_interaction(public, None, {}, binding=None)
            self.assertEqual(result["action"], "WAIT")
            self.assertFalse(result["call_neural_module"])
            self.assertFalse(result["execution_authorized"])
            self.assertTrue(result["control_route"].endswith("EXISTING_PATH"))

    def test_pending_question_has_priority_over_confidence(self):
        public = copy.deepcopy(self.public)
        question = copy.deepcopy(public["history"][-1]["question"])
        question.update(question_id="pending-new-q", issued_at_s=public["as_of_s"], expires_at_s=public["as_of_s"] + 2)
        public["public_state"]["pending_question"] = question
        result = decide_interaction(public, None, {}, binding=None)
        self.assertEqual(result["reason"], "QUESTION_PENDING")

    def test_probe_and_intent_clarification_are_distinct(self):
        pred = self.prediction(); pred["channel_probability"] = .1
        self.assertEqual(self.decide(pred)["action"], "PROBE")
        pred = self.prediction(); pred["probabilities"] = [1 / len(pred["probabilities"])] * len(pred["probabilities"])
        self.assertEqual(self.decide(pred)["action"], "INTENT_QUERY")
        self.assertEqual(self.decide(pred, variant="entropy_threshold")["action"], "INTENT_QUERY")
        pred = self.prediction(); pred["channel_supported"] = False; pred["channel_probability"] = None
        self.assertEqual(self.decide(pred)["action"], "PROBE")

    def test_stale_input_mapping_calibration_and_policy_hashes_are_blocked(self):
        pred = self.prediction(); pred["context"]["mapping_revision"] += 1
        self.assertEqual(self.decide(pred)["reason"], "STALE_PREDICTION_OR_MAPPING")
        pred = self.prediction(); pred["binding"]["model_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "stale prediction"):
            self.decide(pred)
        self.config["thresholds"]["intent_confidence"] = .99
        with self.assertRaisesRegex(ValueError, "policy config changed"):
            self.decide()

    def test_budget_and_outside_unknown_are_visible(self):
        state = {**self.state, "questions": self.config["budgets"]["questions"]}
        self.assertEqual(self.decide(state=state)["reason"], "INTERACTION_BUDGET_EXHAUSTED")
        pred = self.prediction(); pred["probabilities"] = [.1 / len(pred["candidate_ids"])] * len(pred["candidate_ids"]) + [.9]
        self.assertEqual(self.decide(pred)["reason"], "TARGET_MAY_BE_OUTSIDE_CANDIDATES")
        pred["intent_calibrated"] = False; pred["probabilities"] = None
        self.assertEqual(self.decide(pred)["reason"], "INTENT_CALIBRATION_UNSUPPORTED")

    def test_existing_confirmed_task_does_not_authorize_or_lock_new_target(self):
        # An actual contract fixture includes a valid confirmation proof.
        public = json.loads((ROOT / "examples/v4/confirmed_task_survives_nine_probes.json").read_text(encoding="utf-8"))
        # Turn the last observation into an explicit task selection, while
        # retaining its other valid diagnostic receipts and prior confirmation.
        event = public["history"][-1]
        event["role"] = "TASK"; event["original_role"] = None
        event["question"]["role"] = "TASK"
        event["question"]["diagnostic_option_id"] = None
        for option in event["question"]["options"]:
            option["kind"] = "OBJECT"
        public["selection_event_id"] = event["event_id"]
        receipts = [e for e in public["public_state"]["diagnostics"]["receipts"] if e["event_id"] != event["event_id"]]
        layout = receipts[0]["question"]["layout_id"]
        public["public_state"]["diagnostics"] = diagnostic_summary(public["context"], receipts, public["as_of_s"], layout)
        pred = self.prediction(public)
        old = public["public_state"]["confirmed_task"]["task"]["source_id"]
        old_index = pred["candidate_ids"].index(old)
        pred["probabilities"] = [.05 / len(pred["candidate_ids"])] * len(pred["probabilities"])
        pred["probabilities"][old_index] = .95
        result = self.decide(pred, public)
        self.assertEqual(result["reason"], "CONFIRMED_TASK_REQUIRES_SEPARATE_EXECUTION_AUTHORIZATION")
        self.assertFalse(result["execution_authorized"])
        new_index = (old_index + 1) % len(pred["candidate_ids"])
        pred["probabilities"][old_index], pred["probabilities"][new_index] = pred["probabilities"][new_index], pred["probabilities"][old_index]
        changed = self.decide(pred, public)
        self.assertEqual(changed["action"], "CONFIRM")
        self.assertEqual(changed["source_id"], pred["candidate_ids"][new_index])
        self.assertFalse(changed["execution_authorized"])

    def test_variant_cannot_change_without_updating_policy_hash(self):
        with self.assertRaisesRegex(ValueError, "variant differs"):
            decide_interaction(self.public, self.prediction(), self.config, variant="entropy_threshold",
                               binding=self.binding, state=self.state)

    def test_auxiliary_cannot_override_channel_or_confirmation_guards(self):
        pred = self.prediction(); pred["channel_probability"] = .1
        self.assertEqual(self.decide(pred, variant="auxiliary")["action"], "PROBE")
        pred = self.prediction(); pred["auxiliary_action"] = "ASK_INTENT"
        self.assertEqual(self.decide(pred, variant="auxiliary")["action"], "INTENT_QUERY")

    def test_probe_object_reference_is_not_a_task_selection(self):
        public = copy.deepcopy(self.public)
        ids = [o["instance_id"] for o in public["scene"]["objects"]]
        for collection in (public["history"], public["public_state"]["diagnostics"]["receipts"]):
            for event in collection:
                if not event.get("question") or event["question"]["role"] != "PROBE":
                    continue
                for option in event["question"]["options"]:
                    if option["reference"] in ids:
                        option["reference"] = ids[(ids.index(option["reference"]) + 1) % len(ids)]
        before = self.decide()
        after = self.decide(self.prediction(public), public)
        self.assertEqual(before["action"], after["action"])
        self.assertEqual(before["source_id"], after["source_id"])

    def test_validation_only_selection_paired_episodes_and_all_wait_failure(self):
        fixed = copy.deepcopy(self.config); fixed["variant"] = "fixed_threshold"
        aux = copy.deepcopy(self.config); aux["variant"] = "auxiliary"
        candidates = {"fixed": fixed, "aux": aux}
        rows = [{"split": "validation", "policy_id": pid, "group_id": "scene", "episode_id": "ep" + str(i),
                 "resolved_correctly": i == 0, "wrong_submission": False, "interaction_cost": 1 if pid == "aux" else 2}
                for pid in candidates for i in range(2)]
        report = select_policy(rows, candidates=candidates)
        self.assertEqual(report["selected_policy_id"], "fixed")
        self.assertTrue(report["retained_fixed_rule"])
        for split in ("train", "calibration_fit", "calibration_audit", "test", "test_sealed"):
            with self.assertRaisesRegex(ValueError, "validation only"):
                select_policy([{**rows[0], "split": split}], candidates=candidates)
        with self.assertRaisesRegex(ValueError, "same complete"):
            select_policy(rows[:-1], candidates=candidates)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_policy(rows + [rows[0]], candidates=candidates)
        waiting = [{**r, "resolved_correctly": False} for r in rows]
        self.assertIsNone(select_policy(waiting, candidates=candidates, min_resolved_fraction=.1)["selected_policy_id"])

    def test_condition_average_preserves_base_episode_risk_and_sample_count(self):
        config = copy.deepcopy(self.config)
        rows = [{"split": "validation", "policy_id": "fixed", "group_id": "scene", "episode_id": "e" + str(i),
                 "resolved_correctly": False, "condition_resolution_fraction": .5,
                 "wrong_submission": False, "interaction_cost": 12.} for i in range(2)]
        result = select_policy(rows, candidates={"fixed": config}, min_resolved_fraction=.1,
                               resolution_field="condition_resolution_fraction")
        score = result["scores"]["fixed"]
        self.assertEqual(result["selected_policy_id"], "fixed")
        self.assertEqual(score["episodes"], 2)
        self.assertEqual(score["resolved_fraction"], .5)
        self.assertEqual(score["all_conditions_resolved_fraction"], 0.)
        rows[0]["wrong_submission"] = True
        unsafe = select_policy(rows, candidates={"fixed": config}, max_wrong_episode_rate=.1,
                               resolution_field="condition_resolution_fraction")
        self.assertEqual(unsafe["scores"]["fixed"]["wrong_episode_rate"], .5)
        self.assertIsNone(unsafe["selected_policy_id"])


if __name__ == "__main__":
    unittest.main()
