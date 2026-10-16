import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError

from scene_context.baselines import RulePredictor
from scene_context.evaluation import _bootstrap, evaluate, paired_scene_comparison
from scene_context.llm import LLMPredictor, build_prompt, parse_response
from tests.fixtures import case, observation, provider_response, ready_proposal


class LLMOfflineTests(unittest.TestCase):
    def test_constructor_has_no_side_effect(self):
        transport = Mock(return_value=provider_response())
        LLMPredictor(transport=transport)
        transport.assert_not_called()

    def test_cloud_disabled_cannot_touch_network(self):
        with patch("scene_context.llm.request.build_opener", side_effect=AssertionError("network forbidden")) as opener:
            result = LLMPredictor(allow_cloud=False, max_retries=0).predict(observation())
        opener.assert_not_called()
        self.assertEqual(result["status"], "reject")
        self.assertEqual(result["error"], "PermissionError")

    def test_injected_valid_provider_is_shared_guarded(self):
        transport = Mock(return_value=provider_response())
        result = LLMPredictor(transport=transport, max_retries=0).predict(observation())
        self.assertEqual(result["status"], "ready")
        self.assertTrue(result["requires_confirmation"])
        self.assertEqual(transport.call_count, 1)
        prompt = transport.call_args.args[0]["contents"][0]["parts"][0]["text"]
        self.assertIn("PUBLIC_OBSERVATION_JSON", prompt)
        self.assertNotIn('"annotation"', prompt)

    def test_malformed_and_incomplete_provider_outputs_fail_closed(self):
        responses = [provider_response("not JSON"), provider_response("[]"),
                     provider_response(finish_reason="MAX_TOKENS"), {},
                     provider_response({"status": "ready", "intent": None, "clarification_slot": None}),
                     provider_response({**ready_proposal(), "robot_command": "execute"})]
        for response in responses:
            with self.subTest(response=response):
                result = LLMPredictor(transport=lambda payload: response, max_retries=0).predict(observation())
                self.assertEqual(result["status"], "reject")
                self.assertIsNone(result["instruction"])
                self.assertIn("error", result)

    def test_unknown_instance_and_ambiguous_intent_cannot_be_released(self):
        hallucination = LLMPredictor(transport=lambda _: provider_response(ready_proposal("missing")))
        self.assertEqual(hallucination.predict(observation())["status"], "reject")
        ambiguity = LLMPredictor(transport=lambda _: provider_response())
        result = ambiguity.predict(observation(True))
        self.assertEqual(result["raw_proposal"]["status"], "ready")
        self.assertEqual(result["status"], "clarify")

    def test_api_failure_is_not_guessed_pickup_and_does_not_log_credentials(self):
        sensitive = "SECRET_NOT_FOR_LOGGING_492"
        transport = Mock(side_effect=URLError(sensitive))
        result = LLMPredictor(transport=transport, max_retries=0).predict(observation())
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(result["status"], "reject")
        self.assertNotIn(sensitive, json.dumps(result))
        self.assertEqual(result["error"], "URLError")

    def test_duplicate_json_keys_are_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            parse_response('{"status":"ready","status":"reject","intent":null,"clarification_slot":null}')

    def test_prompt_rejects_hidden_extra_fields(self):
        obs = observation()
        obs["annotation"] = {"answer": "hidden"}
        with self.assertRaises(ValueError):
            build_prompt(obs)

    def test_frozen_cache_replays_without_second_provider_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            transport = Mock(return_value=provider_response())
            first = LLMPredictor(transport=transport, cache_dir=tmp, max_retries=0)
            first.predict(observation())
            second = LLMPredictor(cache_dir=tmp, cache_only=True, max_retries=0)
            replay = second.predict(observation())
            self.assertEqual(transport.call_count, 1)
            self.assertTrue(replay["llm_metadata"]["cache_hit"])
            self.assertEqual(replay["status"], "ready")
            cached = json.loads(next(Path(tmp).glob("*.json")).read_text(encoding="utf-8"))
            self.assertNotIn("raw_completion", cached)
            self.assertNotIn("api_key", json.dumps(cached))

    def test_cache_key_changes_with_model_or_input(self):
        model = LLMPredictor()
        obs = observation()
        changed = copy.deepcopy(obs)
        changed["context"]["goal"] = "pick"
        self.assertNotEqual(model.request_key(obs), model.request_key(changed))
        self.assertNotEqual(model.request_key(obs), LLMPredictor(model="other-model").request_key(obs))


class EvaluationTests(unittest.TestCase):
    def test_predictor_never_receives_annotations_or_metadata(self):
        received = []
        class Spy:
            def predict(self, obs):
                received.append(copy.deepcopy(obs))
                return RulePredictor().predict(obs)
        item = case()
        item["hidden_request"] = "DO_NOT_EXPOSE"
        evaluate(Spy(), [item], bootstrap_samples=0)
        self.assertEqual(len(received), 1)
        self.assertNotIn("annotation", received[0])
        self.assertNotIn("case_id", received[0])
        self.assertNotIn("DO_NOT_EXPOSE", json.dumps(received[0]))

    def test_raw_error_is_visible_even_if_common_guard_clarifies(self):
        llm = LLMPredictor(transport=lambda _: provider_response(), max_retries=0)
        report = evaluate(llm, [case(ambiguous=True)], bootstrap_samples=0)
        self.assertEqual(report["metrics"]["raw"]["ready_when_should_not_rate"], 1)
        self.assertEqual(report["metrics"]["guarded"]["ready_when_should_not_rate"], 0)
        self.assertEqual(report["metrics"]["guarded"]["decision_exact_match"], 1)

    def test_no_released_instructions_has_undefined_precision_not_perfect_safety(self):
        report = evaluate(RulePredictor(), [case(ambiguous=True)], bootstrap_samples=0)
        self.assertEqual(report["metrics"]["guarded"]["ready_coverage"], 0)
        self.assertIsNone(report["metrics"]["guarded"]["instruction_precision_on_ready"])
        self.assertIsNone(report["metrics"]["guarded"]["wrong_ready_rate_on_ready"])

    def test_duplicate_cases_fail_before_inference(self):
        predictor = Mock()
        with self.assertRaises(ValueError):
            evaluate(predictor, [case(), case()], bootstrap_samples=0)
        predictor.predict.assert_not_called()

    def test_provider_failure_stays_in_denominator_and_reports_only_error_type(self):
        predictor = Mock(spec=["predict"])
        predictor.predict.side_effect = RuntimeError("SECRET_RUNTIME_DETAIL")
        report = evaluate(predictor, [case()], bootstrap_samples=0)
        self.assertEqual(report["metrics"]["n_cases"], 1)
        self.assertEqual(report["metrics"]["prediction_error_rate"], 1)
        self.assertEqual(report["metrics"]["guarded"]["instruction_recall_on_gold_ready"], 0)
        self.assertNotIn("SECRET_RUNTIME_DETAIL", json.dumps(report))

    def test_bootstrap_draws_whole_scenes_including_all_variants(self):
        rows = [{"scene_id": "A", "case_id": "A1"}, {"scene_id": "A", "case_id": "A2"},
                {"scene_id": "A", "case_id": "A3"}, {"scene_id": "B", "case_id": "B1"}]
        samples = []
        def aggregate(sample):
            samples.append(sample)
            return {"raw": {"metric": 0.5}, "guarded": {}}
        with patch("scene_context.evaluation._aggregate", side_effect=aggregate):
            _bootstrap(rows, samples=40, seed=17)
        self.assertEqual(len(samples), 40)
        for sample in samples:
            counts = {identifier: sum(row["case_id"] == identifier for row in sample)
                      for identifier in ("A1", "A2", "A3", "B1")}
            self.assertEqual(counts["A1"], counts["A2"])
            self.assertEqual(counts["A2"], counts["A3"])
            self.assertEqual(counts["A1"] + counts["B1"], 2)

    def test_paired_comparison_requires_identical_frozen_dataset(self):
        one = evaluate(RulePredictor(), [case()], bootstrap_samples=0)
        two = evaluate(RulePredictor(), [case("other_case", "other_scene")], bootstrap_samples=0)
        with self.assertRaises(ValueError):
            paired_scene_comparison(one, two, bootstrap_samples=10)
        same = paired_scene_comparison(one, one, bootstrap_samples=10)
        for metric in same["differences"].values():
            self.assertEqual(metric["difference"], 0)
            if metric["defined_resamples"]:
                self.assertEqual(metric["low"], 0)
                self.assertEqual(metric["high"], 0)


if __name__ == "__main__":
    unittest.main()
