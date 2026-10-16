"""Offline boundary tests; no policy load, networking or robot execution."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from bci_vla.adapters.vla import DryRunVLAAdapter, FakeVLAAdapter, VLAInterfaceSpec
from bci_vla.protocols import ProtocolError

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "configs/execution/fake_vla_interface.json").read_text(encoding="utf-8"))


def observation():
    return config()["fixture_observation"]


def actions():
    return config()["fixture_actions"]


def submit(adapter, submission_id="submission-1", instruction="Place grapes in the left bowl."):
    return adapter.submit(submission_id, "contract-1", 1, instruction, observation())


class VLAInterfaceTests(unittest.TestCase):
    def test_fixture_spec_and_observation_roundtrip(self):
        spec = VLAInterfaceSpec(config())
        self.assertEqual(spec.validate_observation(observation()), observation())
        self.assertEqual(spec.validate_actions(actions()), actions())
        self.assertFalse(spec.data["deployment_allowed"])
        self.assertIn("exact_pi05_checkpoint", spec.data["unverified_live_fields"])

    def test_live_or_unsupported_spec_rejected(self):
        for key, value in (("scope", "live"), ("deployment_allowed", True),
                           ("interface_version", "openpi-live"), ("schema_version", "2.0")):
            changed = config(); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ProtocolError):
                VLAInterfaceSpec(changed)

    def test_image_keys_must_match_exactly(self):
        for alteration in ("missing", "extra", "renamed"):
            value = observation()
            if alteration == "missing":
                del value["images"]["right"]
            elif alteration == "extra":
                value["images"]["overhead"] = deepcopy(value["images"]["front"])
            else:
                value["images"]["observation.images.front"] = value["images"].pop("front")
            with self.subTest(case=alteration), self.assertRaisesRegex(ProtocolError, "image_keys_mismatch"):
                VLAInterfaceSpec().validate_observation(value)

    def test_image_shapes_channels_and_dtype(self):
        for pixels in ([], [[[0, 0, 0]]], [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
                       [[[0.5, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]],
                       [[[256, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]):
            value = observation(); value["images"]["front"] = pixels
            with self.subTest(pixels=pixels), self.assertRaises(ProtocolError):
                VLAInterfaceSpec().validate_observation(value)

    def test_state_shape_finite_bounds_and_bool(self):
        for state in ([0] * 13, [0] * 15, [float("nan")] + [0] * 13,
                      [float("inf")] + [0] * 13, [2] + [0] * 13, [True] + [0] * 13):
            value = observation(); value["state"] = state
            with self.subTest(state=repr(state)), self.assertRaises(ProtocolError):
                VLAInterfaceSpec().validate_observation(value)

    def test_state_units_coordinates_normalization_and_version(self):
        for key, value in (("state_units", ["radians"] * 14),
                           ("coordinate_frame", "robot_base"),
                           ("normalization", "unverified_checkpoint_stats"),
                           ("interface_version", "fake-vla/2.0")):
            changed = observation(); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ProtocolError):
                VLAInterfaceSpec().validate_observation(changed)

    def test_observation_scene_revision_and_timestamp_required(self):
        for key, value in (("scene_id", ""), ("scene_revision", 0),
                           ("captured_at", "2026-10-15T00:00:00")):
            changed = observation(); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ProtocolError):
                VLAInterfaceSpec().validate_observation(changed)

    def test_action_horizon_and_vector_shape(self):
        for values in ([], [[0] * 14], [[0] * 13, [0] * 14], [[0] * 14] * 3):
            changed = actions(); changed["values"] = values
            with self.subTest(values=values), self.assertRaises(ProtocolError):
                VLAInterfaceSpec().validate_actions(changed)

    def test_action_nonfinite_bounds_units_and_coordinates(self):
        for field, value in (("values", [[float("inf")] + [0] * 13, [0] * 14]),
                             ("values", [[-2] + [0] * 13, [0] * 14]),
                             ("units", ["meters"] * 14),
                             ("coordinate_frame", "base"),
                             ("normalization", "different"),
                             ("interface_version", "2.0")):
            changed = actions(); changed[field] = value
            with self.subTest(field=field), self.assertRaises(ProtocolError):
                VLAInterfaceSpec().validate_actions(changed)

    def test_config_bounds_and_units_are_checked(self):
        for field, value in (("dimension", 0), ("units", []),
                             ("lower", [2] * 14), ("horizon", 0)):
            changed = config(); changed["actions"][field] = value
            with self.subTest(field=field), self.assertRaises(ProtocolError):
                VLAInterfaceSpec(changed)


class FakeAdapterTests(unittest.TestCase):
    def test_submit_acceptance_is_not_start_or_execution(self):
        adapter = FakeVLAAdapter()
        result = submit(adapter)
        self.assertEqual(result["status"], "ACCEPTED")
        self.assertEqual(result["execution_status"], "NOT_STARTED")
        self.assertIsNone(result["last_feedback_id"])
        self.assertFalse(result["actual_execution"])
        self.assertEqual(adapter.submission_count, 1)

    def test_duplicate_submission_returns_original_without_second_submit(self):
        adapter = FakeVLAAdapter()
        submit(adapter); again = submit(adapter)
        self.assertTrue(again["duplicate"])
        self.assertEqual(adapter.submission_count, 1)
        self.assertEqual(len(adapter.audit_log), 1)

    def test_conflicting_payload_cannot_reuse_submission_id(self):
        adapter = FakeVLAAdapter(); submit(adapter)
        with self.assertRaisesRegex(ProtocolError, "submission_id_payload_conflict"):
            submit(adapter, instruction="Place grapes in the right bowl.")
        self.assertEqual(adapter.submission_count, 1)

    def test_unknown_acceptance_is_not_automatically_retried(self):
        adapter = FakeVLAAdapter(submit_status="UNKNOWN")
        submit(adapter); result = submit(adapter)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(result["execution_status"], "UNKNOWN")
        self.assertEqual(len(adapter.audit_log), 1)
        self.assertEqual(adapter.submission_count, 1)

    def test_unknown_query_does_not_create_submission(self):
        adapter = FakeVLAAdapter()
        self.assertEqual(adapter.query("missing")["status"], "UNKNOWN")
        self.assertEqual(adapter.submission_count, 0)

    def test_stop_request_needs_independent_feedback(self):
        adapter = FakeVLAAdapter(); submit(adapter)
        adapter.record_feedback("submission-1", "STARTED", feedback_id="start-1")
        result = adapter.stop("submission-1")
        self.assertTrue(result["stop_requested"])
        self.assertEqual(result["execution_status"], "STARTED")
        self.assertEqual(result["last_feedback_id"], "start-1")
        stopped = adapter.record_feedback("submission-1", "STOPPED", feedback_id="stop-1")
        self.assertEqual(stopped["execution_status"], "STOPPED")
        self.assertEqual(stopped["last_feedback_id"], "stop-1")
        self.assertFalse(stopped["actual_execution"])

    def test_duplicate_stop_has_no_second_stop_request(self):
        adapter = FakeVLAAdapter(); submit(adapter)
        adapter.stop("submission-1"); adapter.stop("submission-1")
        self.assertEqual(sum(item["kind"] == "STOP_REQUEST" for item in adapter.audit_log), 1)

    def test_explicit_start_then_complete(self):
        adapter = FakeVLAAdapter(); submit(adapter)
        with self.assertRaisesRegex(ProtocolError, "completion_without_start"):
            adapter.record_feedback("submission-1", "COMPLETED", feedback_id="premature")
        adapter.record_feedback("submission-1", "STARTED", feedback_id="start")
        result = adapter.record_feedback("submission-1", "COMPLETED", feedback_id="complete")
        self.assertEqual(result["execution_status"], "COMPLETED")
        with self.assertRaisesRegex(ProtocolError, "terminal_execution_state"):
            adapter.record_feedback("submission-1", "STARTED", feedback_id="late")

    def test_feedback_ids_are_idempotent_and_payload_bound(self):
        adapter = FakeVLAAdapter(); submit(adapter)
        adapter.record_feedback("submission-1", "STARTED", feedback_id="start")
        adapter.record_feedback("submission-1", "STARTED", feedback_id="start")
        self.assertEqual(len(adapter.audit_log), 2)
        with self.assertRaisesRegex(ProtocolError, "feedback_id_payload_conflict"):
            adapter.record_feedback("submission-1", "FAILED", feedback_id="start")

    def test_unknown_can_only_resolve_by_explicit_feedback(self):
        adapter = FakeVLAAdapter(submit_status="UNKNOWN"); submit(adapter)
        result = adapter.record_feedback("submission-1", "STARTED", feedback_id="start")
        self.assertEqual(result["status"], "ACCEPTED")
        self.assertEqual(result["execution_status"], "STARTED")
        self.assertFalse(result["actual_execution"])

    def test_rejected_submission_cannot_receive_execution_feedback(self):
        adapter = FakeVLAAdapter(submit_status="REJECTED"); submit(adapter)
        with self.assertRaisesRegex(ProtocolError, "rejected_submission_feedback"):
            adapter.record_feedback("submission-1", "STARTED", feedback_id="start")

    def test_returned_payloads_are_isolated(self):
        adapter = FakeVLAAdapter(); result = submit(adapter)
        result["status"] = "UNKNOWN"
        self.assertEqual(adapter.query("submission-1")["status"], "ACCEPTED")
        with self.assertRaises(TypeError):
            adapter.audit_log[0]["kind"] = "modified"

    def test_dry_run_validates_but_never_submits(self):
        adapter = DryRunVLAAdapter(); result = submit(adapter)
        self.assertEqual(result["status"], "REJECTED")
        self.assertEqual(result["reason"], "dry_run_validation_only")
        self.assertFalse(result["actual_execution"])
        self.assertEqual(adapter.submission_count, 0)
        self.assertEqual(adapter.audit_log, ())
        changed = observation(); changed["state"] = []
        with self.assertRaises(ProtocolError):
            adapter.submit("s", "c", 1, "instruction", changed)

    def test_live_is_explicitly_rejected(self):
        for adapter_class in (FakeVLAAdapter, DryRunVLAAdapter):
            with self.subTest(cls=adapter_class.__name__), self.assertRaisesRegex(ProtocolError, "live_execution_unverified"):
                adapter_class(execution_mode="live")


if __name__ == "__main__":
    unittest.main()
