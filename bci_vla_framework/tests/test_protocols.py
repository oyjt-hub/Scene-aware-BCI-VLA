"""Source provenance, stale-input and evidence accounting regressions."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from bci_vla import protocols as p
from bci_vla.adapters.bci.replay import ReplayBCIAdapter, CONTEXT_FIELDS
from bci_vla.adapters.bci.ssvep import SSVEPAdapterShell

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples/protocols"
BASE = datetime(2026, 10, 15, tzinfo=timezone.utc)


def example(name):
    return json.loads((EXAMPLES / (name + ".json")).read_text(encoding="utf-8"))


def adapter(cls=ReplayBCIAdapter):
    q = example("task_query_request")
    return cls({k: q[k] for k in CONTEXT_FIELDS}, response_hosts={"bci"}, control_hosts={"ui"})


def register(a, name="task_query_request", **changes):
    q = example(name); q.update(changes)
    a.register_request(q, now=BASE, monotonic_now=0)
    return q


def ingest(a, value, seconds=3, mono=None):
    return a.ingest(value, now=BASE + timedelta(seconds=seconds), monotonic_now=seconds if mono is None else mono)


class SchemaTests(unittest.TestCase):
    def test_all_modes_and_record_types_have_legal_examples(self):
        for name in ["task_query_request", "task_query_response", "channel_probe_request", "channel_probe_response", "confirm_request", "confirm_response", "stop", "cancel", "execution_event"]:
            with self.subTest(name=name):
                self.assertEqual(p.Event.parse(example(name)).to_dict(), example(name))
        records = example("public_records")
        self.assertEqual(len(records), 7)
        for name, data in records.items():
            with self.subTest(record=name):
                self.assertEqual(getattr(p, name)(data).to_dict(), data)

    def test_invalid_examples_have_specific_failures(self):
        for case in example("invalid_cases"):
            with self.subTest(case=case["name"]):
                with self.assertRaisesRegex(p.ProtocolError, "^" + case["expected_error"] + "$"):
                    p.Event.parse(case["event"])

    def test_duplicate_json_keys_nonfinite_and_boolean_numbers_rejected(self):
        with self.assertRaisesRegex(p.ProtocolError, "duplicate_json_key"):
            p.Event.parse('{"schema_version":"bci-vla/1.0","schema_version":"x"}')
        for value in [float("nan"), float("inf"), True, 10 ** 400]:
            q = example("task_query_response"); q["payload"]["raw_scores"]["0"] = value
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(p.ProtocolError): p.Event.parse(q)
        q = example("task_query_request"); q["task_revision"] = True
        with self.assertRaisesRegex(p.ProtocolError, "invalid_integer"): p.Event.parse(q)

    def test_nested_private_annotations_do_not_pass_schema(self):
        for field in ["true_goal", "annotation", "factor_cell", "future_schedule"]:
            q = example("task_query_request"); q["payload"]["options"][0][field] = "secret label"
            with self.subTest(field=field):
                with self.assertRaisesRegex(p.ProtocolError, "unsupported_field"): p.Event.parse(q)

    def test_parsed_event_and_domain_record_are_deeply_immutable(self):
        q = example("channel_probe_request"); event = p.Event.parse(q)
        q["payload"]["probe_prompt"]["symbol"] = "1"
        self.assertEqual(event.data["payload"]["probe_prompt"]["symbol"], "0")
        with self.assertRaises(TypeError): event.data["payload"]["options"][0]["symbol"] = "x"
        copy = event.to_dict(); copy["payload"]["probe_prompt"]["symbol"] = "1"
        self.assertEqual(event.data["payload"]["probe_prompt"]["symbol"], "0")
        record = p.JointBelief(example("public_records")["JointBelief"])
        self.assertEqual(len(record.data["mass"]), 2)
        with self.assertRaises(TypeError): record.data["mass"][0]["probability"] = 1

    def test_domain_records_preserve_joint_mass_and_require_evidence(self):
        records = example("public_records")
        belief = deepcopy(records["JointBelief"]); belief["mass"].append(deepcopy(belief["mass"][0]))
        with self.assertRaisesRegex(p.ProtocolError, "duplicate_state"): p.JointBelief(belief)
        belief = deepcopy(records["JointBelief"]); belief["mass"][0]["probability"] = 0.6
        with self.assertRaisesRegex(p.ProtocolError, "mass_not_normalized"): p.JointBelief(belief)
        profile = deepcopy(records["ChannelProfile"]); profile["status"] = "calibrated"
        with self.assertRaisesRegex(p.ProtocolError, "missing_calibration"): p.ChannelProfile(profile)
        probe = deepcopy(records["ProbeModel"]); probe["status"] = "calibrated"
        with self.assertRaisesRegex(p.ProtocolError, "missing_calibration"): p.ProbeModel(probe)
        contract = deepcopy(records["TaskContract"]); contract["status"] = "confirmed"
        with self.assertRaisesRegex(p.ProtocolError, "missing_confirmation"): p.TaskContract(contract)
        end = deepcopy(records["ExecutionEvent"]); end["stage"] = "endpoint_judged"
        with self.assertRaisesRegex(p.ProtocolError, "missing_endpoint_evidence"): p.ExecutionEvent(end)

    def test_timestamps_require_zones_and_finite_bounded_offsets(self):
        for field, value in [("sent_at", "2026-10-15T00:00:00"), ("clock_offset_ms", 10 ** 10), ("clock_uncertainty_ms", -1)]:
            q = example("task_query_request"); q[field] = value
            with self.subTest(field=field):
                with self.assertRaises(p.ProtocolError): p.Event.parse(q)


class ReplayTests(unittest.TestCase):
    def test_request_response_routes_three_likelihood_kinds_without_task_mutation(self):
        for mode, kind in [("task_query", "TASK"), ("channel_probe", "PROBE"), ("confirm", "CONFIRM")]:
            a = adapter(SSVEPAdapterShell); original = a.current_context
            a.register_request(p.Event.parse(example(mode + "_request")), now=BASE, monotonic_now=0)
            result = ingest(a, p.Event.parse(example(mode + "_response")))
            self.assertTrue(result.accepted, result.reason)
            self.assertEqual(result.likelihood_kind, kind)
            self.assertEqual(result.probe_prompt_symbol, "0" if kind == "PROBE" else None)
            self.assertEqual(result.response.data["payload"]["symbol"], "1")
            self.assertEqual(a.current_context, original)
            self.assertIsNone(a.pending_request)
            if kind == "CONFIRM": self.assertEqual(result.request.data["payload"]["contract"]["status"], "proposed")

    def test_raw_fbcca_scores_preserved_and_never_implicitly_normalized(self):
        a = adapter(); register(a)
        result = ingest(a, example("task_query_response"))
        self.assertEqual(dict(result.response.data["payload"]["raw_scores"]), {"0": 0.5, "1": 5.0})
        self.assertEqual(result.response.data["payload"]["score_kind"], "FBCCA_POWER")

    def test_nonresponses_stay_distinct_and_never_become_likelihood(self):
        for status in p.STATUSES - {"OBSERVATION"}:
            a = adapter(); register(a)
            r = example("task_query_response")
            r["payload"].update(status=status, symbol=None, raw_scores=None, score_kind=None, calibration_id=None, sample_window=None)
            result = ingest(a, r)
            self.assertTrue(result.accepted, result.reason)
            self.assertEqual(result.reason, status.lower())
            self.assertIsNone(result.likelihood_kind)

    def test_no_unsolicited_probe_or_mode_spoofing(self):
        a = adapter()
        self.assertEqual(ingest(a, example("channel_probe_response")).reason, "no_pending_interaction")
        a = adapter()
        register(a, "channel_probe_request")
        r = example("task_query_response"); r["response_to"] = "q-CHANNEL_PROBE"
        self.assertEqual(ingest(a, r).reason, "response_binding_mismatch")
        self.assertIsNotNone(a.pending_request)

    def test_unacknowledged_probe_cannot_supply_channel_evidence(self):
        a = adapter(); q = example("channel_probe_request"); q["payload"]["probe_prompt"].update(presentation_ack=False, presented_at=None)
        a.register_request(q, now=BASE, monotonic_now=0)
        self.assertEqual(ingest(a, example("channel_probe_response")).reason, "probe_not_presented")

    def test_stale_context_wrong_session_and_unknown_symbols_are_quarantined(self):
        for field, value, reason in [("mapping_id", "old-map", "stale_context"), ("task_revision", 0, "stale_context"), ("scene_revision", 0, "stale_context"), ("stimulus_version", "old-ui", "stale_context"), ("session_id", "other", "wrong_session"), ("response_to", "old-query", "response_binding_mismatch")]:
            a = adapter(); register(a); r = example("task_query_response"); r[field] = value
            with self.subTest(field=field):
                result = ingest(a, r); self.assertFalse(result.accepted); self.assertEqual(result.reason, reason)
                self.assertIsNotNone(a.pending_request)
        a = adapter(); register(a); r = example("task_query_response"); r["payload"]["symbol"] = "99"
        self.assertEqual(ingest(a, r).reason, "unknown_response_symbol")

    def test_rejected_message_does_not_consume_valid_response(self):
        a = adapter(); register(a); r = example("task_query_response"); r["mapping_id"] = "stale"
        self.assertFalse(ingest(a, r).accepted)
        self.assertTrue(ingest(a, example("task_query_response")).accepted)

    def test_duplicate_and_out_of_order_messages_not_counted_twice(self):
        a = adapter(); register(a); r = example("task_query_response")
        self.assertTrue(ingest(a, r).accepted)
        self.assertEqual(ingest(a, r).reason, "duplicate_event")
        r["event_id"] = "different-id-same-sequence"
        self.assertEqual(ingest(a, r).reason, "out_of_order")

    def test_new_context_invalidates_pending_and_cannot_regress(self):
        a = adapter(); register(a, "channel_probe_request")
        new = a.current_context; new["task_revision"] = 2
        a.update_context(new)
        self.assertIsNone(a.pending_request)
        self.assertEqual(ingest(a, example("channel_probe_response")).reason, "stale_context")
        new["task_revision"] = 1
        with self.assertRaisesRegex(p.ProtocolError, "revision_regression"): a.update_context(new)

    def test_overlapping_recording_samples_rejected_even_with_new_window_id(self):
        a = adapter(); register(a); self.assertTrue(ingest(a, example("task_query_response")).accepted)
        q = example("task_query_request"); q.update(event_id="q2", sequence=1, interaction_id="i2", query_id="query2", sent_at="2026-10-15T00:00:03Z", expires_at="2026-10-15T00:00:13Z")
        a.register_request(q, now=BASE + timedelta(seconds=3), monotonic_now=3)
        r = example("task_query_response"); r.update(event_id="r2", sequence=1, interaction_id="i2", query_id="query2", response_to="q2", sent_at="2026-10-15T00:00:06Z", expires_at="2026-10-15T00:00:13Z")
        r["payload"]["sample_window"].update(window_id="new-id", sample_start=375, sample_end=625, started_at="2026-10-15T00:00:04Z", ended_at="2026-10-15T00:00:05Z")
        self.assertEqual(ingest(a, r, seconds=6).reason, "overlapping_evidence")
        r["payload"]["sample_window"].update(sample_start=1000, sample_end=1250)
        self.assertTrue(ingest(a, r, seconds=6).accepted)

    def test_wall_clock_and_monotonic_deadlines_are_both_enforced(self):
        a = adapter(); register(a)
        self.assertEqual(ingest(a, example("task_query_response"), seconds=3, mono=11).reason, "local_deadline_expired")
        a = adapter(); register(a)
        self.assertEqual(ingest(a, example("task_query_response"), seconds=11).reason, "expired_or_ambiguous_time")
        a = adapter(); register(a)
        self.assertEqual(a.expire_pending(monotonic_now=10).reason, "timeout")
        self.assertIsNone(a.pending_request)

    def test_remote_monotonic_clocks_never_compared(self):
        a = adapter(); register(a); r = example("task_query_response")
        for key in ["sent_at", "expires_at"]: r[key] = (p.timestamp(r[key]) + timedelta(seconds=20)).isoformat()
        for key in ["started_at", "ended_at"]: r["payload"]["sample_window"][key] = (p.timestamp(r["payload"]["sample_window"][key]) + timedelta(seconds=20)).isoformat()
        r["clock_offset_ms"] = 20000
        self.assertTrue(ingest(a, r).accepted)
        a = adapter(); register(a); r["clock_uncertainty_ms"] = 101
        self.assertEqual(ingest(a, r).reason, "clock_uncertain")

    def test_stop_preempts_stale_context_order_expiry_and_latches(self):
        a = adapter(); register(a, "channel_probe_request")
        stop = example("stop"); stop.update(task_revision=0, sequence=0)
        result = ingest(a, stop, seconds=30)
        self.assertTrue(result.accepted); self.assertEqual(result.reason, "stop_requested")
        self.assertIsNone(a.pending_request)
        self.assertEqual(ingest(a, example("channel_probe_response"), seconds=30).reason, "session_stopped")

    def test_cancel_only_current_interaction_and_unknown_hosts_cannot_stop(self):
        a = adapter(); register(a); cancel = example("cancel"); cancel["interaction_id"] = "old"
        self.assertEqual(ingest(a, cancel).reason, "cancel_not_current")
        self.assertIsNotNone(a.pending_request)
        cancel["interaction_id"] = "i-TASK_QUERY"
        self.assertEqual(ingest(a, cancel).reason, "cancelled")
        stop = example("stop"); stop["host_id"] = "untrusted"
        self.assertEqual(ingest(a, stop).reason, "unexpected_control_host")

    def test_unknown_calibration_cannot_self_certify_probabilities(self):
        a = adapter(); register(a); r = example("task_query_response")
        r["payload"].update(raw_scores={"0": 0.1, "1": 0.9}, score_kind="CALIBRATED_PROBABILITY", calibration_id="invented")
        self.assertEqual(ingest(a, r).reason, "unregistered_calibration")

    def test_registered_calibration_and_score_mapping_are_explicit(self):
        q = example("task_query_request")
        a = ReplayBCIAdapter({k: q[k] for k in CONTEXT_FIELDS}, response_hosts={"bci"}, control_hosts={"ui"}, known_calibration_ids={"fixture-calibration"})
        register(a); r = example("task_query_response")
        r["payload"].update(raw_scores={"0": 0.1, "1": 0.9}, score_kind="CALIBRATED_PROBABILITY", calibration_id="fixture-calibration")
        r["payload"]["raw_scores"] = {"0": 0.1, "unknown": 0.9}
        self.assertEqual(ingest(a, r).reason, "score_mapping_mismatch")
        r["payload"]["raw_scores"] = {"0": 0.1, "1": 0.9}
        self.assertTrue(ingest(a, r).accepted)

    def test_window_must_belong_to_query_and_reported_samplerate(self):
        a = adapter(); register(a); r = example("task_query_response")
        r["payload"]["sample_window"].update(started_at="2026-10-14T23:59:59Z", ended_at="2026-10-15T00:00:00Z")
        self.assertEqual(ingest(a, r).reason, "window_outside_interaction")
        r = example("task_query_response"); r["payload"]["sample_window"]["sample_end"] = 10 ** 300
        self.assertEqual(ingest(a, r).reason, "invalid_integer")
        r = example("task_query_response"); r["clock_offset_ms"] = 1
        r["payload"]["sample_window"].update(started_at="0001-01-01T00:00:00Z", ended_at="0001-01-01T00:00:01Z")
        self.assertEqual(ingest(a, r).reason, "invalid_clock_range")

    def test_window_id_cannot_be_reused_with_disjoint_indices(self):
        a = adapter(); register(a); self.assertTrue(ingest(a, example("task_query_response")).accepted)
        q = example("task_query_request"); q.update(event_id="q2", sequence=1, interaction_id="i2", query_id="query2", sent_at="2026-10-15T00:00:03Z", expires_at="2026-10-15T00:00:13Z")
        a.register_request(q, now=BASE + timedelta(seconds=3), monotonic_now=3)
        r = example("task_query_response"); r.update(event_id="r2", sequence=1, interaction_id="i2", query_id="query2", response_to="q2", sent_at="2026-10-15T00:00:06Z", expires_at="2026-10-15T00:00:13Z")
        r["payload"]["sample_window"].update(sample_start=1000, sample_end=1250, started_at="2026-10-15T00:00:04Z", ended_at="2026-10-15T00:00:05Z")
        self.assertEqual(ingest(a, r, seconds=6).reason, "window_reused")

    def test_local_clock_cannot_reverse_and_pending_cannot_be_overwritten(self):
        a = adapter(); register(a)
        with self.assertRaisesRegex(p.ProtocolError, "interaction_pending"):
            a.register_request(example("channel_probe_request"), now=BASE, monotonic_now=0)
        ingest(a, example("task_query_response"))
        with self.assertRaisesRegex(p.ProtocolError, "local_clock_reversed"): a.expire_pending(monotonic_now=1)

    def test_confirm_contract_expiry_is_not_a_free_success_label(self):
        a = adapter(); q = example("confirm_request"); q["payload"]["contract"]["valid_until"] = "2026-10-15T00:00:02Z"
        a.register_request(q, now=BASE, monotonic_now=0)
        result = ingest(a, example("confirm_response"))
        self.assertEqual(result.reason, "local_deadline_expired")


class ImportTests(unittest.TestCase):
    def test_fresh_import_and_help_have_no_network_or_device_import(self):
        script = '''
import builtins, socket, sys, runpy
original = builtins.__import__
blocked = {"mne", "pygame", "piper_sdk", "can", "cv2", "torch", "jax", "requests", "paramiko"}
def guarded(name, *a, **kw):
    if name.split(".")[0] in blocked: raise AssertionError("device or network SDK import")
    return original(name, *a, **kw)
builtins.__import__ = guarded
socket.socket = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("network forbidden"))
import bci_vla.protocols
import bci_vla.adapters.bci.ssvep
sys.argv = ["bci_vla", "--help"]
runpy.run_module("bci_vla", run_name="__main__")
'''
        result = subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("offline", result.stdout)


if __name__ == "__main__":
    unittest.main()
