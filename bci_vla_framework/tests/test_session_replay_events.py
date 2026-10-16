"""Rebuild from public events; coherent checkpoint edits cannot invent authority."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from bci_vla.protocols import ProtocolError, thaw
from bci_vla.monitoring.journal import EventJournal
from bci_vla.contracts.records import digest, EndpointEvidence
from bci_vla.contracts.replay import replay_session_state
from bci_vla.adapters.vla import FakeVLAAdapter

from tests.test_execution_session import (
    make_session, propose, confirm, authorized, completed, response, observation,
    trajectory, read, stamp, BASE,
)


def changed(journal, predicate, mutation, *, remove=False):
    """Rechain edits to test semantic replay beyond a mere digest check."""
    result = EventJournal(journal.session_id)
    found = False
    for record in journal.records:
        payload = thaw(record["payload"])
        if predicate(record):
            found = True
            if remove:
                continue
            mutation(payload)
            payload["state_sha256"] = digest(payload["state"])
        result.append(record["kind"], payload, at=record["at"], event_id=record["event_id"])
    assert found
    return result


def endpoint(s, *, match=True, observed=4.05):
    c = s.submission_contract.data
    p = c["proposal"]
    return EndpointEvidence({
        "evidence_id": "endpoint-evidence-1", "session_id": s.context["session_id"],
        "submission_id": s.submission["submission_id"], "contract_id": p["contract_id"],
        "contract_revision": p["revision"], "scene_id": p["scene_id"], "scene_revision": p["scene_revision"],
        "observed_at": {"host_id": s.config["endpoint_hosts"][0], "at": stamp(observed), "offset_ms": 0, "uncertainty_ms": 0},
        "source": "simulation_fixture_annotation", "trajectory": thaw(trajectory(match=match).data),
    })


class EventDrivenReplayTests(unittest.TestCase):
    def assert_rebuild(self, session):
        restored = EventJournal.from_jsonl(session.journal.to_jsonl())
        with patch.object(FakeVLAAdapter, "__init__", side_effect=AssertionError("no backend construction")), \
             patch.object(FakeVLAAdapter, "submit", side_effect=AssertionError("no dispatch")), \
             patch.object(FakeVLAAdapter, "query", side_effect=AssertionError("no backend query")), \
             patch.object(FakeVLAAdapter, "stop", side_effect=AssertionError("no stop call")):
            self.assertEqual(replay_session_state(restored), session.snapshot())

    def test_initial_and_legal_noisy_confirmation_rebuild_without_backend(self):
        self.assert_rebuild(make_session())
        self.assert_rebuild(authorized(dynamic=True))

    def test_probe_and_task_evidence_are_recomputed(self):
        s = make_session(dynamic=True)
        propose(s)
        q = s.begin_interaction("CHANNEL_PROBE", now=stamp(.1), monotonic_now=.1,
            prompt_symbol="1", presentation_ack=True,
            presented_at={"host_id": "ui", "at": stamp(.1), "offset_ms": 0, "uncertainty_ms": 0})
        s.ingest(response(q, symbol="1"), now=stamp(2.2), monotonic_now=2.2)
        q = s.begin_interaction("TASK_QUERY", now=stamp(3), monotonic_now=3)
        s.ingest(response(q, symbol="0", start=3.2, index=2), now=stamp(4.4), monotonic_now=4.4)
        self.assert_rebuild(s)

    def test_policy_planning_copy_is_recomputed(self):
        s = make_session(dynamic=True)
        s.request_policy(remaining_s=10, scene_valid_until_s=BASE.timestamp()+60, now=stamp(.5))
        self.assert_rebuild(s)
        altered = changed(s.journal, lambda r: r["kind"] == "POLICY_RETURNED",
            lambda p: p["event"]["decision"].update(execution_allowed=True))
        with self.assertRaisesRegex(ProtocolError, "replay_policy_decision_mismatch"):
            replay_session_state(altered)

    def test_filter_checkpoint_alteration_fails_even_after_all_hashes_rebuilt(self):
        s = make_session(dynamic=True)
        confirm(s)
        altered = changed(s.journal, lambda r: r["kind"] == "INPUT_RECEIVED",
            lambda p: p["state"]["filter"].update(reason="invented"))
        with self.assertRaisesRegex(ProtocolError, "session_replay_checkpoint_mismatch"):
            replay_session_state(altered)

    def test_coherent_fake_filter_update_is_not_trusted(self):
        s = make_session()
        confirm(s)
        def modify(payload):
            payload["event"]["filter_update"]["snapshot"]["goal_marginals"][s.contract.data["goal_id"]] = .2
            payload["state"]["filter"] = deepcopy(payload["event"]["filter_update"]["snapshot"])
        altered = changed(s.journal, lambda r: r["kind"] == "INPUT_RECEIVED", modify)
        with self.assertRaisesRegex(ProtocolError, "replay_filter_update_mismatch"):
            replay_session_state(altered)

    def test_authorization_cannot_be_invented_by_removing_confirmation(self):
        s = authorized()
        altered = changed(s.journal, lambda r: r["kind"] == "INPUT_RECEIVED", lambda _: None, remove=True)
        with self.assertRaisesRegex(ProtocolError, "actual_confirmation_not_pending_authorization"):
            replay_session_state(altered)

    def test_dispatch_requires_valid_unconsumed_authority(self):
        s = authorized()
        s.dispatch("submission-1", observation(), now=stamp(2.4))
        altered = changed(s.journal, lambda r: r["kind"] == "AUTHORIZATION_GRANTED", lambda _: None, remove=True)
        with self.assertRaisesRegex(ProtocolError, "valid_unused_authorization_required"):
            replay_session_state(altered)
        self.assert_rebuild(s)

    def test_complete_flow_and_endpoint_recomputed(self):
        for match in (True, False, None):
            with self.subTest(match=match):
                s = completed()
                s.verify_endpoint(endpoint(s, match=match), now=stamp(4.1))
                self.assert_rebuild(s)
        s = completed()
        s.verify_endpoint(endpoint(s), now=stamp(4.1))
        altered = changed(s.journal, lambda r: r["kind"] == "ENDPOINT_JUDGED",
            lambda p: p["event"]["result"].update(endpoint_matches_goal=False))
        with self.assertRaisesRegex(ProtocolError, "replay_endpoint_result_mismatch"):
            replay_session_state(altered)

    def test_stop_latch_and_actual_control_replay(self):
        s = make_session()
        v = read("examples/protocols/stop.json")
        v.update(s.context)
        v.update(task_revision=0, scene_revision=0, sent_at=stamp(-5), expires_at=stamp(-4))
        s.ingest(v, now=stamp(.1), monotonic_now=.1)
        self.assert_rebuild(s)
        altered = changed(s.journal, lambda r: r["kind"] == "CONTROL_RECEIVED",
            lambda p: p["state"].update(stop_latched=False))
        with self.assertRaisesRegex(ProtocolError, "session_replay_checkpoint_mismatch"):
            replay_session_state(altered)

    def test_timeout_revision_and_model_invalidation_replay(self):
        s = make_session()
        propose(s)
        s.begin_interaction("CONFIRM", now=stamp(.1), monotonic_now=.1, expires_in_s=1)
        s.expire(now=stamp(1.1), monotonic_now=1.1)
        self.assert_rebuild(s)
        s = authorized()
        s.revise_context(dict(s.context, task_revision=2), s.scene, now=stamp(2.4))
        self.assert_rebuild(s)
        s = authorized()
        s.invalidate_models(now=stamp(2.4))
        self.assert_rebuild(s)

    def test_rejected_input_has_no_public_state_effect(self):
        s = authorized()
        r = thaw(s.authorization.data["response"])
        s.ingest(r, now=stamp(2.4), monotonic_now=2.4)
        self.assert_rebuild(s)
        altered = changed(s.journal, lambda r: r["kind"] == "INPUT_REJECTED",
            lambda p: p["state"].update(state="SUCCEEDED"))
        with self.assertRaisesRegex(ProtocolError, "session_replay_checkpoint_mismatch"):
            replay_session_state(altered)

    def test_unknown_submission_and_query_failure_do_not_repeat_backend_calls(self):
        s = authorized(submit_status="UNKNOWN")
        s.dispatch("submission-1", observation(), now=stamp(2.4))
        s.refresh_submission(now=stamp(2.5))
        self.assert_rebuild(s)
        def fail(*args):
            raise ConnectionError("fixture lost result")
        s.backend.query = fail
        s.refresh_submission(now=stamp(2.6))
        self.assert_rebuild(s)

    def test_semantically_impossible_checkpoint_rejected(self):
        s = make_session()
        altered = changed(s.journal, lambda r: True, lambda p: p["state"].update(state="AUTHORIZED"))
        with self.assertRaisesRegex(ProtocolError, "session_replay_checkpoint_mismatch"):
            replay_session_state(altered)

    def test_completed_feedback_requires_prior_started_event(self):
        s = completed()
        altered = changed(s.journal, lambda r: r["kind"] == "SIMULATED_EXECUTION_FEEDBACK" and
            r["payload"]["event"]["execution_status"] == "STARTED", lambda _: None, remove=True)
        with self.assertRaisesRegex(ProtocolError, "completion_without_start"):
            replay_session_state(altered)

    def test_stop_then_contract_expiry_and_active_expiry_are_distinct(self):
        s = make_session()
        propose(s)
        s.request_stop(now=stamp(.1))
        s.expire(now=stamp(60), monotonic_now=60)
        self.assert_rebuild(s)
        s = authorized()
        s.dispatch("submission-1", observation(), now=stamp(2.4))
        s.expire(now=stamp(60), monotonic_now=60)
        self.assert_rebuild(s)


if __name__ == "__main__":
    unittest.main()
