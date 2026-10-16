"""Public event continuity, strict JSON and offline fold regression checks."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import unittest

from bci_vla.monitoring.journal import EventJournal, GENESIS_HASH, SCHEMA_VERSION
from bci_vla.protocols import ProtocolError, thaw


BASE = datetime(2026, 10, 16, tzinfo=timezone.utc)


def fixture():
    journal = EventJournal("session-11")
    journal.append("STATE", {"state": "PROPOSED", "nested": [{"x": 1}]},
                   at=BASE, event_id="first")
    journal.append("STATE", {"state": "CONFIRMED"},
                   at=BASE + timedelta(seconds=1), event_id="second")
    return journal


def mutate_lines(journal, mutate, *, rehash=False):
    records = [thaw(record) for record in journal.records]
    mutate(records)
    if rehash:
        previous = GENESIS_HASH
        for record in records:
            record["previous_hash"] = previous
            body = {key: value for key, value in record.items() if key != "record_hash"}
            record["record_hash"] = hashlib.sha256(json.dumps(
                body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                allow_nan=False).encode("utf-8")).hexdigest()
            previous = record["record_hash"]
    return "\n".join(json.dumps(record) for record in records)


class EventJournalTests(unittest.TestCase):
    def test_append_clones_and_recursively_freezes(self):
        journal = EventJournal("s")
        payload = {"items": [{"a": 1}]}
        record = journal.append("PUBLIC", payload, at=BASE, event_id="e")
        payload["items"][0]["a"] = 7
        self.assertEqual(record["payload"]["items"][0]["a"], 1)
        with self.assertRaises(TypeError):
            record["payload"]["items"][0]["a"] = 2
        self.assertIsInstance(journal.records, tuple)
        self.assertEqual(record["previous_hash"], GENESIS_HASH)
        self.assertEqual(record["sequence"], 1)
        self.assertEqual(record["schema_version"], SCHEMA_VERSION)

    def test_roundtrip_and_exact_replay(self):
        journal = fixture()
        restored = EventJournal.from_jsonl(journal.to_jsonl())
        self.assertEqual(restored.to_jsonl(), journal.to_jsonl())
        states = restored.replay(lambda state, row: state + [row["payload"]["state"]], [])
        self.assertEqual(states, ["PROPOSED", "CONFIRMED"])

    def test_empty_import_needs_session_and_replays_initial(self):
        with self.assertRaisesRegex(ProtocolError, "journal_session_required"):
            EventJournal.from_jsonl("")
        journal = EventJournal.from_jsonl("", session_id="s")
        initial = object()
        self.assertIs(journal.replay(lambda *_: self.fail("unexpected reducer"), initial), initial)
        self.assertEqual(journal.to_jsonl(), "")

    def test_time_normalizes_offsets_and_accepts_equal_timestamps(self):
        journal = EventJournal("s")
        first = journal.append("A", {}, at="2026-10-16T08:00:00+08:00")
        second = journal.append("B", {}, at=BASE)
        self.assertEqual(first["at"], "2026-10-16T00:00:00+00:00")
        self.assertEqual(first["at"], second["at"])
        self.assertNotEqual(first["event_id"], second["event_id"])

    def test_bad_appends_leave_journal_unchanged(self):
        journal = fixture()
        before = journal.to_jsonl()
        cases = [
            {"at": BASE, "event_id": "third"},
            {"at": BASE + timedelta(seconds=2), "event_id": "first"},
            {"at": datetime(2026, 10, 16), "event_id": "third"},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ProtocolError):
                journal.append("STATE", {}, **kwargs)
            self.assertEqual(journal.to_jsonl(), before)

    def test_strict_json_and_private_truth_rejected_recursively(self):
        journal = EventJournal("s")
        for payload in ({"x": float("nan")}, {1: "non-string"}, {"x": object()},
                        {"nested": [{"ground_truth": "private"}]},
                        {"nested": {"TRUE_GOAL_ID": "private"}}, [1]):
            with self.subTest(payload=payload), self.assertRaises(ProtocolError):
                journal.append("A", payload, at=BASE)
        self.assertEqual(journal.records, ())

    def test_duplicate_json_key_and_nan_rejected_at_import(self):
        source = fixture().to_jsonl()
        duplicated = source.replace('"kind":"STATE"', '"kind":"STATE","kind":"OTHER"', 1)
        with self.assertRaisesRegex(ProtocolError, "duplicate_json_key"):
            EventJournal.from_jsonl(duplicated)
        nan_source = source.replace('"x":1', '"x":NaN', 1)
        with self.assertRaises(ProtocolError):
            EventJournal.from_jsonl(nan_source)

    def test_truncation_middle_deletion_and_reordering_detected(self):
        journal = fixture()
        journal.append("FINISHED", {}, at=BASE + timedelta(seconds=2), event_id="third")
        for mutate in (lambda rows: rows.pop(0), lambda rows: rows.pop(1),
                       lambda rows: rows.reverse()):
            with self.subTest(mutate=mutate), self.assertRaises(ProtocolError):
                EventJournal.from_jsonl(mutate_lines(journal, mutate))
        # Tail truncation is a valid shorter chain without an external anchor.
        shorter = EventJournal.from_jsonl(journal.to_jsonl().splitlines()[0])
        self.assertEqual(len(shorter.records), 1)

    def test_corrupt_payload_or_previous_hash_detected(self):
        journal = fixture()
        for mutate in (
            lambda rows: rows[0]["payload"].update(state="ALTERED"),
            lambda rows: rows[1].update(previous_hash="1" * 64),
        ):
            with self.subTest(mutate=mutate), self.assertRaises(ProtocolError):
                EventJournal.from_jsonl(mutate_lines(journal, mutate))

    def test_rehashed_invalid_sequence_session_id_time_and_private_truth_rejected(self):
        journal = fixture()
        cases = [
            lambda rows: rows[1].update(sequence=4),
            lambda rows: rows[1].update(session_id="another-session"),
            lambda rows: rows[1].update(event_id="first"),
            lambda rows: rows[1].update(at=(BASE - timedelta(seconds=1)).isoformat()),
            lambda rows: rows[0]["payload"].update(private_truth={"z": 1}),
            lambda rows: rows[0].update(extra="unsupported"),
            lambda rows: rows[0].update(schema_version="unknown/2"),
            lambda rows: rows[0].update(at="2026-10-16T08:00:00+08:00"),
        ]
        for mutate in cases:
            with self.subTest(mutate=mutate), self.assertRaises(ProtocolError):
                EventJournal.from_jsonl(mutate_lines(journal, mutate, rehash=True))

    def test_explicit_import_session_must_match(self):
        with self.assertRaisesRegex(ProtocolError, "journal_session_mismatch"):
            EventJournal.from_jsonl(fixture().to_jsonl(), session_id="other")

    def test_validation_completes_before_any_reducer_call(self):
        journal = fixture()
        # Deliberately damage private storage to verify export/replay checks.
        damaged = thaw(journal._records[-1])
        damaged["payload"]["state"] = "CHANGED"
        journal._records[-1] = damaged
        calls = []
        with self.assertRaises(ProtocolError):
            journal.replay(lambda state, row: calls.append(row), None)
        self.assertEqual(calls, [])
        with self.assertRaises(ProtocolError):
            journal.to_jsonl()

    def test_malformed_jsonl_and_non_callable_reducer_rejected(self):
        for source in ("\n", "{}\n\n", "[]", "{", 12):
            with self.subTest(source=source), self.assertRaises(ProtocolError):
                EventJournal.from_jsonl(source, session_id="s")
        with self.assertRaisesRegex(ProtocolError, "journal_reducer_must_be_callable"):
            EventJournal("s").replay(None, {})


if __name__ == "__main__":
    unittest.main()
