"""Strict append-only offline journal with replayable, immutable public records.

The SHA256 chain detects accidental discontinuity against retained records. It
does not authenticate a sender, prevent rewriting, or prove a physical event.
Replay is a pure fold supplied by the caller and never contacts a backend.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from uuid import uuid4

from bci_vla.protocols import (
    fields, freeze, integer, json_object, require, text,
    thaw, timestamp,
)


SCHEMA_VERSION = "execution-journal/1.0"
GENESIS_HASH = "0" * 64
_FIELDS = {
    "schema_version", "session_id", "sequence", "event_id", "at", "kind",
    "payload", "previous_hash", "record_hash",
}
_PRIVATE_TRUTH_KEYS = frozenset({
    "private_truth", "ground_truth", "hidden_truth", "latent_truth",
    "oracle_truth", "truth", "true_goal", "true_goal_id", "true_intent",
    "true_intent_id", "true_channel", "true_channel_mode",
    "true_channel_mode_id", "true_mode", "true_mode_id", "oracle_state",
    "hidden_state",
})


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _hash(record):
    body = {key: value for key, value in record.items() if key != "record_hash"}
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


def _public_payload(value):
    require(isinstance(value, dict), "journal_payload_must_be_object")
    def walk(item):
        if isinstance(item, dict):
            require(not (_PRIVATE_TRUTH_KEYS & {key.casefold() for key in item}),
                    "journal_private_truth_forbidden")
            for child in item.values():
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)
    walk(value)


def _at(value):
    if isinstance(value, datetime):
        require(value.tzinfo is not None and value.utcoffset() is not None,
                "timezone_required")
        parsed = value.astimezone(timezone.utc)
    else:
        parsed = timestamp(value)
    return parsed.isoformat()


class EventJournal:
    """A single-session in-memory journal with strict JSONL import/export.

    ``append`` clones the public JSON payload, normalizes ``at`` to UTC and
    returns a recursively immutable record. Failed appends are transactional.
    Empty JSONL imports must supply ``session_id`` because it cannot be inferred.
    ``replay`` passes immutable records to ``reducer(state, record)``; the caller
    owns the state and the reducer's effects. No reducer is chosen automatically.
    """

    def __init__(self, session_id: str):
        text(session_id)
        self.session_id = session_id
        self._records = []

    @property
    def records(self):
        return tuple(self._records)

    def _validated(self):
        text(self.session_id)
        previous_hash = GENESIS_HASH
        previous_at = None
        event_ids = set()
        validated = []
        for sequence, frozen in enumerate(self._records, 1):
            record = json_object(thaw(frozen))
            fields(record, _FIELDS)
            require(record["schema_version"] == SCHEMA_VERSION,
                    "journal_schema_mismatch")
            require(record["session_id"] == self.session_id,
                    "journal_session_mismatch")
            integer(record["sequence"], 1)
            require(record["sequence"] == sequence, "journal_sequence_mismatch")
            text(record["event_id"])
            require(record["event_id"] not in event_ids, "journal_duplicate_event_id")
            text(record["kind"])
            require(isinstance(record["at"], str), "invalid_timestamp")
            canonical_at = _at(record["at"])
            require(record["at"] == canonical_at, "journal_noncanonical_utc_time")
            current_at = timestamp(canonical_at)
            require(previous_at is None or current_at >= previous_at,
                    "journal_time_regression")
            _public_payload(record["payload"])
            require(record["previous_hash"] == previous_hash,
                    "journal_previous_hash_mismatch")
            require(record["record_hash"] == _hash(record),
                    "journal_record_hash_mismatch")
            event_ids.add(record["event_id"])
            previous_hash = record["record_hash"]
            previous_at = current_at
            validated.append(freeze(record))
        return tuple(validated)

    def append(self, kind: str, payload: dict, *, at, event_id: str | None = None):
        """Append one public event; nothing is committed if validation fails."""
        records = self._validated()
        text(kind)
        clean = json_object(payload)
        _public_payload(clean)
        normalized_at = _at(at)
        if records:
            require(timestamp(normalized_at) >= timestamp(records[-1]["at"]),
                    "journal_time_regression")
        if event_id is None:
            event_id = "journal-" + uuid4().hex
        text(event_id)
        require(all(record["event_id"] != event_id for record in records),
                "journal_duplicate_event_id")
        record = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "sequence": len(records) + 1,
            "event_id": event_id,
            "at": normalized_at,
            "kind": kind,
            "payload": clean,
            "previous_hash": records[-1]["record_hash"] if records else GENESIS_HASH,
        }
        record["record_hash"] = _hash(record)
        frozen = freeze(record)
        self._records.append(frozen)
        return frozen

    def to_jsonl(self):
        """Serialize only after rechecking the complete local chain."""
        records = self._validated()
        return "".join(_canonical(thaw(record)) + "\n" for record in records)

    @classmethod
    def from_jsonl(cls, source: str, *, session_id: str | None = None):
        """Import a strict complete chain. No malformed lines are skipped."""
        require(isinstance(source, str), "journal_jsonl_must_be_text")
        lines = source.splitlines()
        raw = []
        for line in lines:
            require(bool(line.strip()), "journal_empty_line")
            record = json_object(line)
            require(isinstance(record, dict), "expected_object")
            fields(record, _FIELDS)
            raw.append(record)
        if session_id is None:
            require(bool(raw), "journal_session_required_for_empty_import")
            session_id = raw[0]["session_id"]
        journal = cls(session_id)
        journal._records = [freeze(record) for record in raw]
        journal._validated()
        return journal

    def replay(self, reducer, initial_state):
        """Validate every record before invoking the supplied local reducer."""
        require(callable(reducer), "journal_reducer_must_be_callable")
        records = self._validated()
        state = initial_state
        for record in records:
            state = reducer(state, record)
        return state
