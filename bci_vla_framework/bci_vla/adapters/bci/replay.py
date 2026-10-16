"""One pending interaction, disjoint-window replay. Never dispatches an action."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from types import MappingProxyType

from ...protocols import Event, ProtocolError, event_clock, interval, number, require, text, integer, timestamp

CONTEXT_FIELDS = {"session_id", "task_id", "task_revision", "scene_id", "scene_revision", "mapping_id", "stimulus_version", "channel_profile_id"}
BIND_FIELDS = CONTEXT_FIELDS | {"mode", "interaction_id", "query_id", "probe_id"}


@dataclass(frozen=True)
class IngestResult:
    accepted: bool
    reason: str
    response: Event | None = None
    request: Event | None = None
    likelihood_kind: str | None = None
    probe_prompt_symbol: str | None = None


def context(value):
    require(isinstance(value, dict) and set(value) == CONTEXT_FIELDS, "invalid_context")
    for k, v in value.items():
        integer(v) if k.endswith("revision") else text(v)
    return MappingProxyType(dict(value))


def reference_now(now):
    require(isinstance(now, datetime) and now.tzinfo is not None and now.utcoffset() is not None, "timezone_required")
    return now.astimezone(timezone.utc)


class ReplayBCIAdapter:
    """Public-data validator, not a signal decoder, belief filter or authenticator.

    Explicit timestamps and a local monotonic clock make replay deterministic.
    STOP/CANCEL yield control notifications; physical stopping is owned elsewhere.
    """

    def __init__(self, current_context, *, response_hosts, control_hosts, max_clock_uncertainty_ms=100, known_calibration_ids=()):
        self._context = context(current_context)
        require(bool(response_hosts) and bool(control_hosts), "missing_hosts")
        self._response_hosts = frozenset(response_hosts)
        self._control_hosts = frozenset(control_hosts)
        for host in self._response_hosts | self._control_hosts: text(host)
        number(max_clock_uncertainty_ms, 0, 86400000)
        self._max_uncertainty = max_clock_uncertainty_ms
        self._calibration_ids = frozenset(known_calibration_ids)
        for identifier in self._calibration_ids: text(identifier)
        self._pending = None
        self._deadline = None
        self._seen = set()
        self._last_sequence = {}
        self._windows = []
        self._window_ids = set()
        self._interactions = set()
        self._last_local_mono = None
        self._stopped = False

    @property
    def current_context(self):
        return dict(self._context)

    @property
    def pending_request(self):
        return self._pending

    def _local_clock(self, monotonic_now):
        number(monotonic_now, 0)
        require(self._last_local_mono is None or monotonic_now >= self._last_local_mono, "local_clock_reversed")
        self._last_local_mono = monotonic_now

    def update_context(self, new_context):
        fresh = context(new_context)
        require(fresh["session_id"] == self._context["session_id"], "new_session_requires_new_adapter")
        require(fresh["task_revision"] >= self._context["task_revision"] and fresh["scene_revision"] >= self._context["scene_revision"], "revision_regression")
        require(fresh != self._context, "unchanged_context")
        require(fresh["task_id"] == self._context["task_id"] or fresh["task_revision"] > self._context["task_revision"], "task_change_requires_revision")
        require(fresh["scene_id"] == self._context["scene_id"] or fresh["scene_revision"] > self._context["scene_revision"], "scene_change_requires_revision")
        self._context = fresh
        self._pending = None
        self._deadline = None
        # Preserve replay/overlap history; STOP remains latched for this session.

    def _matches_context(self, v):
        require(all(v[k] == self._context[k] for k in CONTEXT_FIELDS), "stale_context")

    def _fresh(self, v):
        require(v["event_id"] not in self._seen, "duplicate_event")
        require(v["sequence"] > self._last_sequence.get(v["host_id"], -1), "out_of_order")

    def _remember(self, v):
        self._seen.add(v["event_id"])
        self._last_sequence[v["host_id"]] = v["sequence"]

    def _wallclock(self, v, now):
        require(v["clock_uncertainty_ms"] <= self._max_uncertainty, "clock_uncertain")
        sent_low, sent_high = event_clock(v)
        expiry_low, _ = event_clock(v, "expires_at")
        require(sent_high <= now, "future_or_ambiguous_time")
        require(now < expiry_low, "expired_or_ambiguous_time")

    def register_request(self, value, *, now, monotonic_now):
        """Explicitly register a trusted UI request; does not send or draw stimuli."""
        event = Event.parse(value); v = event.data
        now = reference_now(now); self._local_clock(monotonic_now)
        require(not self._stopped, "session_stopped")
        require(self._pending is None, "interaction_pending")
        require(v["kind"] == "REQUEST", "not_request")
        require(v["host_id"] in self._control_hosts, "unexpected_request_host")
        self._matches_context(v); self._fresh(v); self._wallclock(v, now)
        require(v["interaction_id"] not in self._interactions, "interaction_id_reused")
        if v["mode"] == "CHANNEL_PROBE":
            prompt = v["payload"]["probe_prompt"]
            if prompt["presented_at"] is not None:
                require(prompt["presented_at"]["uncertainty_ms"] <= self._max_uncertainty, "clock_uncertain")
                require(interval(dict(prompt["presented_at"]))[1] <= now, "future_presentation")
        if v["mode"] == "CONFIRM":
            # Contract timestamp is expressed in request-host wall-clock domain.
            expiry = interval({"host_id": v["host_id"], "at": v["payload"]["contract"]["valid_until"], "offset_ms": v["clock_offset_ms"], "uncertainty_ms": v["clock_uncertainty_ms"]})[0]
            require(now < expiry, "contract_expired")
        else:
            expiry = event_clock(v, "expires_at")[0]
        self._pending = event
        lifetime = min(expiry, event_clock(v, "expires_at")[0]) - now
        self._deadline = monotonic_now + lifetime.total_seconds()
        self._interactions.add(v["interaction_id"])
        self._remember(v)
        return event

    def ingest(self, value, *, now, monotonic_now):
        """Quarantine invalid messages by reason code, without returning raw input."""
        try:
            event = Event.parse(value); v = event.data
            now = reference_now(now)
            require(v["session_id"] == self._context["session_id"], "wrong_session")
            if v["mode"] in {"STOP", "CANCEL"}:
                require(v["host_id"] in self._control_hosts, "unexpected_control_host")
                require(v["event_id"] not in self._seen, "duplicate_event")
                if v["mode"] == "CANCEL":
                    require(self._pending is not None and v["interaction_id"] == self._pending.data["interaction_id"], "cancel_not_current")
                    self._wallclock(v, now)
                # STOP bypasses stale revisions, expiry and stream ordering.
                self._seen.add(v["event_id"])
                self._pending = None; self._deadline = None
                if v["mode"] == "STOP": self._stopped = True
                return IngestResult(True, "stop_requested" if self._stopped else "cancelled", response=event)
            self._local_clock(monotonic_now)
            require(not self._stopped, "session_stopped")
            require(v["kind"] == "RESPONSE", "not_response")
            require(v["host_id"] in self._response_hosts, "unexpected_response_host")
            self._matches_context(v); self._fresh(v); self._wallclock(v, now)
            require(self._pending is not None, "no_pending_interaction")
            request = self._pending; q = request.data
            require(v["response_to"] == q["event_id"] and all(v[k] == q[k] for k in BIND_FIELDS), "response_binding_mismatch")
            require(monotonic_now < self._deadline, "local_deadline_expired")
            require(now < event_clock(q, "expires_at")[0], "request_expired")
            p = v["payload"]; symbols = {o["symbol"] for o in q["payload"]["options"]}
            if p["score_kind"] == "CALIBRATED_PROBABILITY":
                require(p["calibration_id"] in self._calibration_ids, "unregistered_calibration")
            require(p["symbol"] is None or p["symbol"] in symbols, "unknown_response_symbol")
            require(p["raw_scores"] is None or set(p["raw_scores"]) == symbols, "score_mapping_mismatch")
            w = p["sample_window"]
            if w is not None:
                start = interval({"host_id": v["host_id"], "at": w["started_at"], "offset_ms": v["clock_offset_ms"], "uncertainty_ms": v["clock_uncertainty_ms"]})[0]
                end = interval({"host_id": v["host_id"], "at": w["ended_at"], "offset_ms": v["clock_offset_ms"], "uncertainty_ms": v["clock_uncertainty_ms"]})[1]
                _, issued = event_clock(q)
                if q["mode"] == "CHANNEL_PROBE":
                    prompt = q["payload"]["probe_prompt"]
                    if prompt["presented_at"] is not None:
                        issued = max(issued, interval(dict(prompt["presented_at"]))[1])
                require(start >= issued and end <= event_clock(v)[0] and end < event_clock(q, "expires_at")[0], "window_outside_interaction")
                require(w["window_id"] not in self._window_ids, "window_reused")
                for rid, a, b in self._windows:
                    require(not (rid == w["recording_id"] and w["sample_start"] < b and a < w["sample_end"]), "overlapping_evidence")
            likelihood = None; prompt_symbol = None
            if p["status"] == "OBSERVATION":
                likelihood = {"TASK_QUERY": "TASK", "CHANNEL_PROBE": "PROBE", "CONFIRM": "CONFIRM"}[q["mode"]]
                if q["mode"] == "CHANNEL_PROBE":
                    require(q["payload"]["probe_prompt"]["presentation_ack"], "probe_not_presented")
                    prompt_symbol = q["payload"]["probe_prompt"]["symbol"]
            # Commit only after every binding/time/evidence check passed.
            if w is not None:
                self._window_ids.add(w["window_id"])
                self._windows.append((w["recording_id"], w["sample_start"], w["sample_end"]))
            self._remember(v)
            self._pending = None; self._deadline = None
            return IngestResult(True, "observation" if likelihood else p["status"].lower(), event, request, likelihood, prompt_symbol)
        except ProtocolError as exc:
            return IngestResult(False, str(exc))

    def expire_pending(self, *, monotonic_now):
        """Timeout on this process's monotonic clock; no default response generated."""
        self._local_clock(monotonic_now)
        if self._pending is not None and monotonic_now >= self._deadline:
            self._pending = None; self._deadline = None
            return IngestResult(True, "timeout")
        return IngestResult(False, "not_due")
