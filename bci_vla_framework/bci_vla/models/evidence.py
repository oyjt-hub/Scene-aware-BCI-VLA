"""Final-prefix-only evidence policy and actual interval-based time accounting."""
from dataclasses import dataclass

from ..protocols import fields, number, require, text


@dataclass(frozen=True)
class EvidenceToken:
    event_id: str
    interaction_id: str
    mode: str
    recording_id: str
    sample_start: int
    sample_end: int


class FinalPrefixLedger:
    """Choose one final prefix before inference. No product of overlapping prefixes.

    This does not implement conditional score calibration or posterior replacement.
    """
    def __init__(self):
        self._interactions = set()
        self._events = set()
        self._windows = []

    def claim(self, accepted_result, *, finalized):
        require(finalized is True, "intermediate_prefix_not_evidence")
        require(accepted_result.accepted and accepted_result.request is not None and accepted_result.response is not None and accepted_result.likelihood_kind is not None, "accepted_observation_required")
        q, r = accepted_result.request.data, accepted_result.response.data
        w = r["payload"]["sample_window"]
        require(w is not None, "missing_window")
        key = (q["session_id"], q["interaction_id"])
        require(key not in self._interactions and r["event_id"] not in self._events, "prefix_already_consumed")
        for recording, start, end in self._windows:
            require(not (recording == w["recording_id"] and w["sample_start"] < end and start < w["sample_end"]), "overlapping_prefix_evidence")
        token = EvidenceToken(r["event_id"], q["interaction_id"], q["mode"], w["recording_id"], w["sample_start"], w["sample_end"])
        self._interactions.add(key); self._events.add(r["event_id"])
        self._windows.append((w["recording_id"], w["sample_start"], w["sample_end"]))
        return token


PHASES = {"reading", "switching", "acquisition", "decoding", "network", "retry", "return_to_task", "confirmation", "computation"}


def union_seconds(intervals):
    end = None; total = 0.0
    for a, b in sorted(intervals):
        if end is None or a >= end:
            total += b - a
        else:
            total += max(0, b - end)
        end = b if end is None else max(end, b)
    return total


def summarize_timing(segments, *, required_phases, clock_id, window_start_s, window_end_s):
    """Times must already share one trusted local/reference clock domain."""
    text(clock_id); number(window_start_s, 0); number(window_end_s, 0)
    require(window_end_s >= window_start_s, "invalid_time_window")
    require(isinstance(required_phases, (list, tuple)) and all(p in PHASES for p in required_phases), "invalid_phase")
    require(isinstance(segments, list), "invalid_segments")
    by_phase = {}; busy = []
    for item in segments:
        fields(item, {"phase", "clock_id", "start_s", "end_s", "user_busy"})
        require(item["phase"] in PHASES and item["clock_id"] == clock_id, "incompatible_clock_or_phase")
        require(type(item["user_busy"]) is bool, "invalid_busy_flag")
        a, b = item["start_s"], item["end_s"]
        number(a, 0); number(b, 0)
        require(window_start_s <= a <= b <= window_end_s, "segment_outside_window")
        by_phase.setdefault(item["phase"], []).append((a, b))
        if item["user_busy"]: busy.append((a, b))
    missing = sorted(set(required_phases) - set(by_phase))
    return {"clock_id": clock_id, "total_elapsed_s": window_end_s - window_start_s, "observed_user_busy_s": union_seconds(busy), "phase_seconds": {p: union_seconds(rows) for p, rows in by_phase.items()}, "missing_phases": missing, "decomposition_complete": not missing, "user_busy_s": None if missing else union_seconds(busy), "warning": "phase times can overlap; never sum them to charge total time twice"}
