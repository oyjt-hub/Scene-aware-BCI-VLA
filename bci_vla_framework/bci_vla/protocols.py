"""Strict, versioned public records. Calibration and inference live elsewhere."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import json
import math
from types import MappingProxyType
from typing import Mapping

SCHEMA_VERSION = "bci-vla/1.0"
MODES = {"TASK_QUERY", "CHANNEL_PROBE", "CONFIRM", "CANCEL", "STOP", "EXECUTION_EVENT"}
STATUSES = {"OBSERVATION", "NO_CONTROL", "NO_RESPONSE", "INVALID_SIGNAL", "TRANSPORT_FAILURE", "TIMEOUT"}


class ProtocolError(ValueError):
    """A stable reason code; never includes untrusted input or private labels."""


def require(ok, code):
    if not ok:
        raise ProtocolError(code)


def fields(value, required, optional=()):
    require(isinstance(value, dict), "expected_object")
    require(set(required) <= set(value), "missing_field")
    require(set(value) <= set(required) | set(optional), "unsupported_field")


def text(value, nullable=False):
    if value is None and nullable:
        return
    require(isinstance(value, str) and 0 < len(value) <= 256 and not any(ord(c) < 32 for c in value), "invalid_text")


def number(value, low=None, high=None):
    try:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except (OverflowError, ValueError):
        valid = False
    require(valid, "invalid_number")
    require((low is None or value >= low) and (high is None or value <= high), "number_out_of_range")


def integer(value, low=0):
    require(type(value) is int and low <= value <= 2**63 - 1, "invalid_integer")


def enum(value, values):
    require(isinstance(value, str) and value in values, "invalid_enum")


def timestamp(value):
    require(isinstance(value, str), "invalid_timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ProtocolError("invalid_timestamp") from None
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None, "timezone_required")
    return parsed.astimezone(timezone.utc)


def interval(value):
    fields(value, {"host_id", "at", "offset_ms", "uncertainty_ms"})
    text(value["host_id"])
    number(value["offset_ms"], -86400000, 86400000)
    number(value["uncertainty_ms"], 0, 86400000)
    try:
        center = timestamp(value["at"]) - timedelta(milliseconds=value["offset_ms"])
        error = timedelta(milliseconds=value["uncertainty_ms"])
        return center - error, center + error
    except OverflowError:
        raise ProtocolError("invalid_clock_range") from None


def unique_strings(value, allow_empty=False):
    require(isinstance(value, list) and (allow_empty or bool(value)) and len(value) <= 256, "invalid_list")
    for item in value:
        text(item)
    require(len(set(value)) == len(value), "duplicate_item")


def freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(freeze(v) for v in value)
    return value


def thaw(value):
    if isinstance(value, Mapping):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [thaw(v) for v in value]
    return value


def json_object(value):
    """Clone and reject NaN, non-JSON types, non-string keys, duplicate JSON keys."""
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, "duplicate_json_key")
            result[k] = v
        return result
    def walk(v):
        if isinstance(v, dict):
            require(all(isinstance(k, str) for k in v), "invalid_key")
            for item in v.values():
                walk(item)
        elif isinstance(v, list):
            for item in v:
                walk(item)
        else:
            require(v is None or type(v) in {str, bool, int, float}, "non_json_value")
            if isinstance(v, float):
                number(v)
    try:
        if isinstance(value, str):
            require(len(value) <= 1_000_000, "message_too_large")
            value = json.loads(value, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ProtocolError("invalid_number")))
        walk(value)
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, OverflowError, RecursionError, json.JSONDecodeError):
        raise ProtocolError("invalid_json") from None


def validate_intent(v):
    fields(v, {"action", "source_id", "destination_id"})
    enum(v["action"], {"pick", "place", "fold"})
    text(v["source_id"])
    text(v["destination_id"], nullable=True)
    require((v["action"] == "place") == (v["destination_id"] is not None), "invalid_destination")
    require(v["source_id"] != v["destination_id"], "invalid_destination")


def validate_record(kind, v):
    if kind == "SceneState":
        fields(v, {"scene_id", "revision", "observed_at", "objects"})
        text(v["scene_id"]); integer(v["revision"]); interval(v["observed_at"])
        require(isinstance(v["objects"], list) and len(v["objects"]) <= 64, "invalid_objects")
        ids = []
        for obj in v["objects"]:
            fields(obj, {"instance_id", "label", "position_xy", "affordances"})
            text(obj["instance_id"]); text(obj["label"])
            require(isinstance(obj["position_xy"], list) and len(obj["position_xy"]) == 2, "invalid_position")
            for x in obj["position_xy"]:
                number(x, 0, 1)
            unique_strings(obj["affordances"], allow_empty=True)
            require(set(obj["affordances"]) <= {"pickable", "container", "foldable"}, "unsupported_affordance")
            ids.append(obj["instance_id"])
        require(len(ids) == len(set(ids)), "duplicate_instance")
    elif kind == "JointBelief":
        fields(v, {"goal_ids", "mode_ids", "mass", "as_of_event_id"})
        unique_strings(v["goal_ids"]); unique_strings(v["mode_ids"]); text(v["as_of_event_id"])
        require(isinstance(v["mass"], list) and len(v["mass"]) <= 65536, "invalid_mass")
        seen = set(); total = 0.0
        for row in v["mass"]:
            fields(row, {"goal_id", "mode_id", "probability"})
            text(row["goal_id"]); text(row["mode_id"])
            require(row["goal_id"] in v["goal_ids"] and row["mode_id"] in v["mode_ids"], "unknown_state")
            pair = row["goal_id"], row["mode_id"]
            require(pair not in seen, "duplicate_state"); seen.add(pair)
            number(row["probability"], 0, 1); total += row["probability"]
        require(abs(total - 1) <= 1e-9, "mass_not_normalized")
    elif kind == "ChannelProfile":
        fields(v, {"profile_id", "mode_ids", "calibration_id", "status", "evidence_kind"})
        text(v["profile_id"]); unique_strings(v["mode_ids"]); text(v["calibration_id"], nullable=True)
        enum(v["status"], {"uncalibrated", "calibrated", "simulation_only"})
        enum(v["evidence_kind"], {"none", "simulation", "independent_calibration"})
        if v["status"] == "calibrated":
            require(v["calibration_id"] is not None and v["evidence_kind"] == "independent_calibration", "missing_calibration")
        if v["status"] == "simulation_only":
            require(v["evidence_kind"] == "simulation", "invalid_provenance")
    elif kind == "ProbeModel":
        fields(v, {"model_id", "channel_profile_id", "compliance_model_ref", "transfer_calibration_ref", "status"})
        for k in ["model_id", "channel_profile_id"]:
            text(v[k])
        for k in ["compliance_model_ref", "transfer_calibration_ref"]:
            text(v[k], nullable=True)
        enum(v["status"], {"uncalibrated", "calibrated", "simulation_only"})
        if v["status"] == "calibrated":
            require(v["compliance_model_ref"] is not None and v["transfer_calibration_ref"] is not None, "missing_calibration")
    elif kind == "TaskContract":
        fields(v, {"contract_id", "task_id", "revision", "scene_id", "scene_revision", "intent", "status", "valid_until", "confirmation_event_id"})
        for k in ["contract_id", "task_id", "scene_id"]:
            text(v[k])
        integer(v["revision"]); integer(v["scene_revision"]); validate_intent(v["intent"])
        enum(v["status"], {"proposed", "confirmed", "cancelled", "expired"}); timestamp(v["valid_until"])
        text(v["confirmation_event_id"], nullable=True)
        require(v["status"] != "confirmed" or v["confirmation_event_id"] is not None, "missing_confirmation")
    elif kind == "ConsequenceModel":
        fields(v, {"model_id", "executor_id", "predicate_version", "cost_source", "evidence_ref", "status"})
        for k in ["model_id", "executor_id", "predicate_version"]:
            text(v[k])
        text(v["evidence_ref"], nullable=True)
        enum(v["cost_source"], {"unavailable", "simulation", "independent_trajectories"})
        enum(v["status"], {"unavailable", "simulation_only", "calibrated"})
        require(v["status"] != "calibrated" or (v["cost_source"] == "independent_trajectories" and v["evidence_ref"] is not None), "missing_calibration")
        require(v["status"] != "simulation_only" or (v["cost_source"] == "simulation" and v["evidence_ref"] is not None), "invalid_provenance")
        require(v["status"] != "unavailable" or v["cost_source"] == "unavailable", "invalid_provenance")
    elif kind == "ExecutionEvent":
        fields(v, {"execution_id", "contract_id", "contract_revision", "stage", "at", "evidence_ref"})
        text(v["execution_id"]); text(v["contract_id"]); integer(v["contract_revision"]); interval(v["at"])
        enum(v["stage"], {"command_dispatched", "action_finished", "endpoint_judged", "stopped", "failed"})
        text(v["evidence_ref"], nullable=True)
        require(v["stage"] != "endpoint_judged" or v["evidence_ref"] is not None, "missing_endpoint_evidence")
    else:
        raise ProtocolError("unknown_record")


@dataclass(frozen=True)
class PublicRecord:
    data: Mapping

    def __post_init__(self):
        value = json_object(self.data)
        validate_record(type(self).__name__, value)
        object.__setattr__(self, "data", freeze(value))

    def to_dict(self):
        return thaw(self.data)


class SceneState(PublicRecord): pass
class JointBelief(PublicRecord): pass
class ChannelProfile(PublicRecord): pass
class ProbeModel(PublicRecord): pass
class TaskContract(PublicRecord): pass
class ConsequenceModel(PublicRecord): pass
class ExecutionEvent(PublicRecord): pass


HEADER = {"schema_version", "event_id", "session_id", "host_id", "sequence", "kind", "mode", "interaction_id", "response_to", "query_id", "probe_id", "task_id", "task_revision", "scene_id", "scene_revision", "mapping_id", "stimulus_version", "channel_profile_id", "sent_at", "expires_at", "clock_offset_ms", "clock_uncertainty_ms", "payload"}


def event_clock(v, field="sent_at"):
    return interval({"host_id": v["host_id"], "at": v[field], "offset_ms": v["clock_offset_ms"], "uncertainty_ms": v["clock_uncertainty_ms"]})


def validate_window(w):
    fields(w, {"window_id", "recording_id", "sample_start", "sample_end", "sample_rate_hz", "channels", "started_at", "ended_at", "source_kind"})
    text(w["window_id"]); text(w["recording_id"]); integer(w["sample_start"]); integer(w["sample_end"])
    require(w["sample_end"] > w["sample_start"], "invalid_sample_range")
    number(w["sample_rate_hz"], 1); unique_strings(w["channels"])
    start, end = timestamp(w["started_at"]), timestamp(w["ended_at"])
    require(end > start, "invalid_window_time")
    duration = (w["sample_end"] - w["sample_start"]) / w["sample_rate_hz"]
    require(abs((end - start).total_seconds() - duration) <= 1 / w["sample_rate_hz"] + 1e-6, "sampling_time_mismatch")
    enum(w["source_kind"], {"offline_replay", "live_bci"})


def validate_event(v):
    fields(v, HEADER)
    require(v["schema_version"] == SCHEMA_VERSION, "unsupported_schema")
    enum(v["mode"], MODES); enum(v["kind"], {"REQUEST", "RESPONSE", "CONTROL", "EXECUTION"})
    for key in ["event_id", "session_id", "host_id", "interaction_id", "task_id", "scene_id", "mapping_id", "stimulus_version", "channel_profile_id"]:
        text(v[key])
    for key in ["response_to", "query_id", "probe_id"]:
        text(v[key], nullable=True)
    for key in ["sequence", "task_revision", "scene_revision"]:
        integer(v[key])
    number(v["clock_offset_ms"], -86400000, 86400000); number(v["clock_uncertainty_ms"], 0, 86400000)
    require(timestamp(v["expires_at"]) > timestamp(v["sent_at"]), "invalid_expiry")
    event_clock(v); event_clock(v, "expires_at")
    p = v["payload"]
    if v["mode"] in {"STOP", "CANCEL"}:
        require(v["kind"] == "CONTROL" and v["response_to"] is None and v["query_id"] is None and v["probe_id"] is None, "invalid_control")
        fields(p, {"reason"}); enum(p["reason"], {"user_request", "task_change_request", "system_abort"})
        return
    if v["mode"] == "EXECUTION_EVENT":
        require(v["kind"] == "EXECUTION" and v["response_to"] is None and v["query_id"] is None and v["probe_id"] is None, "invalid_execution")
        validate_record("ExecutionEvent", p)
        require(p["contract_revision"] == v["task_revision"], "contract_version_mismatch")
        return
    require(v["kind"] in {"REQUEST", "RESPONSE"}, "invalid_interaction_kind")
    if v["mode"] == "CHANNEL_PROBE":
        require(v["probe_id"] is not None and v["query_id"] is None, "invalid_probe_id")
    else:
        require(v["query_id"] is not None and v["probe_id"] is None, "invalid_query_id")
    if v["kind"] == "REQUEST":
        require(v["response_to"] is None, "request_has_response_to")
        fields(p, {"question", "options", "probe_prompt", "contract"})
        text(p["question"])
        require(isinstance(p["options"], list) and 2 <= len(p["options"]) <= 64, "invalid_options")
        ids, symbols = [], []
        for option in p["options"]:
            fields(option, {"option_id", "symbol", "semantic_ref"})
            for value in option.values(): text(value)
            ids.append(option["option_id"]); symbols.append(option["symbol"])
        require(len(set(ids)) == len(ids) and len(set(symbols)) == len(symbols), "duplicate_option")
        if v["mode"] == "CHANNEL_PROBE":
            a = p["probe_prompt"]
            fields(a, {"symbol", "source", "presented_at", "presentation_ack"})
            text(a["symbol"]); require(a["symbol"] in symbols, "unknown_prompt_symbol")
            require(a["source"] == "display_controller", "invalid_prompt_source")
            require(type(a["presentation_ack"]) is bool, "invalid_ack")
            if a["presented_at"] is not None: interval(a["presented_at"])
            require(not a["presentation_ack"] or a["presented_at"] is not None, "missing_presentation_time")
            require(p["contract"] is None, "probe_contains_contract")
        else:
            require(p["probe_prompt"] is None, "task_contains_probe_label")
            if v["mode"] == "CONFIRM":
                validate_record("TaskContract", p["contract"])
                c = p["contract"]
                require(c["status"] == "proposed", "confirmation_requires_proposal")
                require((c["task_id"], c["revision"], c["scene_id"], c["scene_revision"]) == (v["task_id"], v["task_revision"], v["scene_id"], v["scene_revision"]), "contract_version_mismatch")
            else:
                require(p["contract"] is None, "task_contains_contract")
    else:
        require(v["response_to"] is not None, "missing_response_to")
        fields(p, {"status", "symbol", "raw_scores", "score_kind", "calibration_id", "quality", "sample_window"})
        enum(p["status"], STATUSES); text(p["symbol"], nullable=True); text(p["calibration_id"], nullable=True)
        if p["raw_scores"] is not None:
            require(isinstance(p["raw_scores"], dict) and 1 <= len(p["raw_scores"]) <= 64, "invalid_scores")
            for k, val in p["raw_scores"].items(): text(k); number(val)
            enum(p["score_kind"], {"FBCCA_POWER", "UNCALIBRATED_SCORE", "CALIBRATED_PROBABILITY"})
            if p["score_kind"] in {"FBCCA_POWER", "CALIBRATED_PROBABILITY"}:
                for val in p["raw_scores"].values(): number(val, 0)
            if p["score_kind"] == "CALIBRATED_PROBABILITY":
                require(p["calibration_id"] is not None, "missing_calibration")
                require(abs(sum(p["raw_scores"].values()) - 1) <= 1e-9, "scores_not_normalized")
        else:
            require(p["score_kind"] is None and p["calibration_id"] is None, "score_metadata_without_scores")
        if p["quality"] is not None:
            fields(p["quality"], {"snr_db", "artifact_flags"})
            if p["quality"]["snr_db"] is not None: number(p["quality"]["snr_db"])
            unique_strings(p["quality"]["artifact_flags"], allow_empty=True)
        if p["sample_window"] is not None: validate_window(p["sample_window"])
        if p["status"] == "OBSERVATION":
            require(p["sample_window"] is not None and (p["symbol"] is not None or p["raw_scores"] is not None), "missing_observation")
        else:
            require(p["symbol"] is None, "nonresponse_has_selection")


@dataclass(frozen=True)
class Event:
    data: Mapping

    def __post_init__(self):
        v = json_object(self.data)
        validate_event(v)
        object.__setattr__(self, "data", freeze(v))

    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            return value
        return cls(json_object(value))

    def to_dict(self):
        return thaw(self.data)
