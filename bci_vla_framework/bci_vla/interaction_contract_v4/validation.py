"""Reference validation of the frozen V4-02 schema subset and semantic bindings.

No network, devices, model imports, action dispatch or authorization is present.
This is not a general JSON Schema engine and cannot certify feature provenance.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "configs/v4/interfaces.json"
ROLES = {"TASK", "INTENT_QUERY", "PROBE", "CONFIRM"}


class ContractError(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise ContractError(code)


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "json:duplicate_key")
        result[key] = value
    return result


def load_json(path):
    def bad_constant(_):
        raise ContractError("json:nonfinite")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_pairs, parse_constant=bad_constant)


def canonical_digest(value):
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ContractError("json:noncanonical_value") from exc
    return hashlib.sha256(raw.encode()).hexdigest()


def _shape(value, schema, definitions, path="$"):
    if "$ref" in schema:
        key = schema["$ref"]
        require(key.startswith("#/$defs/") and key[8:] in definitions, "schema:local_ref_only")
        return _shape(value, definitions[key[8:]], definitions, path)
    for name in ("anyOf", "oneOf"):
        if name in schema:
            count = 0
            for branch in schema[name]:
                try:
                    _shape(value, branch, definitions, path)
                    count += 1
                except ContractError:
                    pass
            require(count >= 1 if name == "anyOf" else count == 1, "shape:" + name + ":" + path)
    if "const" in schema:
        require(type(value) is type(schema["const"]) and value == schema["const"], "shape:const:" + path)
    if "enum" in schema:
        require(any(type(value) is type(x) and value == x for x in schema["enum"]), "shape:enum:" + path)
    kind = schema.get("type")
    if kind is not None:
        valid = {"null": value is None, "boolean": type(value) is bool, "integer": type(value) is int,
                 "number": type(value) in (int, float), "string": type(value) is str,
                 "array": type(value) is list, "object": type(value) is dict}
        require(valid.get(kind, False), "shape:type:" + path)
    if type(value) in (int, float):
        require(math.isfinite(value), "shape:nonfinite:" + path)
        if "minimum" in schema:
            require(value >= schema["minimum"], "shape:minimum:" + path)
        if "maximum" in schema:
            require(value <= schema["maximum"], "shape:maximum:" + path)
    if type(value) is str:
        require(len(value) >= schema.get("minLength", 0) and len(value) <= schema.get("maxLength", 10**9), "shape:string_length:" + path)
        if "pattern" in schema:
            require(re.search(schema["pattern"], value) is not None, "shape:pattern:" + path)
    if type(value) is list:
        require(len(value) >= schema.get("minItems", 0) and len(value) <= schema.get("maxItems", 10**9), "shape:array_length:" + path)
        for i, item in enumerate(value):
            if "items" in schema:
                _shape(item, schema["items"], definitions, f"{path}[{i}]")
    if type(value) is dict:
        require(all(type(key) is str for key in value), "shape:nonstring_key:" + path)
        require(set(schema.get("required", ())) <= set(value), "shape:missing_field:" + path)
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            require(set(value) <= set(props), "shape:unexpected_field:" + path)
        for key in set(value) & set(props):
            _shape(value[key], props[key], definitions, path + "." + key)


def _unique(rows, key, code):
    require(len({r[key] for r in rows}) == len(rows), code)


def _task(task, public, current=True):
    ids = {o["instance_id"] for o in public["scene"]["objects"] if o["object_valid"]}
    if current:
        require(task["context"] == public["context"], "task:stale_context")
        require(task["valid_until_s"] > public["as_of_s"], "task:expired")
        require(task["source_id"] in ids, "task:unknown_source")
    if task["action"] == "pick":
        require(task["destination_id"] is None, "task:pick_destination_must_be_null")
    else:
        require(task["destination_id"] is not None and task["destination_id"] != task["source_id"], "task:place_needs_distinct_destination")
        if current:
            require(task["destination_id"] in ids, "task:unknown_destination")


def _question(q, public, current=True):
    require(q["issued_at_s"] < q["expires_at_s"], "question:invalid_time_window")
    _unique(q["options"], "option_id", "question:duplicate_option")
    _unique(q["options"], "symbol", "question:duplicate_symbol")
    options = q["options"]
    if current:
        require(q["context"] == public["context"], "question:stale_context")
    if q["role"] == "PROBE":
        require(q["diagnostic_option_id"] in {o["option_id"] for o in options}, "question:known_probe_target_required")
        require(all(o["kind"] == "DIAGNOSTIC" for o in options), "question:probe_options_not_tasks")
    else:
        require(q["diagnostic_option_id"] is None, "question:unexpected_probe_target")
    if q["role"] == "CONFIRM":
        require(q["proposal"] is not None, "question:exact_confirmation_task_required")
        _task(q["proposal"], public, current)
        require(q["proposal"]["context"] == q["context"] and q["expires_at_s"] <= q["proposal"]["valid_until_s"], "question:confirmation_binding")
        require(len(options) == 2 and {o["reference"] for o in options} == {"accept", "reject"} and all(o["kind"] == "DECISION" for o in options), "question:confirmation_options")
    else:
        require(q["proposal"] is None, "question:proposal_only_for_confirmation")
    if q["role"] in {"TASK", "INTENT_QUERY"}:
        allowed = {"OBJECT"} if q["role"] == "TASK" else {"OBJECT", "TASK"}
        require(all(o["kind"] in allowed for o in options), "question:invalid_task_option_kind")
        if current:
            ids = {o["instance_id"] for o in public["scene"]["objects"] if o["object_valid"]}
            tasks = {t["task_id"] for t in public["task_options"]}
            require(all(o["reference"] in (ids if o["kind"] == "OBJECT" else tasks) for o in options), "question:unknown_reference")


def _event(e, public):
    now = public["as_of_s"]
    require(e["started_at_s"] <= e["ended_at_s"] <= now, "event:future_or_reversed_time")
    if e["role"] in {"STOP", "CANCEL"}:
        require(e["question"] is None and e["evidence"] is None and e["original_role"] is None and e["valid"] and e["invalid_reason"] is None, "event:invalid_control")
        return
    q = e["question"]
    require(q is not None and e["evidence"] is not None, "event:question_and_evidence_required")
    require(q["issued_at_s"] <= e["started_at_s"], "event:response_before_question")
    if e["role"] == "NO_RESPONSE":
        require(e["original_role"] == q["role"], "event:no_response_lost_original_role")
        require(e["ended_at_s"] >= q["expires_at_s"], "event:no_response_before_timeout")
    elif e["role"] == "REJECT":
        require(e["original_role"] == q["role"] == "CONFIRM", "event:reject_not_confirmation")
    else:
        require(e["role"] == q["role"] and e["original_role"] is None, "event:role_binding")
    violation = ("STALE_CONTEXT" if q["context"] != public["context"] else
                 "PRESENTATION_UNACKNOWLEDGED" if not q["presentation_ack"] else
                 "EXPIRED" if e["role"] != "NO_RESPONSE" and e["ended_at_s"] > q["expires_at_s"] else None)
    require(e["valid"] == (violation is None) and e["invalid_reason"] == violation, "event:invalid_validity_claim")
    _question(q, public, e["valid"])
    ev = e["evidence"]
    opts = {o["option_id"]: o for o in q["options"]}
    if e["role"] == "NO_RESPONSE":
        require(ev == {"score_kind": "NONE", "values": [], "selected_option_id": None, "calibration": None}, "evidence:no_response_cannot_select")
        return
    require(ev["score_kind"] != "NONE" and ev["selected_option_id"] in opts, "evidence:observed_selection_required")
    _unique(ev["values"], "option_id", "evidence:duplicate_score")
    require({v["option_id"] for v in ev["values"]} == set(opts), "evidence:score_axis_mismatch")
    if ev["score_kind"] == "CALIBRATED_PROBABILITY":
        require(all(0 <= v["value"] <= 1 for v in ev["values"]) and math.isclose(sum(v["value"] for v in ev["values"]), 1, abs_tol=1e-6), "evidence:invalid_probability_mass")
        c = ev["calibration"]
        require(c is not None and c["mapping_id"] == q["context"]["mapping_id"] and c["mapping_revision"] == q["context"]["mapping_revision"] and c["layout_id"] == q["layout_id"] and q["role"] in c["roles"] and c["valid_until_s"] >= e["ended_at_s"], "evidence:calibration_scope_mismatch")
    else:
        require(ev["calibration"] is None, "evidence:raw_or_hard_not_calibrated")
        if ev["score_kind"] == "HARD_LABEL":
            require(all(v["value"] == (1 if v["option_id"] == ev["selected_option_id"] else 0) for v in ev["values"]), "evidence:hard_label_encoding")
    if q["role"] == "CONFIRM":
        decision = opts[ev["selected_option_id"]]["reference"]
        require((e["role"] == "CONFIRM" and decision == "accept") or (e["role"] == "REJECT" and decision == "reject"), "event:confirmation_decision_role")


def _public(public):
    now = public["as_of_s"]
    scene = public["scene"]
    require(scene["observed_at_s"] <= now, "scene:future_observation")
    require(scene["image_valid"] == (scene["image_ref"] is not None), "scene:image_availability")
    objects = scene["objects"]
    _unique(objects, "instance_id", "scene:duplicate_instance")
    _unique(objects, "alias", "scene:duplicate_alias")
    for o in objects:
        require(o["bbox_xyxy"] is not None or not o["object_valid"], "scene:valid_object_needs_box")
        if o["bbox_xyxy"] is not None:
            x1,y1,x2,y2 = o["bbox_xyxy"]
            require(x1 < x2 and y1 < y2, "scene:box_order")
        for name, dim in (("roi", scene["roi_dim"]), ("semantic", scene["semantic_dim"])):
            feature = o[name + "_feature"]
            require(o[name + "_valid"] == (feature is not None), "scene:feature_mask_mismatch")
            require(feature is None or len(feature) == dim, "scene:feature_dimension_mismatch")
    tr = scene["truncation"]
    require(tr["retained_count"] == len(objects) and tr["original_count"] >= len(objects), "scene:truncation_counts")
    require((tr["policy"] == "none") == (tr["original_count"] == len(objects)), "scene:truncation_policy")
    history = public["history"]
    _unique(history, "event_id", "history:duplicate_event")
    require(all(a["sequence"] < b["sequence"] and a["ended_at_s"] <= b["ended_at_s"] for a,b in zip(history,history[1:])), "history:order")
    require(public["current_event_id"] == (history[-1]["event_id"] if history else None), "history:current_event_not_last_once")
    for t in public["task_options"]:
        _task(t, public)
    _unique(public["task_options"], "task_id", "task:duplicate_candidate")
    state = public["public_state"]
    summary = state["diagnostics"]
    require(summary["context"] == public["context"], "diagnostics:stale_context")
    all_events = list(history) + summary["receipts"]
    confirmed = state["confirmed_task"]
    if confirmed is not None:
        _task(confirmed["task"], public)
        proof = confirmed["confirmation_receipt"]
        all_events.append(proof)
        require(proof["role"] == "CONFIRM" and proof["valid"] and proof["question"] is not None and proof["question"]["proposal"] == confirmed["task"], "state:confirmation_proof_mismatch")
    if state["pending_suggestion"] is not None:
        _task(state["pending_suggestion"], public)
    seen = {}; questions = {}
    for e in all_events:
        _event(e, public)
        require(e["event_id"] not in seen or e == seen[e["event_id"]], "history:conflicting_receipt")
        seen[e["event_id"]] = e
        if e["question"] is not None:
            qid = e["question"]["question_id"]
            require(qid not in questions or questions[qid] == e["event_id"], "history:question_answer_replayed")
            questions[qid] = e["event_id"]
    by_id = {e["event_id"]:e for e in history}
    if public["selection_event_id"] is not None:
        require(public["selection_event_id"] in by_id, "selection:event_not_in_window")
        e = by_id[public["selection_event_id"]]
        require(e["valid"] and e["role"] in {"TASK", "INTENT_QUERY"} and all(o["kind"] == "OBJECT" for o in e["question"]["options"]), "selection:diagnostic_or_invalid_not_intent")
    pending = state["pending_question"]
    if pending is not None:
        _question(pending, public)
        require(pending["issued_at_s"] <= now < pending["expires_at_s"], "state:pending_question_expired_or_future")
        require(pending["question_id"] not in questions, "state:pending_question_already_answered")
        if pending["role"] == "CONFIRM":
            require(pending["proposal"] == state["pending_suggestion"], "state:pending_confirmation_not_exact_suggestion")
    receipts = summary["receipts"]
    _unique(receipts, "event_id", "diagnostics:duplicate_attempt")
    successes = 0; noresponses = 0; covered = set()
    for e in receipts:
        q = e["question"]
        require(e["valid"] and q is not None and q["role"] == "PROBE" and e["role"] in {"PROBE", "NO_RESPONSE"}, "diagnostics:task_or_invalid_evidence")
        require(q["layout_id"] == summary["layout_id"] and 0 <= now-e["ended_at_s"] <= summary["window_s"], "diagnostics:wrong_condition_or_window")
        target = next(o for o in q["options"] if o["option_id"] == q["diagnostic_option_id"])
        covered.add(target["symbol"])
        noresponses += e["role"] == "NO_RESPONSE"
        successes += e["role"] != "NO_RESPONSE" and e["evidence"]["selected_option_id"] == target["option_id"]
    latest = max((e["ended_at_s"] for e in receipts), default=None)
    require(summary["attempt_count"] == len(receipts) and summary["correct_count"] == successes and summary["no_response_count"] == noresponses and summary["covered_symbols"] == sorted(covered) and summary["latest_ended_at_s"] == latest, "diagnostics:invented_statistics")
    status = ("EXPIRED" if latest is not None and now-latest > summary["ttl_s"] else
              "SUFFICIENT" if len(receipts) >= summary["min_attempts"] and len(covered) >= summary["min_covered_symbols"] and not summary["truncated"] else "INSUFFICIENT")
    require(summary["status"] == status, "diagnostics:insufficient_evidence_claim")
    if state["stop_latched"]:
        require(confirmed is None and state["pending_question"] is None and state["pending_suggestion"] is None, "state:stop_did_not_clear")
    if history and history[-1]["role"] in {"STOP", "CANCEL", "REJECT"}:
        require(confirmed is None and pending is None and state["pending_suggestion"] is None, "state:control_did_not_clear")
        require(history[-1]["role"] != "STOP" or state["stop_latched"], "state:stop_not_latched")


def _private(labels, public):
    require(labels["sample_id"] == public["sample_id"] and labels["public_input_sha256"] == canonical_digest(public), "labels:public_binding")
    identity = labels["identifiability"]
    targets = labels["labels"]
    intent = targets["intent"]
    ids = {o["instance_id"] for o in public["scene"]["objects"]}
    require(len(set(intent["candidate_ids"])) == len(intent["candidate_ids"]) and set(intent["candidate_ids"]) <= ids, "labels:invalid_candidates")
    if intent["mask"]:
        require(identity["intent"] != "UNOBSERVABLE", "labels:unobservable_intent_must_be_masked")
        require(intent["mode"] != "NONE" and (bool(intent["candidate_ids"]) or intent["outside"]), "labels:empty_intent_target")
        if intent["mode"] == "POINT":
            require(identity["intent"] == "IDENTIFIED" and len(intent["candidate_ids"])+int(intent["outside"]) == 1, "labels:unidentifiable_point_target")
        if identity["intent"] == "AMBIGUOUS":
            require(intent["mode"] == "SET" and len(intent["candidate_ids"])+int(intent["outside"]) >= 2, "labels:ambiguity_needs_set")
    else:
        require(intent["mode"] == "NONE" and not intent["candidate_ids"] and not intent["outside"], "labels:masked_intent_not_empty")
    main = targets["main_task"]
    if main["mask"]:
        require(main["mode"] != "NONE" and main["text"] is not None, "labels:main_target_missing")
        if main["mode"] == "TASK":
            require(identity["intent"] == "IDENTIFIED" and main["task"] is not None, "labels:unidentifiable_forced_task")
            _task(main["task"], public)
        else:
            require(main["task"] is None, "labels:clarification_not_forced_task")
    else:
        require(main["mode"] == "NONE" and main["task"] is None and main["text"] is None, "labels:masked_main_not_empty")
    channel = targets["channel"]
    if channel["mask"]:
        require(identity["channel"] == "SUPPORTED" and public["public_state"]["diagnostics"]["status"] == "SUFFICIENT" and type(channel["next_probe_success"]) is bool and channel["target_time_s"] is not None and channel["target_time_s"] > public["as_of_s"], "labels:channel_target_not_supported")
        summary = public["public_state"]["diagnostics"]
        require(channel["target_time_s"] <= min(public["as_of_s"]+summary["window_s"], summary["latest_ended_at_s"]+summary["ttl_s"]), "labels:channel_target_outside_horizon")
    else:
        require(channel["next_probe_success"] is None and channel["target_time_s"] is None, "labels:masked_channel_not_empty")
    require(targets["interaction"]["teacher_input_sha256"] == canonical_digest(public), "labels:teacher_input_binding")


def _output(output, public):
    require(output["source_input_sha256"] == canonical_digest(public) and output["context"] == public["context"] and output["as_of_s"] == public["as_of_s"], "output:stale_input_binding")
    prefix = output["soft_prefix"]
    require(prefix["shape"] == [8, len(prefix["tokens"][0])] and all(len(row) == prefix["shape"][1] for row in prefix["tokens"]), "output:prefix_shape_mismatch")
    require(public["provenance"]["feature_profile_id"] != "fixture-4x3" or prefix["shape"][1] == 6, "output:fixture_vlm_dimension")
    intent = output["intent"]; rows = intent["candidate_distribution"]
    ids = {o["instance_id"] for o in public["scene"]["objects"] if o["object_valid"]}
    _unique(rows, "instance_id", "output:duplicate_candidate")
    require({r["instance_id"] for r in rows} == ids and set(intent["allowed_candidate_ids"]) <= ids and len(set(intent["allowed_candidate_ids"])) == len(intent["allowed_candidate_ids"]), "output:candidate_axis_mismatch")
    require(math.isclose(sum(r["probability"] for r in rows)+intent["outside_probability"], 1, abs_tol=1e-6), "output:intent_mass")
    channel = output["channel"]; summary = public["public_state"]["diagnostics"]
    require(channel["status"] == summary["status"] and channel["target_layout_id"] == summary["layout_id"], "output:diagnostic_scope")
    require((channel["probability"] is not None) == (channel["status"] == "SUFFICIENT"), "output:unsupported_reliability_probability")
    proposal = output["task_proposal"]; interaction = output["interaction"]
    if proposal is not None:
        _task(proposal, public)
        require(proposal["source_id"] in intent["allowed_candidate_ids"] and not intent["abstain"] and not intent["missing_slots"], "output:pointer_text_conflict")
    if interaction["action"] == "PROPOSE_TASK":
        require(proposal is not None and interaction["question"] is None, "output:proposal_required")
    elif interaction["action"] in {"ASK_INTENT", "CHECK_CHANNEL"}:
        q = interaction["question"]
        require(q is not None and q["role"] == ("INTENT_QUERY" if interaction["action"] == "ASK_INTENT" else "PROBE"), "output:wrong_clarification_role")
        _question(q, public)
        require(q["issued_at_s"] == public["as_of_s"], "output:question_not_current")
    else:
        require(interaction["question"] is None, "output:wait_cannot_issue_question")
    if public["public_state"]["stop_latched"]:
        require(interaction["action"] == "WAIT" and proposal is None, "output:stop_must_wait")
    if public["public_state"]["pending_question"] is not None:
        require(interaction["action"] == "WAIT" and proposal is None, "output:pending_question_requires_wait")


def validate_document(value, public_input=None):
    schema = load_json(SCHEMA_PATH)
    require(type(value) is dict, "document:object_required")
    kind = {"PUBLIC_INPUT": "public_input", "PRIVATE_LABELS": "private_labels", "MODULE_OUTPUT": "module_output"}.get(value.get("record_type"))
    require(kind is not None, "document:unknown_record_type")
    _shape(value, schema["$defs"][kind], schema["$defs"])
    if kind == "public_input":
        _public(value)
    else:
        require(public_input is not None, "document:bound_public_input_required")
        validate_document(public_input)
        (_private if kind == "private_labels" else _output)(value, public_input)
    return True


def validate_transition(previous, current):
    validate_document(previous); validate_document(current)
    require(current["as_of_s"] >= previous["as_of_s"], "transition:clock_reversed")
    if previous["context"] != current["context"]:
        state = current["public_state"]
        require(state["confirmed_task"] is None and state["pending_question"] is None and state["pending_suggestion"] is None and state["diagnostics"]["attempt_count"] == 0, "transition:context_change_not_invalidated")
    else:
        a = previous["public_state"]; b = current["public_state"]
        require(not a["stop_latched"] or b["stop_latched"], "transition:stop_cannot_unlatch")
        event = current["history"][-1] if current["history"] else None
        if a["confirmed_task"] != b["confirmed_task"]:
            expired_clear = b["confirmed_task"] is None and a["confirmed_task"] is not None and a["confirmed_task"]["task"]["valid_until_s"] <= current["as_of_s"]
            require(expired_clear or (event is not None and event["role"] in {"CONFIRM", "STOP", "CANCEL", "REJECT"} and event["valid"]), "transition:nonconfirmation_changed_task")
        if event and event["role"] == "CONFIRM" and a["confirmed_task"] != b["confirmed_task"]:
            require(b["confirmed_task"] is not None and b["confirmed_task"]["confirmation_receipt"] == event and event["valid"], "transition:new_confirmation_proof_required")
    return True


def control_route(event):
    """Contract fixture routing only; upstream authentication/existing STOP is separate."""
    schema = load_json(SCHEMA_PATH)
    _shape(event, schema["$defs"]["event"], schema["$defs"])
    require(event["role"] in {"STOP", "CANCEL"} and event["question"] is None and event["evidence"] is None and event["original_role"] is None and event["started_at_s"] <= event["ended_at_s"] and event["valid"] and event["invalid_reason"] is None, "control:not_validated_control_event")
    return "STOP_DIRECT_EXISTING_PATH" if event["role"] == "STOP" else "CANCEL_REVOKE_EXISTING_PATH"


def _diag_features(event, now):
    q = event["question"]; ev = event["evidence"]
    values = {x["option_id"]: x["value"] for x in ev["values"]}
    target = q["diagnostic_option_id"]
    return {"no_response": event["role"] == "NO_RESPONSE", "known_target_match": ev["selected_option_id"] == target,
        "known_target_score": values.get(target), "other_scores_sorted": sorted(v for k,v in values.items() if k != target),
        "score_kind": ev["score_kind"], "duration_s": event["ended_at_s"]-event["started_at_s"], "age_s": now-event["ended_at_s"]}


def neural_view(public):
    """Allowlisted interface view, not tensorization or a neural encoder."""
    validate_document(public)
    aliases = {o["instance_id"]: o["alias"] for o in public["scene"]["objects"]}
    def task_view(t):
        return None if t is None else {"action":t["action"], "source_alias":aliases[t["source_id"]], "destination_alias":aliases.get(t["destination_id"])}
    task_lookup = {t["task_id"]:task_view(t) for t in public["task_options"]}
    def reference(option):
        if option["kind"] == "OBJECT":
            return aliases[option["reference"]]
        if option["kind"] == "TASK":
            return task_lookup[option["reference"]]
        return option["reference"]
    task_events = []; diagnostic_events = []
    for e in public["history"]:
        if not e["valid"] or e["question"] is None:
            continue
        q = e["question"]
        if q["role"] == "PROBE":
            diagnostic_events.append(_diag_features(e, public["as_of_s"]))
        else:
            scores = {r["option_id"]:r["value"] for r in e["evidence"]["values"]}
            task_events.append({"role": e["role"], "original_role": e["original_role"], "age_s": public["as_of_s"]-e["ended_at_s"],
                "score_kind": e["evidence"]["score_kind"], "scores": [scores.get(o["option_id"]) for o in q["options"]],
                "selected_index": next((i for i,o in enumerate(q["options"]) if o["option_id"] == e["evidence"]["selected_option_id"]), None),
                "proposal": task_view(q["proposal"]),
                "option_refs": [reference(o) for o in q["options"]], "option_kinds": [o["kind"] for o in q["options"]]})
    s = public["public_state"]; d = s["diagnostics"]
    confirmed = s["confirmed_task"]
    selection = next((e for e in public["history"] if e["event_id"] == public["selection_event_id"]), None)
    selection_view = None
    if selection is not None:
        ev = selection["evidence"]; scores = {r["option_id"]:r["value"] for r in ev["values"]}
        selection_view = {"score_kind":ev["score_kind"], "object_scores":[{"alias":reference(o),"score":scores[o["option_id"]]} for o in selection["question"]["options"]],
            "selected_alias": next(reference(o) for o in selection["question"]["options"] if o["option_id"] == ev["selected_option_id"]),
            "same_history_event":True}
    return {"objects": [{k:deepcopy(v) for k,v in o.items() if k != "instance_id"} for o in sorted(public["scene"]["objects"], key=lambda o:int(o["alias"].split("_")[1]))],
        "selection":selection_view, "task_options":[task_view(t) for t in public["task_options"]], "pending_suggestion":task_view(s["pending_suggestion"]),
        "task_history": task_events, "diagnostic_history": diagnostic_events,
        "confirmed_task": None if confirmed is None else task_view(confirmed["task"]),
        "pending_role": None if s["pending_question"] is None else s["pending_question"]["role"], "stop_latched":s["stop_latched"],
        "pending_question": None if s["pending_question"] is None else {"role":s["pending_question"]["role"], "remaining_s":s["pending_question"]["expires_at_s"]-public["as_of_s"],
            "proposal":task_view(s["pending_question"]["proposal"]), "option_refs":[] if s["pending_question"]["role"]=="PROBE" else [reference(o) for o in s["pending_question"]["options"]]},
        "diagnostic_summary": {"status":d["status"],"attempt_count":d["attempt_count"],"correct_count":d["correct_count"],"no_response_count":d["no_response_count"],"covered_symbol_count":len(d["covered_symbols"]),"latest_age_s":None if d["latest_ended_at_s"] is None else public["as_of_s"]-d["latest_ended_at_s"],"truncated":d["truncated"]}}
