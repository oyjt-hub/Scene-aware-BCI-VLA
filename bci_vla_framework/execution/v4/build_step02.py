"""Create versioned V4 interface schemas and synthetic contract fixtures only."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT / "configs/v4"
EXAMPLES = PROJECT / "examples/v4"


def obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def arr(items, lo=0, hi=64):
    return {"type": "array", "items": items, "minItems": lo, "maxItems": hi}


def enum(*values):
    return {"enum": list(values)}


def nullable(schema):
    return {"anyOf": [{"type": "null"}, schema]}


def ref(name):
    return {"$ref": "#/$defs/" + name}


S = {"type": "string", "minLength": 1, "maxLength": 512}
ID = {"type": "string", "minLength": 1, "maxLength": 128}
N = {"type": "number", "minimum": 0}
I = {"type": "integer", "minimum": 1}
B = {"type": "boolean"}
P = {"type": "number", "minimum": 0, "maximum": 1}
HEX = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
ROLES = ("TASK", "INTENT_QUERY", "PROBE", "CONFIRM")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def schemas():
    d = {}
    d["context"] = obj({"session_id": ID, "scene_id": ID, "scene_revision": I, "mapping_id": ID, "mapping_revision": I})
    d["task"] = obj({"task_id": ID, "task_revision": I, "suggestion_id": ID, "context": ref("context"),
        "action": enum("pick", "place"), "source_id": ID, "destination_id": nullable(ID),
        "valid_until_s": N, "capability_verified": {"const": False}})
    d["option"] = obj({"option_id": ID, "symbol": ID, "kind": enum("OBJECT", "TASK", "DIAGNOSTIC", "DECISION"), "reference": ID})
    d["question"] = obj({"question_id": ID, "role": enum(*ROLES), "context": ref("context"), "layout_id": ID,
        "issued_at_s": N, "expires_at_s": N, "presentation_ack": B, "options": arr(ref("option"), 1, 16),
        "diagnostic_option_id": nullable(ID), "proposal": nullable(ref("task"))})
    d["calibration"] = obj({"artifact_id": ID, "provenance": enum("synthetic_fixture"), "fit_split": {"const": "calibration-fit"},
        "mapping_id": ID, "mapping_revision": I, "layout_id": ID, "roles": arr(enum(*ROLES), 1, 4), "valid_until_s": N})
    d["evidence"] = obj({"score_kind": enum("RAW_SCORE", "CALIBRATED_PROBABILITY", "HARD_LABEL", "NONE"),
        "values": arr(obj({"option_id": ID, "value": {"type": "number"}}), 0, 16),
        "selected_option_id": nullable(ID), "calibration": nullable(ref("calibration"))})
    d["event"] = obj({"event_id": ID, "sequence": I, "role": enum(*ROLES, "REJECT", "NO_RESPONSE", "CANCEL", "STOP"),
        "original_role": nullable(enum(*ROLES)), "question": nullable(ref("question")), "started_at_s": N, "ended_at_s": N,
        "valid": B, "invalid_reason": nullable(enum("EXPIRED", "STALE_CONTEXT", "PRESENTATION_UNACKNOWLEDGED")),
        "evidence": nullable(ref("evidence"))})
    d["object"] = obj({"instance_id": ID, "alias": {"type": "string", "pattern": "^OBJ_([0-9]|1[0-5])$"},
        "label": S, "bbox_xyxy": nullable(arr(P, 4, 4)), "object_valid": B,
        "roi_feature": nullable(arr({"type": "number"}, 1, 8192)), "roi_valid": B,
        "semantic_feature": nullable(arr({"type": "number"}, 1, 8192)), "semantic_valid": B})
    d["diagnostics"] = obj({"context": ref("context"), "layout_id": ID, "window_s": {"const": 30}, "ttl_s": {"const": 15},
        "min_attempts": {"const": 3}, "min_covered_symbols": {"const": 2}, "receipts": arr(ref("event"), 0, 64),
        "attempt_count": {"type": "integer", "minimum": 0}, "correct_count": {"type": "integer", "minimum": 0},
        "no_response_count": {"type": "integer", "minimum": 0}, "covered_symbols": arr(ID, 0, 16),
        "latest_ended_at_s": nullable(N), "status": enum("SUFFICIENT", "INSUFFICIENT", "EXPIRED"), "truncated": B})
    d["state"] = obj({"confirmed_task": nullable(obj({"task": ref("task"), "confirmation_receipt": ref("event")})),
        "pending_suggestion": nullable(ref("task")), "pending_question": nullable(ref("question")),
        "diagnostics": ref("diagnostics"), "stop_latched": B,
        "execution_authorized": {"const": False}})
    d["public_input"] = obj({"record_type": {"const": "PUBLIC_INPUT"}, "schema_version": {"const": "bci-vla-interaction/4.0"},
        "sample_id": ID, "context": ref("context"), "as_of_s": N,
        "provenance": obj({"bci": enum("synthetic_fixture"), "vision": enum("annotated_fixture", "privileged_simulator", "detector_recording"),
            "feature_profile_id": ID, "real_eeg": {"const": False}, "physical_robot": {"const": False}}),
        "scene": obj({"observed_at_s": N, "image_ref": nullable(S), "image_valid": B, "roi_dim": I, "semantic_dim": I,
            "objects": arr(ref("object"), 0, 16), "truncation": obj({"original_count": {"type": "integer", "minimum": 0},
                "retained_count": {"type": "integer", "minimum": 0, "maximum": 16}, "policy": enum("none", "public_detector_order"),
                "target_recall_known": {"const": False}})}),
        "history": arr(ref("event"), 0, 8), "current_event_id": nullable(ID), "selection_event_id": nullable(ID),
        "task_options": arr(ref("task"), 0, 32), "public_state": ref("state")})
    d["private_labels"] = obj({"record_type": {"const": "PRIVATE_LABELS"}, "schema_version": {"const": "bci-vla-interaction/4.0"},
        "sample_id": ID, "variant_id": ID, "public_input_sha256": HEX, "split": {"const": "development_fixture"},
        "hidden_truth": obj({"goal": nullable(ref("task")), "channel_injected_mode": ID, "changed_at_s": nullable(N)}),
        "identifiability": obj({"intent": enum("IDENTIFIED", "AMBIGUOUS", "UNOBSERVABLE"), "channel": enum("SUPPORTED", "INSUFFICIENT")}),
        "labels": obj({
            "intent": obj({"mode": enum("POINT", "SET", "NONE"), "candidate_ids": arr(ID, 0, 16), "outside": B, "mask": B}),
            "main_task": obj({"mode": enum("TASK", "CLARIFICATION", "NONE"), "task": nullable(ref("task")), "text": nullable(S), "mask": B}),
            "channel": obj({"mask": B, "next_probe_success": nullable(B), "target_time_s": nullable(N)}),
            "interaction": obj({"mask": B, "action": enum("PROPOSE_TASK", "ASK_INTENT", "CHECK_CHANNEL", "WAIT"),
                "teacher": enum("public_rule_v1", "annotator_visible_history"), "teacher_input_sha256": HEX})})})
    d["module_output"] = obj({"record_type": {"const": "MODULE_OUTPUT"}, "schema_version": {"const": "bci-vla-interaction/4.0"},
        "context": ref("context"), "source_input_sha256": HEX, "as_of_s": N,
        "provenance": {"const": "synthetic_fixture_not_model_prediction"},
        "soft_prefix": obj({"dtype": {"const": "float32"}, "shape": arr(I, 2, 2), "tokens": arr(arr({"type": "number"}, 1, 8192), 8, 8),
            "intent_slots": {"const": 4}, "diagnostic_summary_slots": {"const": 4}, "raw_channel_queries_included": {"const": False}}),
        "intent": obj({"candidate_distribution": arr(obj({"instance_id": ID, "probability": P}), 0, 16), "outside_probability": P,
            "allowed_candidate_ids": arr(ID, 0, 16), "abstain": B, "missing_slots": arr(enum("ACTION", "SOURCE", "DESTINATION"), 0, 3)}),
        "channel": obj({"status": enum("SUFFICIENT", "INSUFFICIENT", "EXPIRED"), "probability": nullable(P),
            "probability_calibrated": {"const": False}, "target_layout_id": ID}),
        "task_proposal": nullable(ref("task")), "interaction": obj({"action": enum("PROPOSE_TASK", "ASK_INTENT", "CHECK_CHANNEL", "WAIT"),
            "question": nullable(ref("question")), "reason_code": enum("ambiguous_target", "missing_task_slot", "diagnostic_evidence_insufficient", "target_outside_candidates", "awaiting_reply", "stop_latched", "proposal_ready")}),
        "control": obj({"execution_authorized": {"const": False}, "requires_confirmation": {"const": True},
            "stop_route": {"const": "existing_control_path"}, "raw_robot_command": {"type": "null"}})})
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "urn:bci-vla:interaction:4.0",
        "title": "V4-02 synthetic public/private/output interface contract", "$defs": d,
        "oneOf": [ref("public_input"), ref("private_labels"), ref("module_output")],
        "x-protocol": {
            "version": "V4-02-contract-1", "scope": "synthetic schema fixtures only; real EEG, physical actions and trained outputs require a later version",
            "prototype_defaults_not_empirically_optimal": True,
            "limits": {"objects": 16, "recent_events_total": 8, "diagnostic_receipts": 64, "prefix_tokens": 8},
            "tensor_contract": {"roi": ["B", 16, "D_roi"], "semantic": ["B", 16, "D_sem"], "boxes": ["B", 16, 4],
                "candidate_mask": ["B", 16], "roi_mask": ["B", 16], "semantic_mask": ["B", 16],
                "task_history": ["B", 8, "D_task"], "diagnostic_history": ["B", 8, "D_diag"],
                "combined_real_history_count_max": 8, "history_valid_masks": ["B", 8],
                "prefix": ["B", 8, "D_vlm"], "D_vlm_verified": False,
                "padding_rule": "explicit PAD or EMPTY_INPUT plus mask, never an outside/ambiguous label", "fixture_dimensions": {"D_roi": 4, "D_sem": 3, "D_vlm": 6}},
            "model_feature_allowlist": ["scene features/boxes/aliases and masks", "valid role-separated past events", "valid public session state", "deidentified diagnostic summary"],
            "model_input_excludes": ["sample_id", "split", "private_labels", "hidden_truth", "future answers", "injected schedules", "raw PROBE object semantics in channel branch", "raw channel query tokens at VLM"],
            "score_kinds": {"RAW_SCORE": "finite raw values, never renamed probabilities", "CALIBRATED_PROBABILITY": "normalized probabilities plus role/layout/mapping-scoped fixture calibration", "HARD_LABEL": "single declared selection, not probability calibration", "NONE": "no decoded observation"},
            "event_effects": {"TASK": "candidate evidence only", "INTENT_QUERY": "clarify task candidates, cannot authorize", "PROBE": "diagnostic summary only, never replace confirmed task", "CONFIRM": "noisy observed accept of exact task; separate existing authorization still required", "REJECT": "reject bound pending suggestion, pause, no automatic old-task resume", "NO_RESPONSE": "retain original question role; no task update; eligible timed-out PROBE is diagnostic nonresponse", "CANCEL": "clear pending/confirmed task and request queue/cache invalidation", "STOP": "existing direct stop path and latch, independent of neural/VLM"},
            "diagnostic_estimand": "Next known-target attempt succeeds under the same mapping/layout/covered conditions and time window; eligible timed-out no-response counts unsuccessful, not a task choice",
            "diagnostic_policy": {"window_s": 30, "ttl_s": 15, "min_attempts": 3, "min_covered_symbols": 2, "overflow": "mark truncated and insufficient; never selectively retain by hidden target"},
            "state_invalidation": {"mapping_change": "drop pending question/suggestion/confirmation and diagnostic applicability; pause/reconfirm, keep old records as invalid audit only", "scene_instance_or_binding_change": "invalidate affected task/aliases/authorization, clear queue/cache before new confirmation", "probe": "no confirmed-task mutation", "cancel_or_stop": "clear task and pending state; STOP stays latched", "completed_task": "clear current task, retain public audit"},
            "task_grammar_status": "pick/place are syntactic development fixtures; backend capability remains false pending V4-03",
            "legacy_boundary": {"schema_compatible_with_v2_event": False, "role_mapping": {"TASK": "TASK_QUERY", "INTENT_QUERY": "TASK_QUERY with distinct question provenance", "PROBE": "CHANNEL_PROBE", "CONFIRM": "CONFIRM"}, "adapter_implementation_step": "V4-11", "authority": "existing V2 authorization/STOP must remain independent"},
            "validator_scope": "stdlib reference checks the generated schema subset plus cross-field invariants; not a general-purpose JSON Schema implementation; does not prove feature independence or real calibration"
        }}


CTX = {"session_id": "fixture-session", "scene_id": "table-1", "scene_revision": 1, "mapping_id": "screen-map", "mapping_revision": 1}


def task(source="cup-blue", action="pick", destination=None, name="task-blue"):
    return {"task_id": name, "task_revision": 1, "suggestion_id": "suggestion-" + name, "context": deepcopy(CTX),
            "action": action, "source_id": source, "destination_id": destination, "valid_until_s": 120.0, "capability_verified": False}


def question(name="q-task", role="TASK", issued=4.0, expiry=7.0, target=None, proposal=None):
    if role == "CONFIRM":
        options = [{"option_id": "accept", "symbol": "s0", "kind": "DECISION", "reference": "accept"}, {"option_id": "reject", "symbol": "s1", "kind": "DECISION", "reference": "reject"}]
    else:
        options = [{"option_id": "o" + str(i), "symbol": "s" + str(i), "kind": "DIAGNOSTIC" if role == "PROBE" else "OBJECT", "reference": oid} for i,oid in enumerate(("cup-blue", "cup-red", "tray"))]
    return {"question_id": name, "role": role, "context": deepcopy(CTX), "layout_id": "confirm-2" if role == "CONFIRM" else ("probe-3" if role == "PROBE" else "task-3"),
        "issued_at_s": issued, "expires_at_s": expiry, "presentation_ack": True, "options": options,
        "diagnostic_option_id": target, "proposal": proposal}


def event(name="e-task", role="TASK", seq=1, start=4.2, end=5.0, q=None, selected="o0"):
    q = deepcopy(q or question())
    options = q["options"]
    return {"event_id": name, "sequence": seq, "role": role, "original_role": None, "question": q,
        "started_at_s": start, "ended_at_s": end, "valid": True, "invalid_reason": None,
        "evidence": {"score_kind": "RAW_SCORE", "values": [{"option_id": x["option_id"], "value": 0.8 if x["option_id"] == selected else 0.2} for x in options], "selected_option_id": selected, "calibration": None}}


def diagnostics(receipts=(), as_of=12.0):
    receipts = deepcopy(list(receipts)); count = len(receipts)
    good = sum(e["role"] != "NO_RESPONSE" and e["evidence"]["selected_option_id"] == e["question"]["diagnostic_option_id"] for e in receipts)
    covered = sorted({next(o["symbol"] for o in e["question"]["options"] if o["option_id"] == e["question"]["diagnostic_option_id"]) for e in receipts})
    latest = max((e["ended_at_s"] for e in receipts), default=None)
    status = "EXPIRED" if latest is not None and as_of-latest > 15 else ("SUFFICIENT" if count >= 3 and len(covered) >= 2 else "INSUFFICIENT")
    return {"context": deepcopy(CTX), "layout_id": "probe-3", "window_s": 30, "ttl_s": 15, "min_attempts": 3, "min_covered_symbols": 2,
        "receipts": receipts, "attempt_count": count, "correct_count": int(good), "no_response_count": sum(e["role"] == "NO_RESPONSE" for e in receipts),
        "covered_symbols": covered, "latest_ended_at_s": latest, "status": status, "truncated": False}


def public():
    objects = [{"instance_id": oid, "alias": "OBJ_" + str(i), "label": label, "bbox_xyxy": box, "object_valid": True,
                "roi_feature": [0.1+i, 0.2, 0.3, 0.4], "roi_valid": True, "semantic_feature": [0.2,0.3,0.4], "semantic_valid": True}
               for i,(oid,label,box) in enumerate((("cup-blue","cup",[0.1,0.2,0.3,0.6]),("cup-red","cup",[0.4,0.2,0.6,0.6]),("tray","tray",[0.7,0.3,0.95,0.8])))]
    task_event = event()
    probe = event("e-probe", "PROBE", 2, 8.1, 9.0, question("q-probe", "PROBE", 8.0, 10.0, "o1"), "o1")
    return {"record_type": "PUBLIC_INPUT", "schema_version": "bci-vla-interaction/4.0", "sample_id": "fixture-001", "context": deepcopy(CTX), "as_of_s": 12.0,
        "provenance": {"bci": "synthetic_fixture", "vision": "annotated_fixture", "feature_profile_id": "fixture-4x3", "real_eeg": False, "physical_robot": False},
        "scene": {"observed_at_s": 11.0, "image_ref": None, "image_valid": False, "roi_dim": 4, "semantic_dim": 3, "objects": objects,
            "truncation": {"original_count": 3, "retained_count": 3, "policy": "none", "target_recall_known": False}},
        "history": [task_event, probe], "current_event_id": "e-probe", "selection_event_id": "e-task", "task_options": [],
        "public_state": {"confirmed_task": None, "pending_suggestion": None, "pending_question": None, "diagnostics": diagnostics([probe]), "stop_latched": False, "execution_authorized": False}}


def labels(p):
    sha = digest(p)
    return {"record_type": "PRIVATE_LABELS", "schema_version": "bci-vla-interaction/4.0", "sample_id": p["sample_id"], "variant_id": "hidden-blue",
        "public_input_sha256": sha, "split": "development_fixture", "hidden_truth": {"goal": task(), "channel_injected_mode": "synthetic_noise", "changed_at_s": None},
        "identifiability": {"intent": "AMBIGUOUS", "channel": "INSUFFICIENT"}, "labels": {
            "intent": {"mode": "SET", "candidate_ids": ["cup-blue", "cup-red"], "outside": False, "mask": True},
            "main_task": {"mode": "CLARIFICATION", "task": None, "text": "Which cup and what task?", "mask": True},
            "channel": {"mask": False, "next_probe_success": None, "target_time_s": None},
            "interaction": {"mask": True, "action": "ASK_INTENT", "teacher": "public_rule_v1", "teacher_input_sha256": sha}}}


def output(p):
    q = question("q-next", "INTENT_QUERY", p["as_of_s"], p["as_of_s"]+4)
    return {"record_type": "MODULE_OUTPUT", "schema_version": "bci-vla-interaction/4.0", "context": deepcopy(p["context"]), "source_input_sha256": digest(p), "as_of_s": p["as_of_s"],
        "provenance": "synthetic_fixture_not_model_prediction",
        "soft_prefix": {"dtype": "float32", "shape": [8,6], "tokens": [[0.0]*6 for _ in range(8)], "intent_slots": 4, "diagnostic_summary_slots": 4, "raw_channel_queries_included": False},
        "intent": {"candidate_distribution": [{"instance_id": x["instance_id"], "probability": v} for x,v in zip(p["scene"]["objects"],(0.5,0.5,0.0))], "outside_probability": 0.0,
            "allowed_candidate_ids": ["cup-blue","cup-red"], "abstain": True, "missing_slots": ["ACTION"]},
        "channel": {"status": p["public_state"]["diagnostics"]["status"], "probability": None, "probability_calibrated": False, "target_layout_id": "probe-3"},
        "task_proposal": None, "interaction": {"action": "ASK_INTENT", "question": q, "reason_code": "ambiguous_target"},
        "control": {"execution_authorized": False, "requires_confirmation": True, "stop_route": "existing_control_path", "raw_robot_command": None}}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def main():
    CONFIG.mkdir(parents=True, exist_ok=True); EXAMPLES.mkdir(parents=True, exist_ok=True)
    assert not (CONFIG/"interfaces.json").exists(), "Preserve frozen interfaces"
    p = public()
    write(CONFIG/"interfaces.json", schemas())
    write(EXAMPLES/"public_input.json", p)
    write(EXAMPLES/"private_labels.json", labels(p))
    write(EXAMPLES/"module_output.json", output(p))
    other = labels(p); other["variant_id"] = "hidden-red"; other["hidden_truth"]["goal"] = task("cup-red", name="task-red")
    write(EXAMPLES/"private_labels_alias_variant.json", other)
    fixtures = {}
    empty = public(); empty["sample_id"] = "empty-input"; empty["scene"]["objects"] = []; empty["scene"]["truncation"].update(original_count=0,retained_count=0)
    empty["history"] = []; empty["current_event_id"] = None; empty["selection_event_id"] = None; empty["public_state"]["diagnostics"] = diagnostics()
    fixtures["empty_scene_and_history"] = empty
    missing = public(); missing["scene"]["objects"][0].update(roi_feature=None,roi_valid=False); fixtures["missing_roi_feature"] = missing
    noresponse = public(); e = noresponse["history"][-1]; e.update(role="NO_RESPONSE",original_role="PROBE",ended_at_s=10.0)
    e["evidence"] = {"score_kind":"NONE","values":[],"selected_option_id":None,"calibration":None}
    noresponse["public_state"]["diagnostics"] = diagnostics([e]); fixtures["no_response_retains_probe_role"] = noresponse
    mapped = public(); mapped["context"]["mapping_revision"] = 2
    for e in mapped["history"]: e.update(valid=False,invalid_reason="STALE_CONTEXT")
    mapped["selection_event_id"] = None; mapped["public_state"]["diagnostics"] = diagnostics(); mapped["public_state"]["diagnostics"]["context"] = deepcopy(mapped["context"])
    fixtures["mapping_change_invalidates_old_evidence"] = mapped
    for role in ("STOP","CANCEL"):
        stopped = public(); stopped["history"] = [{"event_id":"e-control","sequence":3,"role":role,"original_role":None,"question":None,"started_at_s":11.5,"ended_at_s":11.5,"valid":True,"invalid_reason":None,"evidence":None}]
        stopped["current_event_id"] = "e-control"; stopped["selection_event_id"] = None; stopped["public_state"]["diagnostics"] = diagnostics(); stopped["public_state"]["stop_latched"] = role=="STOP"
        fixtures[role.lower()+"_clears_task"] = stopped
    persistent = public(); t = task(); confirm = event("e-confirm", "CONFIRM", 1, 1.1, 1.5, question("q-confirm", "CONFIRM", 1.0, 2.0, proposal=t), "accept")
    probes = [event("p"+str(i),"PROBE",i+2,2.1+i,2.5+i,question("qp"+str(i),"PROBE",2.0+i,3.0+i,"o"+str(i%2)),"o"+str(i%2)) for i in range(9)]
    persistent["history"] = probes[-8:]; persistent["current_event_id"] = probes[-1]["event_id"]; persistent["selection_event_id"] = None
    persistent["public_state"]["confirmed_task"] = {"task":t,"confirmation_receipt":confirm}; persistent["public_state"]["diagnostics"] = diagnostics(probes)
    fixtures["confirmed_task_survives_nine_probes"] = persistent
    reordered = public(); reordered["scene"]["objects"].reverse(); fixtures["object_reorder_with_alias_binding"] = reordered
    for name, value in fixtures.items(): write(EXAMPLES/(name+".json"),value)
    write(EXAMPLES/"fixture_index.json", {"scope":"synthetic contract examples, not training data or model predictions", "valid_public_inputs":["public_input.json"]+[k+".json" for k in fixtures], "private_labels":["private_labels.json","private_labels_alias_variant.json"], "module_output":"module_output.json"})
    print(json.dumps({"schema":str(CONFIG/"interfaces.json"),"valid_public_fixtures":len(fixtures)+1,"scope":"contracts only"}))


if __name__ == "__main__":
    main()
