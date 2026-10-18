"""Run local V4-02 contract checks. No models, device calls or network."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bci_vla.interaction_contract_v4 import (
    ContractError, canonical_digest, control_route, load_json, neural_view,
    validate_document, validate_transition,
)


def main():
    folder = ROOT / "examples/v4"
    index = load_json(folder / "fixture_index.json")
    base = load_json(folder / "public_input.json")
    labels = load_json(folder / "private_labels.json")
    output = load_json(folder / "module_output.json")
    checks = []

    def check(name, function, reject=False):
        try:
            result = function()
            passed = not reject and result is not False
            detail = "unexpected acceptance" if reject else "accepted"
        except ContractError as exc:
            passed = reject
            detail = str(exc)
        except Exception as exc:
            passed = False
            detail = type(exc).__name__ + ": " + str(exc)
        checks.append({"name": name, "passed": passed, "detail": detail})

    def mutate(name, source, operation, bound=None):
        value = deepcopy(source)
        operation(value)
        check(name, lambda: validate_document(value, bound), reject=True)

    for name in index["valid_public_inputs"]:
        check("valid:" + name, lambda name=name: validate_document(load_json(folder/name)))
        check("neural_view:" + name, lambda name=name: isinstance(neural_view(load_json(folder/name)), dict))
    for name in index["private_labels"]:
        check("valid:" + name, lambda name=name: validate_document(load_json(folder/name), base))
    check("valid:module_output", lambda: validate_document(output, base))
    for name in index.get("additional_module_outputs",[]):
        check("valid:"+name, lambda name=name:validate_document(load_json(folder/name),base))
    mutate("reject:hidden_label_in_public", base, lambda p:p.update(hidden_truth={}))
    mutate("reject:hidden_label_in_object", base, lambda p:p["scene"]["objects"][0].update(future_goal=True))
    mutate("reject:probability_without_calibration", base, lambda p:p["history"][0]["evidence"].update(score_kind="CALIBRATED_PROBABILITY"))
    calibrated = deepcopy(base)
    ev = calibrated["history"][0]["evidence"]
    ev.update(score_kind="CALIBRATED_PROBABILITY", calibration={"artifact_id":"fixture-cal", "provenance":"synthetic_fixture", "fit_split":"calibration-fit", "mapping_id":"screen-map", "mapping_revision":1, "layout_id":"task-3", "roles":["TASK"], "valid_until_s":20.0})
    for row, score in zip(ev["values"], [.6,.3,.1]):
        row["value"] = score
    check("valid:scoped_probability_fixture", lambda:validate_document(calibrated))
    for key, wrong in [("roles",["PROBE"]),("mapping_revision",2),("layout_id","wrong"),("valid_until_s",1.0)]:
        mutate("reject:calibration_"+key, calibrated, lambda p,key=key,wrong=wrong:p["history"][0]["evidence"]["calibration"].update({key:wrong}))
    mutate("reject:probability_mass", calibrated, lambda p:p["history"][0]["evidence"]["values"][0].update(value=.9))
    hard = deepcopy(base)
    hard["history"][0]["evidence"]["score_kind"] = "HARD_LABEL"
    for i,row in enumerate(hard["history"][0]["evidence"]["values"]): row["value"] = int(i == 0)
    check("valid:hard_label_not_probability", lambda:validate_document(hard))
    mutate("reject:hard_label_fraction", hard, lambda p:p["history"][0]["evidence"]["values"][0].update(value=.9))
    mutate("reject:duplicate_alias", base, lambda p:p["scene"]["objects"][1].update(alias="OBJ_0"))
    mutate("reject:duplicate_instance", base, lambda p:p["scene"]["objects"][1].update(instance_id="cup-blue"))
    mutate("reject:box_order", base, lambda p:p["scene"]["objects"][0].update(bbox_xyxy=[.5,.2,.1,.8]))
    mutate("reject:missing_feature_mask", base, lambda p:p["scene"]["objects"][0].update(roi_feature=None))
    mutate("reject:feature_dimension", base, lambda p:p["scene"]["objects"][0].update(roi_feature=[1]))
    mutate("reject:feature_nan", base, lambda p:p["scene"]["objects"][0]["roi_feature"].__setitem__(0,float("nan")))
    mutate("reject:too_many_objects", base, lambda p:p["scene"].update(objects=p["scene"]["objects"]*6))
    mutate("reject:invented_truncation", base, lambda p:p["scene"]["truncation"].update(original_count=4))
    mutate("reject:duplicate_event", base, lambda p:p["history"].append(deepcopy(p["history"][-1])))
    mutate("reject:history_overflow", base, lambda p:p.update(history=p["history"]*5))
    mutate("reject:current_event_not_last", base, lambda p:p.update(current_event_id="e-task"))
    mutate("reject:future_event", base, lambda p:p["history"][0].update(ended_at_s=99))
    mutate("reject:expired_response_claimed_valid", base, lambda p:p["history"][0].update(ended_at_s=7.5))
    mutate("reject:unacknowledged_presentation", base, lambda p:p["history"][0]["question"].update(presentation_ack=False))
    mutate("reject:stale_mapping", base, lambda p:p["context"].update(mapping_revision=2))
    mutate("reject:probe_as_task_selection", base, lambda p:p.update(selection_event_id="e-probe"))
    mutate("reject:invented_diagnostic_counts", base, lambda p:p["public_state"]["diagnostics"].update(correct_count=9))
    mutate("reject:one_probe_sufficient", base, lambda p:p["public_state"]["diagnostics"].update(status="SUFFICIENT"))
    noresponse = load_json(folder/"no_response_retains_probe_role.json")
    mutate("reject:missing_original_role", noresponse, lambda p:p["history"][-1].update(original_role=None))
    mutate("reject:no_response_selected", noresponse, lambda p:p["history"][-1]["evidence"].update(selected_option_id="o0"))
    late = deepcopy(noresponse)
    late["history"][-1]["ended_at_s"] = 10.2
    late["public_state"]["diagnostics"]["receipts"][0]["ended_at_s"] = 10.2
    late["public_state"]["diagnostics"]["latest_ended_at_s"] = 10.2
    check("valid:timeout_recorded_after_deadline", lambda:validate_document(late))
    persistent = load_json(folder/"confirmed_task_survives_nine_probes.json")
    mutate("reject:confirmation_task_version_mismatch", persistent, lambda p:p["public_state"]["confirmed_task"]["task"].update(task_revision=2))
    mutate("reject:future_confirmation_receipt", persistent, lambda p:p["public_state"]["confirmed_task"]["confirmation_receipt"].update(ended_at_s=99))
    rejected=load_json(folder/"reject_clears_task.json")
    check("transition:reject_clears_task",lambda:validate_transition(persistent,rejected))
    mutate("reject:reject_kept_confirmed_task",rejected,lambda p:p["public_state"].update(confirmed_task=deepcopy(persistent["public_state"]["confirmed_task"])))
    stale_summary = deepcopy(persistent); stale_summary["as_of_s"] = 27
    mutate("reject:expired_diagnostics_claim_sufficient", stale_summary, lambda p:None)
    stale_summary["public_state"]["diagnostics"]["status"] = "EXPIRED"
    check("valid:expired_summary_no_probability", lambda:validate_document(stale_summary))
    pending = deepcopy(base)
    pending["public_state"]["pending_question"] = deepcopy(output["interaction"]["question"])
    check("valid:pending_question", lambda:validate_document(pending))
    mutate("reject:expired_pending_question", pending, lambda p:p["public_state"]["pending_question"].update(expires_at_s=12))
    mutate("reject:answered_pending_question", base, lambda p:p["public_state"].update(pending_question=deepcopy(p["history"][0]["question"])))
    for role in ("stop", "cancel"):
        control = load_json(folder/(role+"_clears_task.json"))
        check("valid:direct_"+role, lambda control=control:control_route(control["history"][-1]).endswith("PATH"))
        malformed = deepcopy(control["history"][-1]); malformed["original_role"] = "TASK"
        check("reject:control_original_role_"+role, lambda malformed=malformed:control_route(malformed), reject=True)
        mutate("reject:control_kept_task_"+role, control, lambda p:p["public_state"].update(confirmed_task=deepcopy(persistent["public_state"]["confirmed_task"])))
    stopped = load_json(folder/"stop_clears_task.json")
    mutate("reject:stop_not_latched", stopped, lambda p:p["public_state"].update(stop_latched=False))
    mutate("reject:private_binding", labels, lambda p:p.update(public_input_sha256="0"*64), base)
    mutate("reject:ambiguous_point_label", labels, lambda p:p["labels"]["intent"].update(mode="POINT",candidate_ids=["cup-blue"]), base)
    mutate("reject:unobservable_intent_supervision", labels, lambda p:p["identifiability"].update(intent="UNOBSERVABLE"), base)
    mutate("reject:unsupported_channel_target", labels, lambda p:p["labels"]["channel"].update(mask=True,next_probe_success=True,target_time_s=13), base)
    supported=deepcopy(labels)
    supported["public_input_sha256"]=canonical_digest(persistent)
    supported["labels"]["interaction"]["teacher_input_sha256"]=canonical_digest(persistent)
    supported["identifiability"]["channel"]="SUPPORTED"
    supported["labels"]["channel"].update(mask=True,next_probe_success=True,target_time_s=13)
    check("valid:supported_future_channel_label_private_only",lambda:validate_document(supported,persistent))
    mutate("reject:channel_label_outside_horizon",supported,lambda p:p["labels"]["channel"].update(target_time_s=99),persistent)
    mutate("reject:teacher_binding", labels, lambda p:p["labels"]["interaction"].update(teacher_input_sha256="0"*64), base)
    mutate("reject:output_binding", output, lambda p:p.update(source_input_sha256="0"*64), base)
    mutate("reject:prefix_shape", output, lambda p:p["soft_prefix"].update(shape=[8,7]), base)
    mutate("reject:raw_channel_queries", output, lambda p:p["soft_prefix"].update(raw_channel_queries_included=True), base)
    mutate("reject:intent_probability_mass", output, lambda p:p["intent"].update(outside_probability=.5), base)
    mutate("reject:candidate_axis", output, lambda p:p["intent"]["candidate_distribution"].pop(), base)
    mutate("reject:unsupported_reliability", output, lambda p:p["channel"].update(probability=.9), base)
    mutate("reject:wrong_clarification_role", output, lambda p:p["interaction"]["question"].update(role="TASK"), base)
    mutate("reject:module_execution_authority", output, lambda p:p["control"].update(execution_authorized=True), base)
    mutate("reject:raw_robot_command", output, lambda p:p["control"].update(raw_robot_command=[0,1]), base)
    pending_output = deepcopy(output); pending_output["source_input_sha256"] = canonical_digest(pending)
    check("reject:second_question_before_answer", lambda:validate_document(pending_output,pending), reject=True)
    check("transition:identity", lambda:validate_transition(persistent,persistent))
    check("transition:mapping_invalidation", lambda:validate_transition(base,load_json(folder/"mapping_change_invalidates_old_evidence.json")))
    check("reject:probe_cleared_confirmed_task", lambda:validate_transition(persistent,base), reject=True)
    check("reject:stop_unlatched", lambda:validate_transition(stopped,base), reject=True)
    check("reject:new_task_without_confirmation", lambda:validate_transition(base,persistent), reject=True)
    reordered = deepcopy(base); reordered["history"][0]["evidence"]["values"].reverse()
    check("invariance:score_row_reordering", lambda:neural_view(base)==neural_view(reordered))
    check("invariance:object_row_reordering", lambda:neural_view(base)==neural_view(load_json(folder/"object_reorder_with_alias_binding.json")))
    scrubbed = deepcopy(base)
    for e in [scrubbed["history"][-1],scrubbed["public_state"]["diagnostics"]["receipts"][0]]:
        for o in e["question"]["options"]: o["reference"] = "different-diagnostic-object"
    check("invariance:probe_object_identity_removed", lambda:neural_view(base)==neural_view(scrubbed))
    other_labels = load_json(folder/"private_labels_alias_variant.json")
    check("invariance:hidden_goal_changes_not_training_target", lambda: labels["labels"]==other_labels["labels"] and labels["public_input_sha256"]==other_labels["public_input_sha256"])
    renamed = deepcopy(base)
    swaps = {"OBJ_0":"OBJ_1","OBJ_1":"OBJ_0","OBJ_2":"OBJ_2"}
    for o in renamed["scene"]["objects"]: o["alias"] = swaps[o["alias"]]
    view = neural_view(renamed)
    check("equivariance:alias_renaming_tracks_selection", lambda:view["selection"]["selected_alias"]==swaps[neural_view(base)["selection"]["selected_alias"]] and next(o for o in view["objects"] if o["alias"]=="OBJ_1")["roi_feature"]==base["scene"]["objects"][0]["roi_feature"])
    check("allowlist:no_sample_id", lambda:base["sample_id"] not in json.dumps(neural_view(base)))
    for row in load_json(ROOT/"execution/v4/02_before_manifest.json")["files"]:
        check("preserved:"+row["path"], lambda row=row:hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()==row["sha256"])
    report = {"step":"V4-02", "scope":"synthetic contract/software checks only; no model inference, training or real devices", "status":"PASS" if all(c["passed"] for c in checks) else "FAIL", "checks":len(checks), "passed":sum(c["passed"] for c in checks), "failed":sum(not c["passed"] for c in checks), "results":checks}
    (ROOT/"execution/v4/02_verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="results"},ensure_ascii=False))
    for c in checks:
        if not c["passed"]: print(json.dumps(c,ensure_ascii=False))
    return 0 if report["status"]=="PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
