"""Reuse legacy grammar, while keeping candidate intent separate from capability."""
import hashlib

from ..protocols import fields, freeze, integer, json_object, require, text, validate_intent, TaskContract, ExecutionEvent
from ._legacy.planning import candidate_key, generate_candidates, structural_clarification
from ._legacy.rendering import render_instruction
from .state import IdentityLedger


class BackendCapabilities:
    def __init__(self, config):
        v=json_object(config)
        fields(v,{"executor_id","checkpoint_ref","status","predicate_version","tasks"})
        text(v["executor_id"]); text(v["checkpoint_ref"],nullable=True); text(v["predicate_version"])
        require(v["status"] in {"unverified","simulation_only"},"real_backend_not_validated")
        require(isinstance(v["tasks"],dict),"invalid_capabilities")
        for action,row in v["tasks"].items():
            require(action in {"pick","place","fold"},"unsupported_action")
            fields(row,{"support","endpoint_predicate","evidence_ref"})
            require(row["support"] in {"unknown","unsupported","simulation_only"},"real_backend_not_validated")
            require(row["endpoint_predicate"] == {"pick":"holding","place":"inside","fold":"folded"}[action],"predicate_action_mismatch")
            text(row["evidence_ref"],nullable=True)
            require(row["support"] != "simulation_only" or (v["status"] == "simulation_only" and row["evidence_ref"] is not None),"missing_simulation_provenance")
        self.data=freeze(v)

    def assess(self,intent,*,execution_mode="offline"):
        validate_intent(intent)
        row=self.data["tasks"].get(intent["action"])
        if row is None or row["support"] == "unsupported": return {"eligibility":"unsupported","reason":"backend_action_not_supported"}
        if execution_mode != "offline" or row["support"] == "unknown": return {"eligibility":"unknown","reason":"actual_backend_capability_unverified"}
        return {"eligibility":"simulation_only","reason":"offline_declared_capability"}


def goal_id(intent):
    return "task-"+hashlib.sha256(candidate_key(intent).encode()).hexdigest()[:24]


def task_candidates(scene,backend):
    # Grammar is determined by visible objects and declared affordances, not ease.
    result=[{"goal_id":"unknown","kind":"unknown","intent":None,"capability":{"eligibility":"unknown","reason":"intent_unresolved"}}, {"goal_id":"none","kind":"none","intent":None,"capability":{"eligibility":"not_applicable","reason":"explicit_no_task_hypothesis"}}]
    for intent in generate_candidates(scene.legacy_observation()):
        result.append({"goal_id":goal_id(intent),"kind":"task","intent":intent,"capability":backend.assess(intent)})
    return result


def clarify_slots(scene,slots):
    """Slots are an explicit current hypothesis, never an inferred history lock."""
    fields(slots,{"action","source_id","destination_id"})
    for key in ["source_id","destination_id"]: text(slots[key],nullable=True)
    require(slots["action"] is None or slots["action"] in {"pick","place","fold"},"unsupported_action")
    obs=scene.legacy_observation()
    visible={o["id"] for o in obs["objects"]}
    if any(slots[k] is not None and slots[k] not in visible for k in ["source_id","destination_id"]):
        return {"status":"clarify","slot":"source" if slots["source_id"] not in visible else "destination","reason":"instance_not_currently_visible"}
    if slots["source_id"] is not None and slots["action"] is None:
        return {"status":"clarify","slot":"action","reason":"object_selection_does_not_authorize_action"}
    if slots["action"] == "place" and slots["destination_id"] is None:
        return {"status":"clarify","slot":"destination","reason":"destination_not_explicit"}
    obs["selection"]["object_id"]=slots["source_id"]
    obs["context"]={"confirmed_action":slots["action"],"confirmed_destination_id":slots["destination_id"]}
    check=structural_clarification(obs)
    if check is not None: return {"status":check["status"],"slot":check["slot"],"reason":check["reason"]}
    validate_intent(slots)
    if candidate_key(slots) not in {candidate_key(i) for i in generate_candidates(obs)}:
        return {"status":"reject","slot":None,"reason":"intent_not_in_scene_grammar"}
    try: instruction=render_instruction(obs,slots)
    except ValueError:
        return {"status":"clarify","slot":"destination" if slots["destination_id"] else "source","reason":"language_reference_ambiguous"}
    return {"status":"proposal","slot":None,"reason":"needs_versioned_confirmation","intent":dict(slots),"instruction":instruction}


def revalidate_contract(contract,old_scene,new_scene,backend):
    require(isinstance(contract,TaskContract),"validated_contract_required")
    v=contract.to_dict(); intent=v["intent"]
    require(v["scene_id"] == old_scene.data["scene_id"] and v["scene_revision"] == old_scene.data["revision"],"contract_baseline_mismatch")
    require(new_scene.data["scene_id"] == old_scene.data["scene_id"] and new_scene.data["identity"] == old_scene.data["identity"],"identity_domain_changed")
    require(new_scene.data["revision"] >= old_scene.data["revision"],"scene_revision_regression")
    if new_scene.data["revision"] == old_scene.data["revision"]:
        require(new_scene.to_dict() == old_scene.to_dict(),"scene_version_reused")
    else:
        identities=IdentityLedger(); identities.observe(old_scene); identities.observe(new_scene)
    if candidate_key(intent) not in {candidate_key(i) for i in generate_candidates(new_scene.legacy_observation())}:
        return {"status":"needs_requery","reason":"relevant_instance_missing_or_affordance_changed","intent":intent,"execute_allowed":False}
    changed=old_scene.relevant_fingerprint(intent) != new_scene.relevant_fingerprint(intent)
    # Even unrelated edits can change natural-language referring expressions.
    if changed or new_scene.data["revision"] != old_scene.data["revision"]:
        return {"status":"needs_reconfirmation","reason":"relevant_scene_changed" if changed else "scene_revision_changed","intent":intent,"execute_allowed":False}
    capability=backend.assess(intent)
    return {"status":"structurally_current","reason":capability["reason"],"intent":intent,"execute_allowed":False}


class TaskHistory:
    """Three distinct append-only streams, not a user-goal estimator or authority."""
    def __init__(self): self._proposals=[]; self._confirmed=[]; self._progress=[]

    @property
    def proposals(self): return tuple(self._proposals)
    @property
    def confirmed(self): return tuple(self._confirmed)
    @property
    def progress(self): return tuple(self._progress)

    def record_proposal(self,contract):
        require(isinstance(contract,TaskContract),"validated_contract_required")
        require(contract.data["status"] == "proposed","proposal_required")
        require(all(c.data["contract_id"] != contract.data["contract_id"] for c in self._proposals),"duplicate_contract")
        self._proposals.append(contract)

    def record_confirmed(self,contract,*,authorization_ref):
        require(isinstance(contract,TaskContract),"validated_contract_required")
        text(authorization_ref)
        require(contract.data["status"] == "confirmed" and contract.data["confirmation_event_id"] == authorization_ref,"external_confirmation_required")
        prior=next((c for c in self._proposals if c.data["contract_id"]==contract.data["contract_id"]),None)
        require(prior is not None,"unknown_proposal")
        require(all(prior.data[k] == contract.data[k] for k in ["task_id","revision","scene_id","scene_revision","intent","valid_until"]),"confirmation_changed_proposal")
        require(all(c.data["contract_id"] != contract.data["contract_id"] for c in self._confirmed),"duplicate_confirmation")
        self._confirmed.append(contract)

    def record_progress(self,event):
        require(isinstance(event,ExecutionEvent),"validated_execution_event_required")
        require(any(c.data["contract_id"]==event.data["contract_id"] and c.data["revision"]==event.data["contract_revision"] for c in self._confirmed),"progress_without_confirmed_contract")
        self._progress.append(event)
