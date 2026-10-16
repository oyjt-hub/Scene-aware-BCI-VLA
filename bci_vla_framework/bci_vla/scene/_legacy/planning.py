"""Shared grammar, missing-information checks and non-executing output guard."""
from __future__ import annotations

import json
import math

from .rendering import render_instruction
from .schema import validate_observation


def candidate_key(intent):
    return json.dumps({key: intent.get(key) for key in ("action", "source_id", "destination_id")}, sort_keys=True, separators=(",", ":"))


def generate_candidates(observation):
    """Affordance grammar only: do not leak BCI or context into ablation candidates."""
    obs = validate_observation(observation)
    result = []
    for source in obs["objects"]:
        affordances = source["affordances"]
        if "pickable" in affordances:
            result.append({"action": "pick", "source_id": source["id"], "destination_id": None})
            for dest in obs["objects"]:
                if "container" in dest["affordances"] and dest["id"] != source["id"]:
                    result.append({"action": "place", "source_id": source["id"], "destination_id": dest["id"]})
        if "foldable" in affordances:
            result.append({"action": "fold", "source_id": source["id"], "destination_id": None})
    return result


def _selected(obs):
    selection = obs["selection"]
    probs = selection["probabilities"]
    if probs:
        ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        best, value = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        # Conservative configurable starting rule, not a calibrated BCI reliability guarantee.
        if value < 0.7 or value - second < 0.2:
            return None
        if selection["object_id"] is not None and selection["object_id"] != best:
            return None
        return best
    return selection["object_id"]


def compatible_candidates(observation):
    obs = validate_observation(observation)
    candidates = generate_candidates(obs)
    source = _selected(obs)
    if source is None:
        return candidates
    candidates = [x for x in candidates if x["source_id"] == source]
    context = obs["context"]
    goal_action = {"pick": "pick", "put_away": "place", "fold": "fold"}.get(context["goal"])
    if context["goal"] == "repeat_last" and context["history"]:
        goal_action = context["history"][-1]["action"]
    if context["confirmed_action"] and goal_action and context["confirmed_action"] != goal_action:
        return []
    action = context["confirmed_action"] or goal_action
    if action:
        candidates = [x for x in candidates if x["action"] == action]
    dest_id = context["confirmed_destination_id"]
    if not dest_id and context["goal"] == "repeat_last" and context["history"]:
        previous_dest = context["history"][-1].get("destination_id")
        if previous_dest in {obj["id"] for obj in obs["objects"]}:
            dest_id = previous_dest
    if dest_id:
        candidates = [x for x in candidates if x["destination_id"] == dest_id]
    hint = context["destination_hint"]
    place = [x for x in candidates if x["action"] == "place"]
    if hint != "unspecified" and place:
        objects = {x["id"]: x for x in obs["objects"]}
        def distance(candidate):
            src = objects[candidate["source_id"]]["position"]
            dst = objects[candidate["destination_id"]]["position"]
            if hint == "leftmost":
                return dst[0]
            if hint == "rightmost":
                return -dst[0]
            return sum((a-b)**2 for a,b in zip(src,dst))
        best = min(distance(x) for x in place)
        selected = {candidate_key(x) for x in place if abs(distance(x)-best) < 1e-6}
        candidates = [x for x in candidates if x["action"] != "place" or candidate_key(x) in selected]
    return candidates


def structural_clarification(observation):
    obs = validate_observation(observation)
    all_candidates = generate_candidates(obs)
    if not all_candidates:
        return {"status": "reject", "slot": None, "clarification_slot": None,
                "reason": "no_supported_task", "options": []}
    selected = _selected(obs)
    if selected is None:
        return {"status": "clarify", "slot": "source", "clarification_slot": "source",
                "reason": "source_missing_or_bci_uncertain", "options": [
                    {"id": x["id"], "label": x["label"]} for x in obs["objects"] if x["affordances"]]}
    candidates = compatible_candidates(obs)
    if not candidates:
        return {"status": "reject", "slot": None, "clarification_slot": None,
                "reason": "selection_or_context_has_no_supported_task", "options": []}
    actions = sorted({x["action"] for x in candidates})
    if len(actions) > 1:
        return {"status": "clarify", "slot": "action", "clarification_slot": "action",
                "reason": "action_not_determined", "options": [{"id": x, "label": x} for x in actions]}
    destinations = {x["destination_id"] for x in candidates}
    if len(destinations) > 1:
        return {"status": "clarify", "slot": "destination", "clarification_slot": "destination",
                "reason": "destination_not_determined", "options": [
                    {"id": x["id"], "label": x["label"]} for x in obs["objects"] if x["id"] in destinations]}
    return None


def finalize_prediction(observation, proposal, confidence=None, threshold=0.0, source=""):
    obs = validate_observation(observation)
    raw = dict(proposal) if isinstance(proposal, dict) else {"status": "reject", "reason": "invalid_proposal"}
    result = {"status": "reject", "intent": None, "instruction": None,
        "confidence": confidence, "clarification": None, "reason": "invalid_proposal",
        "requires_confirmation": False, "raw_proposal": raw, "method": source}
    if confidence is not None and (not isinstance(confidence,(int,float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1):
        result["confidence"] = None
        result["reason"] = "invalid_confidence"
        return result
    status = raw.get("status")
    if status not in {"ready", "clarify", "reject"}:
        return result
    # A failed call/parser must stay rejected, rather than being made to look correct by a guard.
    if status == "reject":
        result["reason"] = raw.get("reason", "predictor_rejected")
        return result
    structural = structural_clarification(obs)
    if structural is not None:
        result.update(status=structural["status"], reason=structural["reason"])
        if structural["status"] == "clarify":
            result["clarification"] = {"slot": structural["slot"], "options": structural["options"]}
        return result
    if status == "clarify":
        slot = raw.get("clarification_slot")
        if slot not in {"source", "action", "destination"}:
            return result
        result.update(status="clarify", reason="predictor_requests_clarification",
            clarification={"slot": slot, "options": []})
        return result
    intent = raw.get("intent")
    if not isinstance(intent, dict):
        return result
    key = candidate_key(intent)
    if key not in {candidate_key(x) for x in generate_candidates(obs)}:
        result["reason"] = "unsupported_or_invalid_intent"
        return result
    if key not in {candidate_key(x) for x in compatible_candidates(obs)}:
        result["reason"] = "intent_conflicts_with_observed_selection_or_context"
        return result
    if confidence is not None and confidence < threshold:
        result.update(status="clarify", reason="model_score_below_validation_threshold",
            clarification={"slot": "action", "options": []})
        return result
    try:
        instruction = render_instruction(obs, intent)
    except ValueError as exc:
        result.update(status="clarify", reason=str(exc), clarification={"slot": "destination" if intent.get("destination_id") else "source", "options": []})
        return result
    result.update(status="ready", intent={k:intent.get(k) for k in ("action","source_id","destination_id")},
        instruction=instruction, reason="proposal_requires_user_confirmation", requires_confirmation=True)
    return result
