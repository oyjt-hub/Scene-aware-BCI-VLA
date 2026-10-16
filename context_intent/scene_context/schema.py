"""Validated public observations and supervised, scene-grouped cases."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

ACTIONS = ("pick", "place", "fold")
GOALS = ("unspecified", "pick", "put_away", "fold", "repeat_last")
RELATIONS = ("left_of", "right_of", "above", "below", "near", "inside", "overlapping", "supports")
HINTS = ("unspecified", "leftmost", "rightmost", "nearest_source")
AFFORDANCES = ("pickable", "container", "foldable")
STATUSES = ("ready", "clarify", "reject")


def _text(value, name, maximum=100):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be nonempty text, at most {maximum} characters")
    if any(ord(c) < 32 for c in value):
        raise ValueError(f"{name} contains control characters")
    return value.strip()


def _prob(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0,1]")
    return float(value)


def validate_observation(data):
    if not isinstance(data, dict):
        raise ValueError("observation must be an object")
    if set(data) - {"objects", "relations", "selection", "context"}:
        raise ValueError("Only public objects, relations, selection, context are allowed in observation")
    if not isinstance(data.get("objects"), list) or len(data["objects"]) > 64:
        raise ValueError("objects must be a list of at most 64 objects")
    result = {"objects": [], "relations": []}
    ids = set()
    for obj in data["objects"]:
        if not isinstance(obj, dict):
            raise ValueError("each object must be a dictionary")
        if set(obj) - {"id", "label", "position", "attributes", "affordances", "detection_confidence"}:
            raise ValueError("unsupported object field")
        identifier = _text(obj.get("id"), "object id", 64)
        if identifier in ids:
            raise ValueError("duplicate object id")
        ids.add(identifier)
        position = obj.get("position")
        if not isinstance(position, (list, tuple)) or len(position) != 2:
            raise ValueError("position must be normalized image [x,y]; do not invent geometry")
        attributes = obj.get("attributes", [])
        affordances = obj.get("affordances", [])
        if not isinstance(attributes, list) or len(attributes) > 8:
            raise ValueError("attributes must be a list of at most eight strings")
        if not isinstance(affordances, list) or set(affordances) - set(AFFORDANCES):
            raise ValueError("unknown affordance; no open-world capability is assumed")
        result["objects"].append({
            "id": identifier, "label": _text(obj.get("label"), "object label", 64),
            "position": [_prob(v, "position") for v in position],
            "attributes": [_text(v, "attribute", 40) for v in attributes],
            "affordances": sorted(set(affordances)),
            "detection_confidence": _prob(obj.get("detection_confidence", 1.0), "detection_confidence"),
        })
    relations = data.get("relations", [])
    if not isinstance(relations, list) or len(relations) > 4096:
        raise ValueError("relations must be a bounded list")
    for edge in relations:
        if not isinstance(edge, dict) or edge.get("subject") not in ids or edge.get("object") not in ids:
            raise ValueError("relation references unknown object")
        if edge.get("predicate") not in RELATIONS:
            raise ValueError("unknown relation predicate")
        result["relations"].append({"subject": edge["subject"], "predicate": edge["predicate"],
            "object": edge["object"], "confidence": _prob(edge.get("confidence", 1.0), "relation confidence")})
    selection = data.get("selection", {})
    if not isinstance(selection, dict) or set(selection) - {"object_id", "probabilities"}:
        raise ValueError("invalid selection fields")
    chosen = selection.get("object_id")
    if chosen is not None and chosen not in ids:
        raise ValueError("selected object is absent from scene")
    probabilities = selection.get("probabilities", {})
    if not isinstance(probabilities, dict) or set(probabilities) - ids:
        raise ValueError("BCI probabilities reference unknown objects")
    probabilities = {key: _prob(value, "BCI probability") for key, value in probabilities.items()}
    if probabilities:
        total = sum(probabilities.values())
        if total <= 0:
            raise ValueError("BCI probabilities must have positive mass")
        probabilities = {key: value / total for key, value in probabilities.items()}
    result["selection"] = {"object_id": chosen, "probabilities": probabilities}
    context = data.get("context", {})
    allowed = {"goal", "confirmed_action", "confirmed_destination_id", "destination_hint", "history"}
    if not isinstance(context, dict) or set(context) - allowed:
        raise ValueError("invalid context fields; annotations are never context")
    goal = context.get("goal", "unspecified")
    action = context.get("confirmed_action")
    destination = context.get("confirmed_destination_id")
    hint = context.get("destination_hint", "unspecified")
    if goal not in GOALS or (action is not None and action not in ACTIONS) or hint not in HINTS:
        raise ValueError("unsupported goal, confirmed action, or destination hint")
    if destination is not None and destination not in ids:
        raise ValueError("confirmed destination is absent from scene")
    history = context.get("history", [])
    if not isinstance(history, list) or len(history) > 8:
        raise ValueError("history must contain at most eight prior confirmed events")
    for event in history:
        if not isinstance(event, dict) or set(event) - {"action", "source_id", "destination_id"}:
            raise ValueError("history must contain public confirmed event dictionaries")
        if event.get("action") not in ACTIONS:
            raise ValueError("history action is invalid")
        for key in ("source_id", "destination_id"):
            if event.get(key) is not None:
                _text(event[key], key, 64)
    result["context"] = {"goal": goal, "confirmed_action": action, "confirmed_destination_id": destination,
        "destination_hint": hint, "history": copy.deepcopy(history)}
    return result


def validate_case(case):
    if not isinstance(case, dict):
        raise ValueError("case must be a dictionary")
    result = copy.deepcopy(case)
    result["case_id"] = _text(case.get("case_id"), "case_id")
    result["scene_id"] = _text(case.get("scene_id"), "scene_id")
    result["observation"] = validate_observation(case.get("observation"))
    annotation = result.get("annotation")
    if not isinstance(annotation, dict) or annotation.get("status") not in STATUSES:
        raise ValueError("case requires a status annotation")
    if annotation["status"] == "ready":
        intent = annotation.get("intent")
        if not isinstance(intent, dict) or set(intent) != {"action", "source_id", "destination_id"} or intent.get("action") not in ACTIONS:
            raise ValueError("ready annotation requires an intent")
        ids = {x["id"] for x in result["observation"]["objects"]}
        if intent.get("source_id") not in ids or (intent.get("destination_id") is not None and intent["destination_id"] not in ids):
            raise ValueError("annotation intent references absent instance")
        objects = {obj["id"]: obj for obj in result["observation"]["objects"]}
        source = objects[intent["source_id"]]
        action = intent["action"]
        destination = intent["destination_id"]
        if action in {"pick", "fold"} and destination is not None:
            raise ValueError("pick/fold annotation cannot have a destination")
        if action == "place" and (destination is None or destination == source["id"] or "container" not in objects[destination]["affordances"]):
            raise ValueError("place annotation needs a distinct supported container")
        required = "foldable" if action == "fold" else "pickable"
        if required not in source["affordances"]:
            raise ValueError("annotation action conflicts with declared source affordance")
    elif annotation.get("intent") is not None:
        raise ValueError("unresolved cases must not expose one hidden intention as a determined label")
    if annotation["status"] == "clarify" and annotation.get("clarification_slot") not in {"source", "action", "destination"}:
        raise ValueError("clarify annotation requires a slot")
    return result


def public_observation(case):
    return validate_observation(case["observation"])


def load_cases(path):
    cases = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                cases.append(validate_case(json.loads(line)))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if len({x["case_id"] for x in cases}) != len(cases):
        raise ValueError("duplicate case_id")
    return cases


def assert_disjoint_scenes(*splits):
    groups = []
    for split in splits:
        cases = load_cases(split) if isinstance(split, (str, Path)) else split
        groups.append({x["scene_id"] for x in cases})
    for i, first in enumerate(groups):
        for j in range(i + 1, len(groups)):
            overlap = first & groups[j]
            if overlap:
                raise ValueError(f"scene leakage between splits {i} and {j}: {sorted(overlap)[:5]}")
