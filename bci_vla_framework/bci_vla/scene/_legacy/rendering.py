"""Deterministic realization of a finite task grammar, not open language generation."""
from __future__ import annotations


def _describe(obs, identifier):
    obj = next((x for x in obs["objects"] if x["id"] == identifier), None)
    if obj is None:
        raise ValueError("cannot describe absent object")
    peers = [x for x in obs["objects"] if x["label"].lower() == obj["label"].lower()]
    attributes = obj.get("attributes", [])
    description = " ".join([*attributes, obj["label"]])
    if len(peers) == 1:
        return description
    if attributes and all(not set(attributes).issubset(set(x.get("attributes", []))) for x in peers if x["id"] != identifier):
        return description
    ordered = sorted(peers, key=lambda x: x["position"][0])
    xs = [x["position"][0] for x in ordered]
    if ordered[0]["id"] == identifier and xs[1] - xs[0] > 0.05:
        return "leftmost " + description
    if ordered[-1]["id"] == identifier and xs[-1] - xs[-2] > 0.05:
        return "rightmost " + description
    raise ValueError("same-class instance lacks an unambiguous supported visual description")


def render_instruction(observation, intent):
    source = _describe(observation, intent["source_id"])
    action = intent["action"]
    if action == "pick" and intent.get("destination_id") is None:
        return f"Pick up the {source}."
    if action == "fold" and intent.get("destination_id") is None:
        return f"Fold the {source}."
    if action == "place" and intent.get("destination_id") is not None:
        destination = _describe(observation, intent["destination_id"])
        return f"Put the {source} into the {destination}."
    raise ValueError("unsupported action grammar")
