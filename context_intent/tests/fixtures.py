"""Small explicit fixtures; these are test data, never research evidence."""
import copy
import json


def observation(two_destinations=False):
    obs = {
        "objects": [
            {"id": "opaque_grape_739", "label": "grapes", "position": [0.2, 0.5],
             "attributes": [], "affordances": ["pickable"], "detection_confidence": 0.95},
            {"id": "opaque_bowl_284", "label": "bowl", "position": [0.7, 0.5],
             "attributes": [], "affordances": ["container"], "detection_confidence": 0.99},
        ],
        "relations": [{"subject": "opaque_grape_739", "predicate": "left_of",
                       "object": "opaque_bowl_284", "confidence": 0.9}],
        "selection": {"object_id": "opaque_grape_739",
                      "probabilities": {"opaque_grape_739": 0.96, "opaque_bowl_284": 0.04}},
        "context": {"goal": "put_away", "confirmed_action": None,
                    "confirmed_destination_id": None, "destination_hint": "unspecified", "history": []},
    }
    if two_destinations:
        obs["objects"].append({"id": "opaque_bowl_991", "label": "bowl", "position": [0.9, 0.5],
                               "attributes": [], "affordances": ["container"], "detection_confidence": 0.98})
    return obs


def intent(destination="opaque_bowl_284"):
    return {"action": "place", "source_id": "opaque_grape_739", "destination_id": destination}


def ready_proposal(destination="opaque_bowl_284"):
    return {"status": "ready", "intent": intent(destination), "clarification_slot": None}


def case(case_id="case_1", scene_id="scene_1", ambiguous=False):
    return {"case_id": case_id, "scene_id": scene_id, "observation": observation(ambiguous),
            "annotation": {"status": "clarify" if ambiguous else "ready",
                           "intent": None if ambiguous else intent(),
                           "clarification_slot": "destination" if ambiguous else None}}


def provider_response(proposal=None, finish_reason="STOP"):
    text = proposal if isinstance(proposal, str) else json.dumps(proposal or ready_proposal())
    return {"candidates": [{"finishReason": finish_reason,
                             "content": {"parts": [{"text": text}]}}],
            "modelVersion": "offline-fixture", "usageMetadata": {"totalTokenCount": 25}}


def rename_instances(obs):
    result = copy.deepcopy(obs)
    mapping = {obj["id"]: "renamed_" + str(i) for i, obj in enumerate(result["objects"])}
    for obj in result["objects"]:
        obj["id"] = mapping[obj["id"]]
    for edge in result["relations"]:
        edge["subject"], edge["object"] = mapping[edge["subject"]], mapping[edge["object"]]
    result["selection"]["object_id"] = mapping.get(result["selection"]["object_id"])
    result["selection"]["probabilities"] = {mapping[k]: v for k, v in result["selection"]["probabilities"].items()}
    result["context"]["confirmed_destination_id"] = mapping.get(result["context"]["confirmed_destination_id"])
    for event in result["context"]["history"]:
        for key in ("source_id", "destination_id"):
            event[key] = mapping.get(event.get(key), event.get(key))
    return result, mapping
