"""Explicit image coordinates and caller-supplied persistent instance identity."""
from dataclasses import dataclass
import hashlib
import json

from ..protocols import fields, freeze, thaw, integer, interval, json_object, number, require, text, unique_strings


@dataclass(frozen=True)
class GroundedScene:
    data: object

    def __post_init__(self):
        v = json_object(self.data)
        fields(v, {"schema_version", "scene_id", "revision", "observed_at", "coordinate_frame", "identity", "objects", "relations"})
        require(v["schema_version"] == "grounded-scene/1.0", "unsupported_scene_schema")
        text(v["scene_id"]); integer(v["revision"]); interval(v["observed_at"])
        c = v["coordinate_frame"]
        fields(c, {"name", "unit", "width_px", "height_px"})
        require(c["name"] == "image_xy" and c["unit"] == "normalized", "unsupported_coordinate_frame")
        integer(c["width_px"], 1); integer(c["height_px"], 1)
        fields(v["identity"], {"session_id", "scope", "tracker_ref"})
        text(v["identity"]["session_id"]); text(v["identity"]["tracker_ref"])
        require(v["identity"]["scope"] == "session", "persistent_identity_required")
        require(isinstance(v["objects"], list) and len(v["objects"]) <= 64, "invalid_objects")
        ids, tokens = set(), set()
        for obj in v["objects"]:
            fields(obj, {"instance_id", "track_token", "label", "box_xyxy", "visible", "affordances", "attributes", "detection_confidence"})
            for k in ["instance_id", "track_token", "label"]: text(obj[k])
            require(obj["instance_id"] not in ids and obj["track_token"] not in tokens, "duplicate_identity")
            ids.add(obj["instance_id"]); tokens.add(obj["track_token"])
            require(type(obj["visible"]) is bool, "invalid_visibility")
            box = obj["box_xyxy"]
            require(box is not None or not obj["visible"], "visible_object_needs_geometry")
            if box is not None:
                require(isinstance(box, list) and len(box) == 4, "invalid_box")
                for value in box: number(value, 0, 1)
                require(box[0] < box[2] and box[1] < box[3], "invalid_box")
            unique_strings(obj["affordances"], allow_empty=True)
            require(set(obj["affordances"]) <= {"pickable", "container", "foldable"}, "unknown_affordance")
            unique_strings(obj["attributes"], allow_empty=True)
            if obj["detection_confidence"] is not None: number(obj["detection_confidence"], 0, 1)
        require(isinstance(v["relations"], list) and len(v["relations"]) <= 4096, "invalid_relations")
        for edge in v["relations"]:
            fields(edge, {"subject", "predicate", "object", "confidence"})
            require(edge["subject"] in ids and edge["object"] in ids and edge["subject"] != edge["object"], "unknown_relation_instance")
            require(edge["predicate"] in {"left_of", "right_of", "above", "below", "overlapping"}, "unverified_physical_relation")
            number(edge["confidence"], 0, 1)
        object.__setattr__(self, "data", freeze(v))

    def to_dict(self): return thaw(self.data)

    def legacy_observation(self):
        """Reuse finite grammar with neutral context; never import old intent inertia."""
        objects = []
        for obj in self.data["objects"]:
            if not obj["visible"]: continue
            x1, y1, x2, y2 = obj["box_xyxy"]
            objects.append({"id": obj["instance_id"], "label": obj["label"], "position": [(x1+x2)/2, (y1+y2)/2], "affordances": list(obj["affordances"]), "attributes": list(obj["attributes"])})
        visible = {o["id"] for o in objects}
        return {"objects": objects, "relations": [thaw(e) for e in self.data["relations"] if e["subject"] in visible and e["object"] in visible], "selection": {"object_id": None, "probabilities": {}}, "context": {}}

    def relevant_fingerprint(self, intent):
        ids = {intent["source_id"], intent["destination_id"]} - {None}
        objects = sorted([thaw(o) for o in self.data["objects"] if o["instance_id"] in ids], key=lambda o:o["instance_id"])
        edges = sorted([thaw(e) for e in self.data["relations"] if e["subject"] in ids or e["object"] in ids], key=lambda e:(e["subject"],e["predicate"],e["object"]))
        packed = json.dumps({"identity": thaw(self.data["identity"]), "coordinates": thaw(self.data["coordinate_frame"]), "objects": objects, "relations": edges}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def from_detections(*, scene_id, revision, observed_at, session_id, tracker_ref, frame_size, detections):
    """Input pixel xyxy and stable identities. No invented tracking or affordance."""
    require(isinstance(frame_size, (list, tuple)) and len(frame_size) == 2, "frame_size_required")
    width, height = frame_size; integer(width, 1); integer(height, 1)
    require(isinstance(detections, list), "invalid_detections")
    objects = []
    for item in detections:
        fields(item, {"instance_id", "track_token", "label", "box_xyxy_px", "visible", "affordances", "attributes", "detection_confidence"})
        box = item["box_xyxy_px"]
        if box is not None:
            require(isinstance(box, list) and len(box) == 4, "invalid_box")
            for value in box: number(value)
            box = [box[0]/width, box[1]/height, box[2]/width, box[3]/height]
        obj = {k:v for k,v in item.items() if k != "box_xyxy_px"}; obj["box_xyxy"] = box
        objects.append(obj)
    relations = []
    for a in objects:
        for b in objects:
            if a is b or not a["visible"] or not b["visible"] or a["box_xyxy"] is None or b["box_xyxy"] is None: continue
            ax, ay = (a["box_xyxy"][0]+a["box_xyxy"][2])/2, (a["box_xyxy"][1]+a["box_xyxy"][3])/2
            bx, by = (b["box_xyxy"][0]+b["box_xyxy"][2])/2, (b["box_xyxy"][1]+b["box_xyxy"][3])/2
            if ax != bx: relations.append({"subject":a["instance_id"],"predicate":"left_of" if ax<bx else "right_of","object":b["instance_id"],"confidence":1.0})
            if ay != by: relations.append({"subject":a["instance_id"],"predicate":"above" if ay<by else "below","object":b["instance_id"],"confidence":1.0})
    return GroundedScene({"schema_version":"grounded-scene/1.0", "scene_id":scene_id,"revision":revision,"observed_at":observed_at,"coordinate_frame":{"name":"image_xy","unit":"normalized","width_px":width,"height_px":height},"identity":{"session_id":session_id,"scope":"session","tracker_ref":tracker_ref},"objects":objects,"relations":relations})


class IdentityLedger:
    """Check identity declarations across snapshots; does not perform tracking."""
    def __init__(self): self._identity=None; self._last=None; self._tokens={}; self._ids={}

    def observe(self, scene):
        identity = (scene.data["identity"]["session_id"], scene.data["identity"]["tracker_ref"], scene.data["scene_id"])
        require(self._identity is None or identity == self._identity, "identity_domain_changed")
        require(self._last is None or scene.data["revision"] > self._last, "scene_revision_not_increasing")
        for obj in scene.data["objects"]:
            i,t = obj["instance_id"],obj["track_token"]
            require(self._tokens.get(t,i) == i and self._ids.get(i,t) == t, "identity_reassigned")
        for obj in scene.data["objects"]:
            self._tokens[obj["track_token"]]=obj["instance_id"]; self._ids[obj["instance_id"]]=obj["track_token"]
        self._identity=identity; self._last=scene.data["revision"]
