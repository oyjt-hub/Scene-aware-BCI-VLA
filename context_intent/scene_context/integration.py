"""Explicit integration seam: decisions only; never imports robot/hardware SDKs."""
from __future__ import annotations

import hashlib
import json
import math
import uuid

from .schema import validate_observation


def observation_from_detections(labels, bboxes, frame_size, *, capabilities,
        selected_label=None, selected_id=None, instance_ids=None,
        context=None, probabilities=None, attributes=None):
    """Adapt current Florence/SAM labels+pixel boxes without inventing intent.

    capabilities maps labels to the caller's actual supported task affordances.
    Same-name objects are never resolved by choosing the first matching label.
    IDs generated here are snapshot-local: use persistent tracker IDs when available.
    """
    if not isinstance(labels,list) or not isinstance(bboxes,list) or len(labels)!=len(bboxes):
        raise ValueError("labels and pixel bboxes must be equally sized lists")
    if len(frame_size)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in frame_size):
        raise ValueError("frame_size must be positive [width,height]")
    if not isinstance(capabilities,dict):
        raise ValueError("explicit supported label-to-affordance capabilities are required")
    identifiers=instance_ids if instance_ids is not None else [f"instance_{i}" for i in range(len(labels))]
    if len(identifiers)!=len(labels):
        raise ValueError("one instance ID is required per detection")
    width,height=frame_size
    objects=[]
    for identifier,label,box in zip(identifiers,labels,bboxes):
        if not isinstance(box,(list,tuple)) or len(box)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in box):
            raise ValueError("bbox must contain finite pixel xyxy coordinates")
        x1,y1,x2,y2=box
        if not 0<=x1<x2<=width or not 0<=y1<y2<=height:
            raise ValueError("bbox must fit its explicitly supplied image coordinate frame")
        objects.append({"id":identifier,"label":label,"position":[(x1+x2)/(2*width),(y1+y2)/(2*height)],
            "affordances":capabilities.get(label,[]),"attributes":(attributes or {}).get(identifier,[])})
    if selected_id is None and selected_label is not None:
        matches=[obj["id"] for obj in objects if obj["label"].casefold()==str(selected_label).casefold()]
        selected_id=matches[0] if len(matches)==1 else None
    relations=[]
    for first in objects:
        for second in objects:
            if first["id"]==second["id"]:
                continue
            dx=second["position"][0]-first["position"][0]
            dy=second["position"][1]-first["position"][1]
            if abs(dx)>.02:
                relations.append({"subject":first["id"],"predicate":"left_of" if dx>0 else "right_of","object":second["id"]})
            if abs(dy)>.02:
                relations.append({"subject":first["id"],"predicate":"above" if dy>0 else "below","object":second["id"]})
            if dx*dx+dy*dy<.09:
                relations.append({"subject":first["id"],"predicate":"near","object":second["id"]})
    return validate_observation({"objects":objects,"relations":relations,
        "selection":{"object_id":selected_id,"probabilities":probabilities or {}},"context":context or {}})


class ContextIntentAdapter:
    def __init__(self, predictor):
        self.predictor=predictor
        self._pending=None

    @staticmethod
    def observation_digest(observation):
        public=validate_observation(observation)
        # Ignore object/edge enumeration order; preserve the public scene state.
        public["objects"].sort(key=lambda item:item["id"])
        public["relations"].sort(key=lambda item:(item["subject"],item["predicate"],item["object"],item["confidence"]))
        packed=json.dumps(public,sort_keys=True,separators=(",",":"),allow_nan=False)
        return hashlib.sha256(packed.encode("utf-8")).hexdigest()

    def propose(self, public_observation):
        public=validate_observation(public_observation)
        self._pending=None
        decision=self.predictor.predict(public)
        if decision.get("status")=="ready":
            proposal_id=uuid.uuid4().hex
            decision["proposal_id"]=proposal_id
            decision["observation_sha256"]=self.observation_digest(public)
            self._pending=json.loads(json.dumps(decision))
        return decision

    def confirm(self, proposal_id, confirmed_intent, current_observation):
        """Consume one exact proposal after an external user's fresh confirmation.

        Scene changes invalidate the pending proposal; the caller should obtain
        a new observation and new confirmation instead of reusing old text.
        """
        pending=self._pending
        self._pending=None
        if not pending or pending["proposal_id"]!=proposal_id:
            raise ValueError("unknown, stale, or already consumed proposal")
        if pending["observation_sha256"]!=self.observation_digest(current_observation):
            raise ValueError("scene or user context changed; propose and confirm again")
        return self.confirmed_instruction(pending,confirmed_intent)

    def cancel(self):
        self._pending=None

    @staticmethod
    def confirmed_instruction(decision, confirmed_intent):
        """A caller must obtain fresh user confirmation of this exact structured intent.

        Returning text does not start the robot. The caller owns always-available
        stop, stale scene checks, user authentication and hardware safeguards.
        """
        if decision.get("status") != "ready" or decision.get("requires_confirmation") is not True:
            raise ValueError("only a ready proposal can be confirmed")
        if not isinstance(confirmed_intent,dict) or confirmed_intent != decision.get("intent"):
            raise ValueError("confirmation must match the exact current proposed intent")
        instruction=decision.get("instruction")
        if not isinstance(instruction,str) or not instruction:
            raise ValueError("proposal has no instruction")
        return instruction
