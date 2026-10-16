"""Strict offline fixture adapter. Acceptance is not evidence of execution.

The selected paper checkpoint and live observation/action conventions remain
unverified. This module imports no network, ML or robot library and cannot
enable live operation. Its ledger is in-memory and is not crash-safe delivery.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from ...protocols import (fields, freeze, integer, json_object, number, require,
                          text, timestamp)


def _clone(value):
    return json_object(value)


class VLAInterfaceSpec:
    """Validate a fully declared fixture interface, without unit conversion."""

    def __init__(self, config=None):
        if config is None:
            path = Path(__file__).resolve().parents[3] / "configs/execution/fake_vla_interface.json"
            config = json.loads(path.read_text(encoding="utf-8"))
        value = _clone(config)
        fields(value, {"schema_version", "scope", "deployment_allowed", "interface_version",
                       "images", "state", "actions", "normalization", "coordinate_frame",
                       "source_evidence", "unverified_live_fields", "fixture_observation",
                       "fixture_actions"})
        require(value["schema_version"] == "vla-interface/1.0", "unsupported_interface_schema")
        require(value["scope"] == "offline_fixture" and value["deployment_allowed"] is False,
                "live_interface_unverified")
        require(value["interface_version"] == "fake-vla/1.0", "unsupported_interface_version")
        fields(value["images"], {"keys", "shape_hwc", "dtype"})
        keys = value["images"]["keys"]
        require(isinstance(keys, list) and len(keys) == 3 and
                all(isinstance(key, str) for key in keys) and len(set(keys)) == 3,
                "invalid_image_keys")
        for key in keys:
            text(key)
        shape = value["images"]["shape_hwc"]
        require(isinstance(shape, list) and len(shape) == 3, "invalid_image_shape")
        for extent in shape:
            integer(extent, 1)
            require(extent <= 64, "fixture_image_too_large")
        require(shape[2] == 3 and value["images"]["dtype"] == "uint8", "unsupported_image_format")
        for part, dimensions in (("state", {"dimension", "units", "lower", "upper"}),
                                 ("actions", {"dimension", "horizon", "units", "lower", "upper"})):
            spec = value[part]
            fields(spec, dimensions)
            integer(spec["dimension"], 1)
            require(spec["dimension"] <= 64, "fixture_vector_too_large")
            if part == "actions":
                integer(spec["horizon"], 1)
                require(spec["horizon"] <= 100, "fixture_horizon_too_large")
            for name in ("units", "lower", "upper"):
                require(isinstance(spec[name], list) and len(spec[name]) == spec["dimension"],
                        "invalid_vector_spec")
            for i in range(spec["dimension"]):
                text(spec["units"][i])
                number(spec["lower"][i]); number(spec["upper"][i])
                require(spec["lower"][i] < spec["upper"][i], "invalid_vector_bounds")
        text(value["normalization"]); text(value["coordinate_frame"])
        require(isinstance(value["source_evidence"], list) and bool(value["source_evidence"]),
                "source_evidence_required")
        require(isinstance(value["unverified_live_fields"], list) and bool(value["unverified_live_fields"]),
                "unverified_live_fields_required")
        self.data = freeze(value)
        self.validate_observation(value["fixture_observation"])
        self.validate_actions(value["fixture_actions"])

    def _versions(self, value):
        require(value["interface_version"] == self.data["interface_version"], "interface_version_mismatch")
        require(value["normalization"] == self.data["normalization"], "normalization_mismatch")
        require(value["coordinate_frame"] == self.data["coordinate_frame"], "coordinate_frame_mismatch")

    def _vector(self, vector, units, part):
        spec = self.data[part]
        require(isinstance(vector, list) and len(vector) == spec["dimension"], "vector_shape_mismatch")
        require(isinstance(units, list) and tuple(units) == spec["units"], "vector_unit_mismatch")
        for i, item in enumerate(vector):
            number(item, spec["lower"][i], spec["upper"][i])

    def validate_observation(self, observation):
        value = _clone(observation)
        fields(value, {"interface_version", "scene_id", "scene_revision", "captured_at", "images",
                       "state", "state_units", "coordinate_frame", "normalization"})
        self._versions(value)
        text(value["scene_id"]); integer(value["scene_revision"], 1); timestamp(value["captured_at"])
        require(isinstance(value["images"], dict) and set(value["images"]) == set(self.data["images"]["keys"]),
                "image_keys_mismatch")
        height, width, channels = self.data["images"]["shape_hwc"]
        for pixels in value["images"].values():
            require(isinstance(pixels, list) and len(pixels) == height, "image_shape_mismatch")
            for row in pixels:
                require(isinstance(row, list) and len(row) == width, "image_shape_mismatch")
                for pixel in row:
                    require(isinstance(pixel, list) and len(pixel) == channels, "image_shape_mismatch")
                    for channel in pixel:
                        integer(channel)
                        require(channel <= 255, "image_pixel_out_of_range")
        self._vector(value["state"], value["state_units"], "state")
        return value

    def validate_actions(self, actions):
        value = _clone(actions)
        fields(value, {"interface_version", "values", "units", "coordinate_frame", "normalization"})
        self._versions(value)
        require(isinstance(value["values"], list) and len(value["values"]) == self.data["actions"]["horizon"],
                "action_horizon_mismatch")
        for row in value["values"]:
            self._vector(row, value["units"], "actions")
        return value


class FakeVLAAdapter:
    """One in-memory submission per ID; all execution feedback is simulated."""

    def __init__(self, spec=None, *, submit_status="ACCEPTED", execution_mode="offline"):
        require(execution_mode == "offline", "live_execution_unverified")
        require(isinstance(submit_status, str) and submit_status in {"ACCEPTED", "UNKNOWN", "REJECTED"},
                "invalid_submission_status")
        require(spec is None or isinstance(spec, VLAInterfaceSpec), "invalid_interface_spec")
        self.spec = spec or VLAInterfaceSpec()
        self.submit_status = submit_status
        self._submissions = {}
        self._feedback = {}
        self._audit = []

    @property
    def audit_log(self):
        return tuple(freeze(deepcopy(item)) for item in self._audit)

    @property
    def submission_count(self):
        return len(self._submissions)

    def validate_observation(self, observation):
        return self.spec.validate_observation(observation)

    def validate_actions(self, actions):
        return self.spec.validate_actions(actions)

    def _payload(self, submission_id, contract_id, revision, instruction, observation):
        text(submission_id); text(contract_id); integer(revision, 1); text(instruction)
        return {"submission_id": submission_id, "contract_id": contract_id, "revision": revision,
                "instruction": instruction, "observation": self.validate_observation(observation)}

    def submit(self, submission_id, contract_id, revision, instruction, observation):
        payload = self._payload(submission_id, contract_id, revision, instruction, observation)
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                                ensure_ascii=False).encode("utf-8")).hexdigest()
        if submission_id in self._submissions:
            require(self._submissions[submission_id]["fingerprint"] == fingerprint,
                    "submission_id_payload_conflict")
            result = self.query(submission_id)
            result["duplicate"] = True
            return result
        result = {"submission_id": submission_id, "contract_id": contract_id, "revision": revision,
                  "instruction": instruction, "status": self.submit_status,
                  "execution_status": "UNKNOWN" if self.submit_status == "UNKNOWN" else "NOT_STARTED",
                  "stop_requested": False, "last_feedback_id": None, "actual_execution": False,
                  "reason": "offline_fake_submission", "duplicate": False}
        self._submissions[submission_id] = {"fingerprint": fingerprint, "payload": payload, "result": result}
        self._audit.append({"kind": "SUBMIT", "submission_id": submission_id, "status": self.submit_status,
                            "actual_execution": False})
        return deepcopy(result)

    def query(self, submission_id):
        text(submission_id)
        if submission_id not in self._submissions:
            return {"submission_id": submission_id, "contract_id": None, "revision": None,
                    "instruction": None, "status": "UNKNOWN", "execution_status": "UNKNOWN",
                    "stop_requested": False, "last_feedback_id": None, "actual_execution": False,
                    "reason": "submission_not_in_local_ledger", "duplicate": False}
        return deepcopy(self._submissions[submission_id]["result"])

    def stop(self, submission_id):
        text(submission_id)
        require(submission_id in self._submissions, "unknown_submission")
        result = self._submissions[submission_id]["result"]
        if not result["stop_requested"]:
            result["stop_requested"] = True
            self._audit.append({"kind": "STOP_REQUEST", "submission_id": submission_id,
                                "actual_execution": False})
        return self.query(submission_id)

    def record_feedback(self, submission_id, status, *, feedback_id):
        text(submission_id); text(feedback_id)
        require(submission_id in self._submissions, "unknown_submission")
        require(isinstance(status, str) and status in {"STARTED", "COMPLETED", "FAILED", "STOPPED", "UNKNOWN"},
                "invalid_execution_feedback")
        feedback = (submission_id, status)
        if feedback_id in self._feedback:
            require(self._feedback[feedback_id] == feedback, "feedback_id_payload_conflict")
            return self.query(submission_id)
        result = self._submissions[submission_id]["result"]
        require(result["status"] != "REJECTED", "rejected_submission_feedback")
        old = result["execution_status"]
        require(old not in {"COMPLETED", "FAILED", "STOPPED"}, "terminal_execution_state")
        if status == "STARTED":
            require(old in {"NOT_STARTED", "UNKNOWN"}, "invalid_execution_transition")
        elif status == "COMPLETED":
            require(old == "STARTED", "completion_without_start")
        result["execution_status"] = status
        result["last_feedback_id"] = feedback_id
        if status in {"STARTED", "COMPLETED", "FAILED", "STOPPED"}:
            result["status"] = "ACCEPTED"
        self._feedback[feedback_id] = feedback
        self._audit.append({"kind": "SIMULATED_FEEDBACK", "submission_id": submission_id,
                            "feedback_id": feedback_id, "execution_status": status,
                            "actual_execution": False})
        return self.query(submission_id)


class DryRunVLAAdapter(FakeVLAAdapter):
    """Validate a proposed submission without creating a backend submission."""

    def submit(self, submission_id, contract_id, revision, instruction, observation):
        self._payload(submission_id, contract_id, revision, instruction, observation)
        return {"submission_id": submission_id, "contract_id": contract_id, "revision": revision,
                "instruction": instruction, "status": "REJECTED", "execution_status": "NOT_STARTED",
                "stop_requested": False, "last_feedback_id": None, "actual_execution": False,
                "reason": "dry_run_validation_only", "duplicate": False}
