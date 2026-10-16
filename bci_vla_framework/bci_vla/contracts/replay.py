"""Event-driven offline session reconstruction without a VLA backend.

Checkpoints are compared with a reconstructed state, never adopted as state.
Public response evidence reruns the BCI validator and joint filter. Rejected
inputs omit their quarantined raw content, so only their lack of public state
effects is replayable. Backend replies remain declared external observations;
replay cannot prove physical execution or authenticate the local hash chain.
"""
from __future__ import annotations

from ..protocols import (
    Event, TaskContract, JointBelief, fields, thaw, require, number, text,
    timestamp, event_clock, interval, integer,
)
from ..adapters.bci.replay import ReplayBCIAdapter, context
from ..core.joint_belief import GoalVocabulary, JointBeliefFilter, TaskQuestionBinding
from ..core.diagnostic_policy import DiagnosticPolicy, PlanningState
from ..models.channel import ChannelModel, JointTransitionModel
from ..models.consequence import OutcomeTable, ObservedTrajectory, evaluate_trajectory
from ..monitoring.journal import EventJournal
from ..scene.state import GroundedScene, IdentityLedger
from ..scene.tasks import goal_id
from ..scene._legacy.planning import candidate_key, generate_candidates
from .records import ExecutionContract, AuthorizationRecord, EndpointEvidence, digest


ACTIVE = {"SUBMITTED", "SUBMISSION_UNKNOWN", "EXECUTING", "VERIFYING"}


def _equal(actual, expected, reason):
    require(actual == expected, reason)


class _Replay:
    def __init__(self, bundle, now, session_id):
        fields(bundle, {"current_context", "scene", "filter_config", "channel_config",
                        "transition_config", "policy_config", "outcome_config",
                        "session_config", "backend_spec"})
        self.config = bundle["session_config"]
        require(self.config["scope"] == "offline_fake_only" and
                self.config["physical_execution_allowed"] is False, "offline_session_only")
        self.context = dict(context(bundle["current_context"]))
        _equal(self.context["session_id"], session_id, "replay_session_mismatch")
        self.scene = GroundedScene(bundle["scene"])
        self.backend_spec = bundle["backend_spec"]
        require(self.backend_spec["scope"] == "offline_fixture" and
                self.backend_spec["deployment_allowed"] is False and
                self.backend_spec["interface_version"] == "fake-vla/1.0", "live_interface_unverified")
        f = bundle["filter_config"]
        _equal(f["current_context"], self.context, "filter_context_mismatch")
        self.channel = ChannelModel(bundle["channel_config"])
        self.transition = JointTransitionModel(bundle["transition_config"])
        self.filter = JointBeliefFilter(
            JointBelief(f["initial_joint"]), vocabulary=GoalVocabulary(f["vocabulary"]),
            channel_model=self.channel, transition_model=self.transition,
            current_context=self.context, layout_id=f["layout_id"],
            initial_at=timestamp(f["initial_at"]), outlier_weight=f["outlier_weight"])
        self.policy = DiagnosticPolicy(bundle["policy_config"], channel_model=self.channel,
            transition_model=self.transition, outcome_table=OutcomeTable(bundle["outcome_config"]))
        _equal(thaw(self.policy.config["context"]), self.context, "policy_context_mismatch")
        _equal((self.scene.data["scene_id"], self.scene.data["revision"]),
               (self.context["scene_id"], self.context["scene_revision"]), "scene_context_mismatch")
        _equal(self.scene.data["identity"]["session_id"], session_id, "scene_identity_session_mismatch")
        self.binding = TaskQuestionBinding(self.config["task_binding"], self.filter.vocabulary)
        self.adapter = ReplayBCIAdapter(self.context,
            response_hosts=set(self.config["response_hosts"]), control_hosts=set(self.config["control_hosts"]))
        self.model_refs = {
            "channel_profile": self.context["channel_profile_id"],
            "channel_sha256": digest(thaw(self.channel.data)),
            "transition_sha256": digest(thaw(self.transition.data)),
            "confirmation_sha256": digest(thaw(self.channel.data["matrices"]["CONFIRM"])),
            "policy_sha256": digest(thaw(self.policy.config)),
            "backend_spec_sha256": digest(self.backend_spec),
        }
        self.state = {
            "schema_version": "interaction-session/1.0", "scope": "offline_fake_only",
            "context": dict(self.context), "state": "COLLECTING", "reason": "session_started",
            "contract": None, "authorization": None, "authorization_consumed": False,
            "submission": None, "submission_contract": None, "submission_at": None,
            "completion_feedback_at": None, "stop_latched": False,
            "stop_feedback": "NOT_REQUESTED", "generation_valid": True,
            "model_refs": dict(self.model_refs), "execution_allowed": False,
            "physical_stop_verified": False,
        }
        self.confirmation = None
        self.proposal_ids = set()
        self.confirmation_contexts = set()
        self.consumed_evidence = set()
        self.feedback_ids = set()
        self.trajectory_ids = set()
        self.evidence_ids = set()
        self.submission_fingerprint = None
        self.submission_return_pending = False
        self.expiry_stop_pending = False
        self.start = now

    def snapshot(self):
        return dict(self.state, filter=self.filter.snapshot())

    def revoke(self):
        self.state["authorization"] = None
        self.confirmation = None

    def change(self, state, reason):
        self.state.update(state=state, reason=reason)

    def current_contract(self, now):
        c = self.state["contract"]
        require(c is not None, "proposal_required")
        p = c["proposal"]
        _equal((p["task_id"], p["revision"], p["scene_id"], p["scene_revision"]),
               (self.context["task_id"], self.context["task_revision"],
                self.context["scene_id"], self.context["scene_revision"]), "stale_contract_context")
        require(now < timestamp(p["valid_until"]), "contract_expired")
        _equal(c["scene_fingerprint"], self.scene.relevant_fingerprint(p["intent"]), "scene_fingerprint_changed")
        _equal(c["backend_spec_sha256"], self.model_refs["backend_spec_sha256"], "backend_spec_changed")
        require(self.state["generation_valid"], "models_or_context_require_explicit_rebuild")
        return c

    def predicted(self, now, snapshot=None):
        snap = self.filter.snapshot() if snapshot is None else snapshot.copy()
        gap = (now - timestamp(snap["as_of_reference_time"])).total_seconds()
        require(gap >= 0, "belief_time_regression")
        cells = {(r["goal_id"], r["mode_id"]): r["probability"] for r in snap["belief"]["mass"]}
        known = [(g, z) for g in self.filter.vocabulary.known_goal_ids for z in self.channel.data["mode_ids"]]
        matrix = self.transition.matrix(gap)
        predicted = dict(cells)
        for j, key in enumerate(known):
            predicted[key] = sum(cells.get(old, 0) * matrix[i][j] for i, old in enumerate(known))
        snap["belief"] = JointBelief(dict(snap["belief"], mass=[
            {"goal_id": g, "mode_id": z, "probability": p} for (g, z), p in predicted.items()])).to_dict()
        snap["as_of_reference_time"] = now.isoformat()
        return snap, gap

    def interaction(self, event, now):
        fields(event, {"request", "monotonic_now"})
        require(not self.state["stop_latched"] and self.state["state"] not in ACTIVE,
                "interaction_while_execution_unresolved")
        require(self.state["generation_valid"], "models_or_context_require_explicit_rebuild")
        require(self.filter.snapshot()["point_inference_ready"], "filter_not_ready")
        q = Event.parse(event["request"])
        mode = q.data["mode"]
        require(mode in {"TASK_QUERY", "CHANNEL_PROBE", "CONFIRM"}, "unsupported_session_interaction")
        if mode == "TASK_QUERY":
            self.binding.validate_request(q.data)
        elif mode == "CHANNEL_PROBE":
            require(q.data["payload"]["contract"] is None, "probe_is_not_contract")
            _equal([{k: o[k] for k in ("option_id", "symbol", "semantic_ref")} for o in q.data["payload"]["options"]],
                   [{"option_id": "probe-" + s, "symbol": s, "semantic_ref": "probe:" + s}
                    for s in self.channel.data["scope"]["symbols"]], "probe_option_mismatch")
        else:
            c = self.current_contract(now)
            _equal(thaw(q.data["payload"]["contract"]), c["proposal"], "confirmation_proposal_mismatch")
            key = tuple(self.context[k] for k in sorted(self.context))
            require(key not in self.confirmation_contexts, "confirmation_already_attempted_for_version")
            self.confirmation_contexts.add(key)
            accept = self.config["accept_symbol"]
            _equal({o["semantic_ref"]: o["symbol"] for o in q.data["payload"]["options"]},
                   {"confirmation:accept": accept, "confirmation:reject": next(
                       s for s in self.channel.data["scope"]["symbols"] if s != accept)},
                   "confirmation_mapping_mismatch")
        self.adapter.register_request(q, now=now, monotonic_now=event["monotonic_now"])
        self.revoke()
        self.change({"TASK_QUERY": "CLARIFYING", "CHANNEL_PROBE": "PROBING",
                     "CONFIRM": "AWAITING_CONFIRMATION"}[mode], "awaiting_bound_response")

    def input(self, event, now):
        require(event["accepted"] is True and event["response"] is not None, "accepted_result_required")
        require(not self.state["stop_latched"], "session_stop_latched")
        result = self.adapter.ingest(event["response"], now=now, monotonic_now=event["monotonic_now"])
        require(result.accepted and result.request is not None, "replay_input_not_accepted")
        _equal(result.reason, event["reason"], "replay_input_reason_mismatch")
        _equal(result.request.to_dict(), event["request"], "replay_input_request_mismatch")
        q, r = result.request.data, result.response.data
        update = self.filter.update(result, binding=self.binding if q["mode"] == "TASK_QUERY" else None,
            compliance_id=self.config["probe_compliance_id"] if q["mode"] == "CHANNEL_PROBE" else None,
            assumption="state_constant_within_window", finalized=True)
        _equal(update, event["filter_update"], "replay_filter_update_mismatch")
        self.revoke()
        if not update["updated"] or r["payload"]["status"] != "OBSERVATION":
            self.change("PAUSED", update["reason"] if not update["updated"] else "nonresponse_requires_pause")
        elif q["mode"] == "CONFIRM":
            require(self.state["contract"] is not None, "proposal_required")
            _equal(thaw(q["payload"]["contract"]), self.state["contract"]["proposal"], "confirmation_proposal_mismatch")
            accept = [o["symbol"] for o in q["payload"]["options"] if o["semantic_ref"] == "confirmation:accept"]
            if len(accept) == 1 and r["payload"]["symbol"] == accept[0]:
                self.confirmation = {"request": result.request.to_dict(), "response": result.response.to_dict(),
                                     "snapshot": update["snapshot"]}
                self.change("AWAITING_AUTHORIZATION", "noisy_accept_requires_separate_authorization")
            else:
                self.change("PAUSED", "confirmation_rejected_no_reasking")
        else:
            self.change("COLLECTING", "evidence_updated_no_execution_authority")

    def authorize(self, event, now):
        require(not self.state["stop_latched"] and self.state["state"] == "AWAITING_AUTHORIZATION",
                "actual_confirmation_not_pending_authorization")
        require(self.adapter.pending_request is None and self.confirmation is not None,
                "actual_confirmation_evidence_required")
        c = self.current_contract(now)
        p, proof = c["proposal"], self.confirmation
        require(now < event_clock(proof["response"], "expires_at")[0], "confirmation_evidence_expired")
        gap = (now - timestamp(proof["snapshot"]["as_of_reference_time"])).total_seconds()
        require(0 <= gap <= self.config["max_confirmation_age_s"], "confirmation_too_old")
        _equal(self.filter.snapshot(), proof["snapshot"], "evidence_changed_after_confirmation")
        require(proof["snapshot"]["point_inference_ready"], "filter_not_ready")
        event_id = proof["response"]["event_id"]
        require(event_id not in self.consumed_evidence, "confirmation_already_consumed")
        predicted, gap = self.predicted(now, proof["snapshot"])
        probability = sum(r["probability"] for r in predicted["belief"]["mass"] if r["goal_id"] == c["goal_id"])
        require(probability >= self.config["minimum_goal_probability"], "confirmation_posterior_below_declared_policy")
        expected = AuthorizationRecord({
            "authorization_id": "auth-" + event_id, "scope": "offline_fake_only",
            "contract_sha256": digest(c), "contract_id": p["contract_id"], "task_id": p["task_id"],
            "task_revision": p["revision"], "scene_id": p["scene_id"], "scene_revision": p["scene_revision"],
            "request": proof["request"], "response": proof["response"], "model_refs": dict(self.model_refs),
            "issued_at": now.isoformat(), "valid_until": min(timestamp(p["valid_until"]),
                event_clock(proof["response"], "expires_at")[0]).isoformat(),
            "belief_at_confirmation": proof["snapshot"]["belief"], "belief_at_authorization": predicted["belief"],
            "authorization_prediction_gap_s": gap, "goal_probability": probability,
            "confirmation_is_noisy": True, "frozen_until_dispatch": True, "consume_once": True,
            "physical_execution_allowed": False}).to_dict()
        _equal(AuthorizationRecord(event).to_dict(), expected, "replay_authorization_mismatch")
        self.state["authorization"] = expected
        self.state["authorization_consumed"] = False
        self.consumed_evidence.add(event_id)
        self.change("AUTHORIZED", "offline_fake_authorization_only")

    def validate_observation(self, observation, now):
        """Validate the declared fixture interface locally; create no adapter."""
        spec = self.backend_spec
        fields(observation, {"interface_version", "scene_id", "scene_revision", "captured_at", "images",
                            "state", "state_units", "coordinate_frame", "normalization"})
        for key in ("interface_version", "coordinate_frame", "normalization"):
            _equal(observation[key], spec[key], "replay_observation_interface_mismatch")
        _equal((observation["scene_id"], observation["scene_revision"]),
               (self.context["scene_id"], self.context["scene_revision"]), "observation_scene_mismatch")
        age = (now - timestamp(observation["captured_at"])).total_seconds()
        require(0 <= age <= self.config["max_observation_age_s"], "observation_stale_or_future")
        require(isinstance(observation["images"], dict) and set(observation["images"]) == set(spec["images"]["keys"]),
                "image_keys_mismatch")
        height, width, channels = spec["images"]["shape_hwc"]
        for pixels in observation["images"].values():
            require(isinstance(pixels, list) and len(pixels) == height, "image_shape_mismatch")
            for row in pixels:
                require(isinstance(row, list) and len(row) == width, "image_shape_mismatch")
                for pixel in row:
                    require(isinstance(pixel, list) and len(pixel) == channels, "image_shape_mismatch")
                    for value in pixel:
                        integer(value)
                        require(value <= 255, "image_pixel_out_of_range")
        _equal(observation["state_units"], spec["state"]["units"], "vector_unit_mismatch")
        require(isinstance(observation["state"], list) and len(observation["state"]) == spec["state"]["dimension"],
                "vector_shape_mismatch")
        for value, low, high in zip(observation["state"], spec["state"]["lower"], spec["state"]["upper"]):
            number(value, low, high)

    def submission_intent(self, event, now):
        require(self.state["submission"] is None, "unresolved_or_reused_submission_no_resubmit")
        require(not self.state["stop_latched"] and self.state["state"] == "AUTHORIZED" and
                not self.state["authorization_consumed"], "valid_unused_authorization_required")
        require(self.adapter.pending_request is None, "probe_or_interaction_freezes_dispatch")
        c = self.current_contract(now)
        auth = self.state["authorization"]
        require(auth is not None and now < timestamp(auth["valid_until"]) and auth["contract_sha256"] == digest(c),
                "authorization_expired_or_contract_changed")
        _equal(auth["model_refs"], self.model_refs, "authorization_model_changed")
        require(self.confirmation is not None and self.filter.snapshot() == self.confirmation["snapshot"],
                "evidence_changed_after_confirmation")
        require((now - timestamp(self.confirmation["snapshot"]["as_of_reference_time"])).total_seconds()
                <= self.config["max_confirmation_age_s"], "confirmation_too_old")
        dispatch_view, dispatch_gap = self.predicted(now, self.confirmation["snapshot"])
        probability = sum(r["probability"] for r in dispatch_view["belief"]["mass"] if r["goal_id"] == c["goal_id"])
        require(probability >= self.config["minimum_goal_probability"], "dispatch_posterior_below_declared_policy")
        _equal(dispatch_gap, event["dispatch_prediction_gap_s"], "replay_dispatch_prediction_gap_mismatch")
        _equal(probability, event["dispatch_goal_probability"], "replay_dispatch_probability_mismatch")
        self.validate_observation(event["observation"], now)
        _equal(digest(event["observation"]), event["observation_sha256"], "replay_observation_hash_mismatch")
        text(event["submission_id"])
        self.submission_fingerprint = digest({"submission_id": event["submission_id"], "observation": event["observation"]})
        self.state.update(authorization_consumed=True, submission_contract=c,
                          submission_at=now.isoformat(), completion_feedback_at=None)
        p = c["proposal"]
        self.state["submission"] = {"submission_id": event["submission_id"], "contract_id": p["contract_id"],
            "revision": p["revision"], "status": "UNKNOWN", "execution_status": "UNKNOWN",
            "actual_execution": False, "last_feedback_id": None}
        self.submission_return_pending = True
        self.change("SUBMISSION_UNKNOWN", "submission_intent_recorded_before_call")

    def reply(self, event):
        require(self.state["submission"] is not None and self.state["authorization_consumed"], "no_submission")
        _equal(event["submission_id"], self.state["submission"]["submission_id"], "feedback_submission_mismatch")
        require(event["actual_execution"] is False, "offline_feedback_only")
        require(event["status"] in {"ACCEPTED", "UNKNOWN", "REJECTED"} and
                event["execution_status"] in {"NOT_STARTED", "UNKNOWN", "STARTED", "COMPLETED", "FAILED", "STOPPED"},
                "invalid_execution_feedback")
        expected = self.state["submission_contract"]["proposal"]
        if event["contract_id"] is not None:
            _equal((event["contract_id"], event["revision"]), (expected["contract_id"], expected["revision"]),
                   "feedback_contract_mismatch")

    def endpoint(self, event, now):
        require(self.state["state"] == "VERIFYING" and not self.state["stop_latched"],
                "execution_completion_required_before_verification")
        evidence = EndpointEvidence(event["evidence"])
        e = evidence.data
        c, submission = self.state["submission_contract"], self.state["submission"]
        p = c["proposal"]
        require(e["evidence_id"] not in self.evidence_ids, "duplicate_endpoint_evidence")
        _equal((e["session_id"], e["submission_id"], e["contract_id"], e["contract_revision"], e["scene_id"], e["scene_revision"]),
               (self.context["session_id"], submission["submission_id"], p["contract_id"], p["revision"],
                p["scene_id"], p["scene_revision"]), "endpoint_context_mismatch")
        low, high = interval(thaw(e["observed_at"]))
        require(e["observed_at"]["host_id"] in self.config["endpoint_hosts"], "unexpected_endpoint_source")
        require(self.state["completion_feedback_at"] is not None and
                timestamp(self.state["submission_at"]) <= timestamp(self.state["completion_feedback_at"]) <= low <= high <= now,
                "endpoint_causal_time_mismatch")
        trajectory = ObservedTrajectory(thaw(e["trajectory"]))
        v = trajectory.data
        require(v["trajectory_id"] not in self.trajectory_ids, "duplicate_endpoint_evidence")
        require(v["provenance"] == "simulation_fixture" and v["executor_id"] == c["backend_id"] and
                v["checkpoint_ref"] is None, "fake_independent_endpoint_source_required")
        _equal(v["predicate_version"], c["completion"]["predicate_version"], "endpoint_predicate_version_mismatch")
        require(v["command_dispatched"] and thaw(v["commanded_intent"]) == p["intent"], "trajectory_command_mismatch")
        result = evaluate_trajectory(trajectory, [{"goal_id": c["goal_id"], "kind": "task", "intent": p["intent"]}])[c["goal_id"]]
        _equal(result, event["result"], "replay_endpoint_result_mismatch")
        self.evidence_ids.add(e["evidence_id"])
        self.trajectory_ids.add(v["trajectory_id"])
        match = result["endpoint_matches_goal"]
        self.change("SUCCEEDED" if match is True else "FAILED" if match is False else "PAUSED",
                    "independent_endpoint_matches" if match is True else "independent_endpoint_fails" if match is False else "endpoint_unknown")

    def apply(self, kind, event, now):
        if kind == "CONTRACT_PROPOSED":
            require(not self.state["stop_latched"] and self.state["state"] not in ACTIVE,
                    "proposal_while_execution_unresolved")
            require(self.adapter.pending_request is None, "interaction_pending")
            c = ExecutionContract(event)
            p = TaskContract(thaw(c.data["proposal"]))
            require(p.data["contract_id"] not in self.proposal_ids, "contract_id_reused")
            _equal((p.data["task_id"], p.data["revision"]),
                   (self.context["task_id"], self.context["task_revision"]), "proposal_task_version_mismatch")
            require(now < timestamp(p.data["valid_until"]), "contract_expired")
            require(goal_id(thaw(p.data["intent"])) in self.filter.vocabulary.known_goal_ids,
                    "proposal_outside_vocabulary")
            require(candidate_key(thaw(p.data["intent"])) in {candidate_key(i) for i in generate_candidates(self.scene.legacy_observation())},
                    "intent_not_in_current_scene_grammar")
            require(p.data["intent"]["action"] in self.config["simulation_supported_actions"], "backend_action_capability_unverified")
            expected = ExecutionContract.from_scene(p, self.scene, backend_id=self.config["backend_id"],
                backend_spec_sha256=self.model_refs["backend_spec_sha256"], predicate_version=self.config["predicate_version"]).to_dict()
            _equal(c.to_dict(), expected, "replay_execution_contract_mismatch")
            self.state["contract"] = expected
            self.proposal_ids.add(p.data["contract_id"])
            self.revoke()
            self.change("CLARIFYING", "proposal_requires_actual_confirmation")
        elif kind == "INTERACTION_REQUESTED":
            self.interaction(event, now)
        elif kind == "INPUT_RECEIVED":
            self.input(event, now)
        elif kind == "INPUT_REJECTED":
            require(event["accepted"] is False and event["request"] is None and event["response"] is None,
                    "rejected_input_must_be_quarantined")
            text(event["reason"])
            number(event["monotonic_now"], 0)
            # Raw rejected input is intentionally absent: no inference/state update.
        elif kind == "AUTHORIZATION_GRANTED":
            self.authorize(event, now)
        elif kind == "SUBMISSION_INTENT":
            self.submission_intent(event, now)
        elif kind == "SUBMISSION_RETURNED":
            require(self.submission_return_pending and not self.state["stop_latched"], "submission_return_without_intent")
            self.reply(event)
            self.submission_return_pending = False
            self.state["submission"] = event
            self.change({"ACCEPTED": "SUBMITTED", "UNKNOWN": "SUBMISSION_UNKNOWN", "REJECTED": "FAILED"}[event["status"]],
                        "transport_acceptance_is_not_execution" if event["status"] == "ACCEPTED" else event["reason"])
        elif kind == "SUBMISSION_UNKNOWN":
            require(self.submission_return_pending and event["submission_id"] == self.state["submission"]["submission_id"],
                    "submission_return_without_intent")
            self.submission_return_pending = False
            self.state["reason"] = "submit_exception_unknown_no_retry"
        elif kind == "DUPLICATE_DISPATCH_IGNORED":
            require(self.state["submission"] is not None and event["submission_id"] == self.state["submission"]["submission_id"],
                    "duplicate_dispatch_without_submission")
        elif kind == "SUBMISSION_QUERY_FAILED":
            require(self.state["submission"] is not None and event["submission_id"] == self.state["submission"]["submission_id"],
                    "query_without_submission")
            self.change("PAUSED" if self.state["stop_latched"] else "SUBMISSION_UNKNOWN", "query_exception_unknown_no_resubmit")
        elif kind in {"SUBMISSION_QUERY_UNKNOWN", "SUBMISSION_QUERIED_NO_NEW_FEEDBACK", "SIMULATED_EXECUTION_FEEDBACK"}:
            self.reply(event)
            old_execution_status = self.state["submission"]["execution_status"]
            self.state["submission"] = event
            feedback = event["last_feedback_id"]
            if kind == "SUBMISSION_QUERY_UNKNOWN":
                require(event["status"] == "UNKNOWN" or event["execution_status"] == "UNKNOWN", "query_not_unknown")
                self.change("PAUSED" if self.state["stop_latched"] else "SUBMISSION_UNKNOWN", "query_unknown_no_resubmit")
            elif kind == "SUBMISSION_QUERIED_NO_NEW_FEEDBACK":
                require(feedback is None or feedback in self.feedback_ids, "feedback_is_new")
            else:
                require(feedback is not None and feedback not in self.feedback_ids, "duplicate_execution_feedback")
                status = event["execution_status"]
                require(status in {"STARTED", "COMPLETED", "FAILED", "STOPPED"}, "invalid_execution_feedback")
                require(old_execution_status not in {"COMPLETED", "FAILED", "STOPPED"}, "terminal_execution_state")
                if status == "STARTED":
                    require(old_execution_status in {"NOT_STARTED", "UNKNOWN"}, "invalid_execution_transition")
                elif status == "COMPLETED":
                    require(old_execution_status == "STARTED", "completion_without_start")
                self.feedback_ids.add(feedback)
                if status == "COMPLETED":
                    self.state["completion_feedback_at"] = now.isoformat()
                if status == "STOPPED":
                    self.state["stop_feedback"] = "SIMULATED_STOPPED"
                    self.change("CANCELLED", "fake_stop_feedback_received")
                elif self.state["stop_latched"]:
                    self.change("PAUSED", "stop_latched_waiting_for_stop_feedback")
                elif not self.state["generation_valid"]:
                    self.change("PAUSED", "old_context_feedback_no_new_authority")
                else:
                    self.change({"STARTED": "EXECUTING", "COMPLETED": "VERIFYING", "FAILED": "FAILED"}[status],
                                {"STARTED": "fake_execution_started", "COMPLETED": "endpoint_verification_required",
                                 "FAILED": "execution_failed_not_decoder_label"}[status])
        elif kind == "ENDPOINT_JUDGED":
            self.endpoint(event, now)
        elif kind == "STOP_REQUESTED":
            require(event["physical_stop_verified"] is False, "offline_stop_not_physical_proof")
            require(event["reason"] in {"stop_requested", "cancel_requested", "deadline_expired_stop_requested",
                                        "context_changed_stop_requested", "models_changed_stop_requested"}, "invalid_stop_reason")
            if event["reason"] in {"deadline_expired_stop_requested", "context_changed_stop_requested", "models_changed_stop_requested"}:
                require(self.state["state"] in ACTIVE, "causal_stop_without_active_submission")
            if event["reason"] == "deadline_expired_stop_requested":
                self.expiry_stop_pending = True
            self.state.update(stop_latched=True, stop_feedback="REQUESTED")
            self.revoke()
            reason = event["delivery_reason"]
            require(reason in {event["reason"], "stop_request_delivery_unknown"}, "invalid_stop_delivery_result")
            self.change("CANCELLED" if self.state["submission"] is None and event["reason"] == "cancel_requested" else "PAUSED", reason)
        elif kind == "CONTROL_RECEIVED":
            require(self.state["stop_latched"] and event["accepted"] is True and event["response"] is not None,
                    "stop_control_requires_latched_stop")
            result = self.adapter.ingest(event["response"], now=now, monotonic_now=event["monotonic_now"])
            require(result.accepted and result.response.data["mode"] in {"STOP", "CANCEL"}, "control_not_accepted")
            _equal(result.reason, event["reason"], "replay_input_reason_mismatch")
        elif kind == "EXPIRED_OR_TIMED_OUT":
            timeout = self.adapter.expire_pending(monotonic_now=event["monotonic_now"])
            expired = self.state["contract"] is not None and now >= timestamp(self.state["contract"]["proposal"]["valid_until"])
            _equal((expired, timeout.accepted), (event["expired"], event["interaction_timeout"]), "replay_expiry_mismatch")
            require(expired or timeout.accepted, "no_expiry_or_timeout")
            self.revoke()
            if not self.expiry_stop_pending:
                self.change("EXPIRED" if expired else "PAUSED", "contract_expired" if expired else "interaction_timeout")
            self.expiry_stop_pending = False
        elif kind == "CONTEXT_INVALIDATED":
            require(self.state["state"] not in ACTIVE, "active_context_change_requires_stop")
            fresh = dict(context(event["context"]))
            scene = GroundedScene(event["scene"])
            _equal(fresh["session_id"], self.context["session_id"], "new_session_required")
            _equal((fresh["scene_id"], fresh["scene_revision"]), (scene.data["scene_id"], scene.data["revision"]), "scene_context_mismatch")
            _equal(scene.data["identity"]["session_id"], fresh["session_id"], "scene_identity_session_mismatch")
            if scene.data["scene_id"] == self.scene.data["scene_id"]:
                if scene.data["revision"] == self.scene.data["revision"]:
                    _equal(scene.to_dict(), self.scene.to_dict(), "scene_version_reused")
                else:
                    ledger = IdentityLedger()
                    ledger.observe(self.scene)
                    ledger.observe(scene)
            self.adapter.update_context(fresh)
            self.context, self.scene = fresh, scene
            self.state.update(context=dict(fresh), generation_valid=False)
            self.revoke()
            self.change("PAUSED", "context_changed_explicit_filter_policy_rebuild_required")
        elif kind == "MODELS_INVALIDATED":
            require(self.state["state"] not in ACTIVE, "active_model_change_requires_stop")
            self.revoke()
            self.state["generation_valid"] = False
            self.change("PAUSED", "models_require_explicit_rebuild")
        elif kind == "POLICY_RETURNED":
            require(not self.state["stop_latched"] and self.state["generation_valid"], "policy_generation_invalid")
            snap, gap = self.predicted(now)
            _equal(gap, event["planning_prediction_gap_s"], "replay_planning_gap_mismatch")
            inputs = event["inputs"]
            _equal(inputs["used_s"], (now - self.start).total_seconds(), "replay_policy_elapsed_time_mismatch")
            contracts = {} if self.state["contract"] is None else {
                self.state["contract"]["goal_id"]: self.state["contract"]["proposal"]}
            planning = PlanningState.from_filter_snapshot(snap, scene_id=self.context["scene_id"],
                scene_revision=self.context["scene_revision"], task_id=self.context["task_id"],
                task_revision=self.context["task_revision"], now_s=now.timestamp(), used_s=inputs["used_s"],
                remaining_s=inputs["remaining_s"], scene_valid_until_s=inputs["scene_valid_until_s"],
                contracts=contracts, pending=self.adapter.pending_request is not None)
            _equal(self.policy.plan(planning, depth=inputs["depth"]), event["decision"], "replay_policy_decision_mismatch")
        else:
            require(False, "unsupported_session_replay_event")


def replay_session_state(journal):
    """Rebuild a public offline session, checking every checkpoint independently.

    No VLA adapter is imported, constructed, queried, stopped or submitted. An
    imported valid shorter prefix remains a prefix; no complete-run claim is
    possible without a separately retained terminal anchor.
    """
    require(isinstance(journal, EventJournal), "validated_journal_required")
    engine = None

    def reduce(previous, record):
        nonlocal engine
        payload = thaw(record["payload"])
        fields(payload, {"event", "state", "state_sha256"})
        _equal(digest(payload["state"]), payload["state_sha256"], "session_snapshot_hash_mismatch")
        now = timestamp(record["at"])
        if engine is None:
            require(record["kind"] == "SESSION_STARTED", "session_start_required")
            event = payload["event"]
            require(event["scope"] == "offline_fake_only", "offline_session_only")
            engine = _Replay(event["replay_bundle"], now, journal.session_id)
        else:
            require(record["kind"] != "SESSION_STARTED", "duplicate_session_start")
            engine.apply(record["kind"], payload["event"], now)
        expected = engine.snapshot()
        _equal(expected, payload["state"], "session_replay_checkpoint_mismatch")
        return expected

    return journal.replay(reduce, None)
