"""Offline diagnostic candidates. No acquisition, display mutation or dispatch."""
from dataclasses import dataclass

from ..protocols import freeze, number, require


@dataclass(frozen=True)
class DiagnosticAction:
    action_id: str
    kind: str
    duration_s: float
    setup_s: float = 0.0
    prompt: str | None = None
    goal_id: str | None = None

    @property
    def elapsed_s(self):
        return self.setup_s + self.duration_s

    def record(self):
        return {"action_id": self.action_id, "kind": self.kind,
                "duration_s": self.duration_s, "setup_s": self.setup_s,
                "elapsed_s": self.elapsed_s, "prompt": self.prompt,
                "goal_id": self.goal_id}


class ActionCatalog:
    """A finite simulated calibration registry; no interpolation or remapping.

    RETRY is a new disjoint acquisition. EXTEND is one longer *final* window,
    selected before collection; intermediate evidence cannot be reused.
    """
    def __init__(self, config, channel, goals):
        self.channel = channel
        self.config = freeze(config)
        self.goals = tuple(goals)
        for key in ("setup_s", "time_weight_per_s"):
            number(config[key], 0)
        for d in config["durations_s"]:
            channel.ensure_applicable(config["scope"], d,
                                      assumption="state_constant_within_window")
        require(bool(config["durations_s"]), "missing_diagnostic_durations")
        require(len(set(config["durations_s"])) == len(config["durations_s"]), "duplicate_duration")
        require(config["confirmation_duration_s"] in config["durations_s"], "confirmation_duration_unavailable")
        require(set(config["goal_symbols"]) == set(goals), "goal_binding_mismatch")
        require(set(config["goal_symbols"].values()) == set(channel.data["scope"]["symbols"]), "goal_binding_mismatch")
        require(config["accept_symbol"] in channel.data["scope"]["symbols"], "invalid_accept_symbol")
        require(config["compliance_id"] in channel.data["compliance"], "unknown_compliance_model")

    def candidates(self):
        c = self.config
        result = [DiagnosticAction("pause", "PAUSE", 0),
                  DiagnosticAction("exit", "EXIT", 0),
                  DiagnosticAction("recalibrate", "RECOMMEND_RECALIBRATION", 0),
                  DiagnosticAction("recode", "RECODE", 0),
                  DiagnosticAction("execute", "EXECUTE_CONTRACT", 0)]
        for g in self.goals:
            result.append(DiagnosticAction("confirm:" + g, "CONFIRM", c["confirmation_duration_s"], c["setup_s"], goal_id=g))
        for index, d in enumerate(c["durations_s"]):
            for kind in (["TASK_QUERY", "RETRY"] if index == 0 else ["EXTEND"]):
                result.append(DiagnosticAction(f"{kind}:{d}", kind, d, c["setup_s"]))
            for prompt in self.channel.data["scope"]["symbols"]:
                result.append(DiagnosticAction(f"probe:{prompt}:{d}", "CHANNEL_PROBE", d, c["setup_s"], prompt=prompt))
        return tuple(result)

    def legality(self, action, state):
        if action.kind in {"PAUSE", "EXIT"}:
            return None
        if state.stopped:
            return "stop_latched"
        if state.pending:
            return "collection_pending_no_menu_change"
        if action.kind == "RECODE":
            return "no_separately_calibrated_alternative_mapping"
        if action.kind == "RECOMMEND_RECALIBRATION":
            return "recommendation_available_but_future_repair_model_unavailable"
        if action.kind == "EXECUTE_CONTRACT":
            return "separate_confirmation_and_authorization_required"
        if action.elapsed_s > state.remaining_s + 1e-12:
            return "insufficient_budget"
        if state.now_s + action.elapsed_s >= state.scene_valid_until_s:
            return "scene_expired_during_action"
        if action.kind == "RETRY" and not state.retry_allowed:
            return "retry_requires_fresh_disjoint_window"
        if action.kind == "EXTEND" and not state.extension_allowed:
            return "extension_requires_unconsumed_final_window"
        if action.kind == "CONFIRM":
            contract = state.contracts.get(action.goal_id)
            if contract is None:
                return "missing_proposed_contract"
            if state.now_s + action.elapsed_s >= contract["valid_until_s"]:
                return "contract_expired_during_confirmation"
        return None
