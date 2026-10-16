"""Standard finite-horizon expected-loss search, not an original POMDP solver.

Only finite illustrative offline models are admitted. A result requests an action;
even a simulated confirmation accept branch cannot authorize a robot.
"""
from dataclasses import dataclass
import math

from ..protocols import JointBelief, TaskContract, freeze, thaw, number, integer, text, require, timestamp
from ..scene.tasks import goal_id
from .diagnostic_actions import ActionCatalog


@dataclass(frozen=True)
class PlanningState:
    belief: JointBelief
    scene_id: str
    scene_revision: int
    task_id: str
    task_revision: int
    now_s: float
    used_s: float
    remaining_s: float
    scene_valid_until_s: float
    contracts: object
    stopped: bool = False
    pending: bool = False
    retry_allowed: bool = False
    extension_allowed: bool = False
    inference_ready: bool = True

    def __post_init__(self):
        require(isinstance(self.belief, JointBelief), "validated_joint_belief_required")
        for k in ("scene_id", "task_id"):
            text(getattr(self, k))
        for k in ("scene_revision", "task_revision"):
            integer(getattr(self, k))
        for k in ("now_s", "used_s", "remaining_s", "scene_valid_until_s"):
            number(getattr(self, k), 0)
        for k in ("stopped", "pending", "retry_allowed", "extension_allowed", "inference_ready"):
            require(type(getattr(self, k)) is bool, "invalid_planning_flag")
        checked = {}
        for g, raw in self.contracts.items():
            contract = TaskContract(raw).to_dict()
            require(contract["status"] == "proposed", "proposed_contract_required")
            require((contract["scene_id"], contract["scene_revision"], contract["task_id"], contract["revision"]) ==
                    (self.scene_id, self.scene_revision, self.task_id, self.task_revision), "contract_context_mismatch")
            require(goal_id(contract["intent"]) == g, "contract_goal_mismatch")
            checked[g] = {"contract": contract, "valid_until_s": timestamp(contract["valid_until"]).timestamp()}
        object.__setattr__(self, "contracts", freeze(checked))

    @classmethod
    def from_filter_snapshot(cls, snapshot, **context):
        require(abs(timestamp(snapshot["as_of_reference_time"]).timestamp() - context["now_s"]) < 1e-6,
                "belief_time_mismatch_predict_before_planning")
        require(snapshot["execution_allowed"] is False, "inference_cannot_authorize_execution")
        return cls(belief=JointBelief(snapshot["belief"]),
                   inference_ready=snapshot["point_inference_ready"],
                   stopped=snapshot["status"] == "stopped", **context)

    def advance(self, elapsed_s):
        """Internal copy bypasses parsing already validated wrapped contracts."""
        require(0 <= elapsed_s <= self.remaining_s + 1e-12, "insufficient_budget")
        obj = object.__new__(PlanningState)
        for field in self.__dataclass_fields__:
            object.__setattr__(obj, field, getattr(self, field))
        object.__setattr__(obj, "now_s", self.now_s + elapsed_s)
        object.__setattr__(obj, "used_s", self.used_s + elapsed_s)
        object.__setattr__(obj, "remaining_s", max(0, self.remaining_s - elapsed_s))
        object.__setattr__(obj, "retry_allowed", False)
        object.__setattr__(obj, "extension_allowed", False)
        return obj


class DiagnosticPolicy:
    def __init__(self, config, *, channel_model, transition_model, outcome_table):
        require(config["status"] == "simulation_only" and config["deployment_allowed"] is False,
                "offline_policy_only")
        require(config["schema_version"] == "diagnosis/1.0", "unsupported_diagnosis_schema")
        require(config["provenance"] == "illustrative_not_fitted", "offline_policy_only")
        self.config = freeze(config)
        self.channel = channel_model
        self.transition = transition_model
        self.outcomes = outcome_table
        self.goals = tuple(transition_model.data["goal_ids"])
        self.modes = tuple(transition_model.data["mode_ids"])
        require(self.modes == tuple(channel_model.data["mode_ids"]), "mode_order_mismatch")
        self.states = tuple((g, z) for g in self.goals for z in self.modes)
        require(set(outcome_table.data["goal_ids"]) == set(self.goals), "outcome_goal_mismatch")
        require(set(outcome_table.data["channel_mode_ids"]) == set(self.modes), "outcome_mode_mismatch")
        require(set(config["decision_by_goal"]) == set(self.goals), "decision_binding_mismatch")
        require(type(config["max_depth"]) is int and 0 <= config["max_depth"] <= 3, "bounded_depth_required")
        for key in ("pause_loss", "exit_loss"):
            number(config[key], 0)
        require(config["pause_loss"] <= config["exit_loss"], "pause_exit_order")
        require(set(config["weights"]) == {"wrong_start", "task_failure", "recovery_s", "user_time_s"}, "cost_weights_required")
        for v in config["weights"].values():
            number(v, 0)
        require(config["weights"]["user_time_s"] == config["actions"]["time_weight_per_s"], "inconsistent_time_weights")
        self.catalog = ActionCatalog(config["actions"], channel_model, self.goals)
        require(config["context"]["channel_profile_id"] == channel_model.data["profile_id"], "profile_mismatch")
        require(all(config["context"][k] == config["actions"]["scope"][k] for k in ("mapping_id", "stimulus_version")), "policy_mapping_mismatch")
        # Terminal rows include a historical fixed confirmation allowance. Strip
        # it explicitly; actual candidate confirmation time is charged exactly once.
        for goal in self.goals:
            for g, z in self.states:
                row = outcome_table.get(config["decision_by_goal"][goal], g, z)
                require(row["stage"] == "terminal" and set(row["included_time_phases"]) == {"confirmation"}, "exclusive_confirmation_accounting_required")
                require(all(v is not None for v in row["components"].values()), "complete_terminal_cost_required")

    def _vector(self, belief):
        cells = {(r["goal_id"], r["mode_id"]): r["probability"] for r in belief.data["mass"]}
        require(all(not p or key in self.states for key, p in cells.items()), "unmodeled_support_pause")
        return tuple(cells.get(key, 0) for key in self.states)

    def _predict(self, b, elapsed_s):
        t = self.transition.matrix(elapsed_s)
        return tuple(math.fsum(b[i] * t[i][j] for i in range(len(b))) for j in range(len(b)))

    def _terminal_loss(self, target, g, z):
        row = self.outcomes.get(self.config["decision_by_goal"][target], g, z)
        return math.fsum(v * self.config["weights"][k] for k, v in row["components"].items() if k != "user_time_s")

    def _branches(self, b, a):
        # Setup elapses before acquisition and uses T(setup). During the declared
        # stable window T=I, exactly as in step08. Total time still spends budget.
        prior = self._predict(b, a.setup_s)
        mode = "TASK_QUERY" if a.kind in {"RETRY", "EXTEND"} else a.kind
        symbols = self.channel.data["scope"]["symbols"]
        reject = next(s for s in symbols if s != self.catalog.config["accept_symbol"])
        branches = []
        for y in (*symbols, "NO_RESPONSE"):
            likelihood = []
            for g, z in self.states:
                intended = (a.prompt if mode == "CHANNEL_PROBE" else
                            self.catalog.config["goal_symbols"][g] if mode == "TASK_QUERY" else
                            self.catalog.config["accept_symbol"] if g == a.goal_id else reject)
                values = self.channel.likelihoods(mode, intended, y, scope=thaw(self.catalog.config["scope"]),
                    duration_s=a.duration_s, compliance_id=self.catalog.config["compliance_id"] if mode == "CHANNEL_PROBE" else None,
                    assumption="state_constant_within_window")
                likelihood.append(values[z])
            mass = tuple(p * l for p, l in zip(prior, likelihood))
            p = math.fsum(mass)
            if p:
                branches.append({"observation": y, "probability": p,
                                 "posterior": [v / p for v in mass]})
        require(abs(math.fsum(r["probability"] for r in branches) - 1) < 1e-9, "predictive_not_normalized")
        return branches

    def _tv(self, a):
        if a.kind != "CHANNEL_PROBE":
            return None
        rows = []
        for y in (*self.channel.data["scope"]["symbols"], "NO_RESPONSE"):
            rows.append(self.channel.likelihoods("CHANNEL_PROBE", a.prompt, y,
                scope=thaw(self.catalog.config["scope"]), duration_s=a.duration_s,
                compliance_id=self.catalog.config["compliance_id"], assumption="state_constant_within_window"))
        return max(math.fsum(abs(r[x] - r[y]) for r in rows) / 2
                   for x in self.modes for y in self.modes)

    def _evaluate(self, b, state, depth, *, diagnostics=True):
        candidates = []
        for a in self.catalog.candidates():
            row = a.record()
            row.update(legal=False, reason=None, expected_loss=None, branches=[],
                       immediate_time_cost=a.elapsed_s * self.catalog.config["time_weight_per_s"])
            reason = self.catalog.legality(a, state)
            if not reason and a.kind not in {"PAUSE", "EXIT", "CONFIRM"} and (depth == 0 or not diagnostics):
                reason = "diagnostic_horizon_exhausted"
            if reason:
                row["reason"] = reason
                candidates.append(row)
                continue
            if a.kind in {"PAUSE", "EXIT"}:
                row.update(legal=True, expected_loss=self.config["pause_loss" if a.kind == "PAUSE" else "exit_loss"])
                candidates.append(row)
                continue
            try:
                tv = self._tv(a)
                if tv is not None:
                    row["channel_total_variation"] = tv
                    if tv <= 1e-12:
                        row["reason"] = "probe_cannot_distinguish_modes"
                        candidates.append(row)
                        continue
                branches = self._branches(b, a)
            except ValueError as exc:
                row["reason"] = str(exc)
                candidates.append(row)
                continue
            after = state.advance(a.elapsed_s)
            expected = row["immediate_time_cost"]
            for branch in branches:
                posterior = branch["posterior"]
                if a.kind == "CONFIRM":
                    if branch["observation"] == self.catalog.config["accept_symbol"]:
                        value = math.fsum(p * self._terminal_loss(a.goal_id, g, z) for p, (g, z) in zip(posterior, self.states))
                        next_id = "SIMULATED_ACCEPT_REQUIRES_EXTERNAL_AUTHORIZATION"
                    else:
                        value, next_id = self.config["pause_loss"], "PAUSE_CONFIRMATION_NOT_ACCEPTED"
                elif branch["observation"] == "NO_RESPONSE":
                    # Nonresponse is informative but never implicitly retries or executes.
                    value, next_id = self.config["pause_loss"], "PAUSE_NONRESPONSE"
                else:
                    best, _ = self._evaluate(tuple(posterior), after, depth - 1)
                    value, next_id = best["expected_loss"], best["action_id"]
                branch.update(continuation_loss=value, next_action=next_id,
                              after_time_s=after.now_s, remaining_s=after.remaining_s)
                expected += branch["probability"] * value
            row.update(legal=True, expected_loss=expected, branches=branches,
                       reason="standard_expected_loss_with_noisy_confirmation")
            candidates.append(row)
        legal = [r for r in candidates if r["legal"]]
        # PAUSE wins equal-cost ties, so pure information without decision value
        # cannot cause an endless dialogue at zero declared time price.
        best = min(legal, key=lambda r: (round(r["expected_loss"], 12), r["kind"] != "PAUSE", r["action_id"]))
        return best, candidates

    def plan(self, state, *, depth=1, execution_mode="offline"):
        require(execution_mode == "offline", "offline_policy_only")
        require(isinstance(state, PlanningState), "validated_planning_state_required")
        require(type(depth) is int and 0 <= depth <= self.config["max_depth"], "bounded_depth_required")
        context = self.config["context"]
        require((state.scene_id, state.scene_revision, state.task_id, state.task_revision) ==
                (context["scene_id"], context["scene_revision"], context["task_id"], context["task_revision"]), "policy_context_mismatch")
        fail = ("stop_latched" if state.stopped else "collection_pending_no_menu_change" if state.pending else
                "filter_not_ready" if not state.inference_ready else
                "scene_expired" if state.now_s >= state.scene_valid_until_s else None)
        try:
            b = self._vector(state.belief)
        except ValueError:
            b, fail = None, fail or "unmodeled_support_pause"
        base, _ = self._evaluate(b, state, 0) if not fail else (None, None)
        if fail:
            # Preserve the input belief for explanation; never renormalize away unknowns.
            candidates = [dict(a.record(), legal=a.kind in {"PAUSE", "EXIT"}, reason=fail,
                               expected_loss=self.config["pause_loss"] if a.kind == "PAUSE" else None,
                               branches=[]) for a in self.catalog.candidates()]
            best = next(r for r in candidates if r["kind"] == "PAUSE")
            pure_choices = []
        else:
            pure_choices = []
            for index, probability in enumerate(b):
                if probability:
                    point = tuple(float(i == index) for i in range(len(b)))
                    choice, _ = self._evaluate(point, state, 0)
                    pure_choices.append({"state": list(self.states[index]), "terminal_choice": choice["action_id"]})
            all_pause = all(r["terminal_choice"] in {"pause", "exit"} for r in pure_choices)
            best, candidates = self._evaluate(b, state, depth, diagnostics=not all_pause)
            if all_pause:
                fail = "all_supported_explanations_require_pause"
        return {"scope": "offline_illustrative_policy_request", "method": "one_step_voi" if depth == 1 else "bounded_candidate_search",
                "depth": depth, "state_order": [list(x) for x in self.states],
                "selected_action": best, "candidates": candidates, "reason": fail or best["reason"],
                "baseline_loss": None if base is None else base["expected_loss"],
                "value_over_no_diagnosis": None if base is None else base["expected_loss"] - best["expected_loss"],
                "context": {"scene_id": state.scene_id, "scene_revision": state.scene_revision,
                    "task_id": state.task_id, "task_revision": state.task_revision,
                    "contracts": thaw(state.contracts), "now_s": state.now_s, "used_s": state.used_s,
                    "remaining_s": state.remaining_s, "scene_valid_until_s": state.scene_valid_until_s},
                "unresolved_explanations": state.belief.to_dict(), "pure_state_choices": pure_choices,
                "execution_allowed": False, "time_accounting": "charge_elapsed_once_strip_terminal_confirmation_allowance",
                "recommendations": ["Request separate recalibration assessment; no repair benefit assumed"] if best["kind"] == "PAUSE" else [],
                "future_task_value": "not_quantified_without_declared_future_task_model_use_channel_handoff"}

    def carry_channel_to_new_task(self, belief, new_goal_prior):
        """Explicit new independent goal prior, retained channel marginal.

        Caller must supply fresh scene/contracts/context to plan the next task.
        This is a handoff, not implicit filter resume or a future value bonus.
        """
        b = self._vector(belief)
        require(set(new_goal_prior) == set(self.goals), "new_task_prior_required")
        for p in new_goal_prior.values():
            number(p, 0, 1)
        require(abs(math.fsum(new_goal_prior.values()) - 1) < 1e-9, "new_task_prior_not_normalized")
        zprior = {z: math.fsum(p for p, (_, mode) in zip(b, self.states) if mode == z) for z in self.modes}
        return JointBelief({"goal_ids": list(self.goals), "mode_ids": list(self.modes),
            "mass": [{"goal_id": g, "mode_id": z, "probability": new_goal_prior[g] * zprior[z]} for g, z in self.states],
            "as_of_event_id": "explicit_new_task_prior_with_retained_channel"})
