"""Public-only interaction proposals; no calibrated confidence authorizes actions."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from pathlib import Path

from bci_vla.data_v4.contracts import validate_public
from bci_vla.interaction_contract_v4.validation import canonical_digest, validate_document
from bci_vla.neural_interaction_v4.control import route_control
from .calibration import validate_binding

VARIANTS = ("fixed_threshold", "entropy_threshold", "auxiliary")


def policy_hash(config):
    """Hash the complete frozen config; binding is kept outside to avoid cycles."""
    return canonical_digest(config)


def _result(action, reason, **extra):
    return {"action": action, "reason": reason, "execution_authorized": False,
            "requires_user_confirmation": True, **extra}


def _validate_config(config):
    implementation = config.get("implementation_binding")
    if implementation is not None:
        required = {"bci_vla/calibration_v4/policy.py", "bci_vla/evaluation_v4/runner.py",
                    "bci_vla/data_v4/channel.py", "bci_vla/calibration_v4/calibration.py",
                    "bci_vla/calibration_v4/evidence_gate.py",
                    "execution/v4/step09_calibrate_and_policy.py"}
        if not isinstance(implementation, dict) or set(implementation) != required:
            raise ValueError("exact policy implementation binding required")
        root = Path(__file__).resolve().parents[2]
        for relative, expected in implementation.items():
            actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
            if expected != actual:
                raise ValueError("policy implementation changed; refit and re-audit required: " + relative)
    thresholds, budgets = config["thresholds"], config["budgets"]
    for key in ("intent_confidence", "normalized_entropy", "channel_success", "outside_probability"):
        if type(thresholds[key]) not in (int, float) or not math.isfinite(thresholds[key]) or not 0 <= thresholds[key] <= 1:
            raise ValueError("policy thresholds must lie in [0,1]")
    for key in ("questions", "probes", "intent_queries", "retries"):
        if type(budgets[key]) is not int or budgets[key] < 0:
            raise ValueError("nonnegative integer interaction budgets required")
    if not math.isfinite(config["selection_freshness_s"]) or config["selection_freshness_s"] <= 0:
        raise ValueError("positive selection freshness required")


def decide_interaction(public, prediction, config, *, variant=None, binding=None, state=None):
    """Return runner-compatible TASK/INTENT_QUERY/PROBE/CONFIRM/WAIT only.

    prediction holds candidates+outside probabilities, channel probability and
    explicit calibrated/support flags; plus exact public hash/context/as_of_s
    and the four calibration binding hashes. state optionally carries cumulative
    questions/probes/intent_queries/retries; it never carries latent targets.
    """
    route = route_control(public)
    if not route["call_neural_module"]:
        return _result("WAIT", "DIRECT_CONTROL_BYPASS", control_route=route["route"], call_neural_module=False)
    if public.get("schema_version") == "bci-vla-interaction/4.1":
        validate_public(public)
    else:
        validate_document(public)
    if public["public_state"]["pending_question"] is not None:
        return _result("WAIT", "QUESTION_PENDING", call_neural_module=False)
    _validate_config(config)
    variant = config.get("variant", "fixed_threshold") if variant is None else variant
    if variant not in VARIANTS:
        raise ValueError("unknown interaction policy variant")
    if variant != config.get("variant", "fixed_threshold"):
        raise ValueError("policy variant differs from the hash-bound config")
    expected = validate_binding(binding)
    if expected["policy_sha256"] != policy_hash(config):
        raise ValueError("policy config changed; recalibration and policy re-audit required")
    if validate_binding(prediction.get("binding")) != expected:
        raise ValueError("stale prediction model/data/schema/policy binding")
    if (prediction.get("input_sha256") != canonical_digest(public) or prediction.get("context") != public["context"]
            or prediction.get("as_of_s") != public["as_of_s"]):
        return _result("WAIT", "STALE_PREDICTION_OR_MAPPING", refresh_required=True)
    history = public["history"]
    if state is None:
        if any(e["sequence"] > 8 for e in history):
            return _result("WAIT", "CUMULATIVE_BUDGET_STATE_REQUIRED")
        state = {"questions": len(history), "probes": sum((e.get("question") or {}).get("role") == "PROBE" for e in history),
                 "intent_queries": sum((e.get("question") or {}).get("role") == "INTENT_QUERY" for e in history), "retries": 0}
    if set(state) != {"questions", "probes", "intent_queries", "retries"} or any(type(v) is not int or v < 0 for v in state.values()):
        raise ValueError("cumulative public interaction counters required")
    budgets, thresholds = config["budgets"], config["thresholds"]
    if state["questions"] >= budgets["questions"] or state["retries"] > budgets["retries"]:
        return _result("WAIT", "INTERACTION_BUDGET_EXHAUSTED")
    ids = [o["instance_id"] for o in public["scene"]["objects"] if o["object_valid"]]
    if not ids:
        return _result("WAIT", "NO_VISIBLE_CANDIDATES", refresh_required=True)
    if prediction.get("candidate_ids") != ids:
        return _result("WAIT", "CANDIDATE_AXIS_MISMATCH", refresh_required=True)
    def ask(role, reason):
        key = "probes" if role == "PROBE" else "intent_queries" if role == "INTENT_QUERY" else None
        if key and state[key] >= budgets[key]:
            return _result("WAIT", reason + "_BUDGET_EXHAUSTED")
        return _result(role, reason)
    diagnostic = public["public_state"]["diagnostics"]
    channel = prediction.get("channel_probability")
    channel_ready = (diagnostic["status"] == "SUFFICIENT" and prediction.get("channel_supported") is True
                     and prediction.get("channel_calibrated") is True and channel is not None)
    if channel_ready and (not isinstance(channel, (float, int)) or not math.isfinite(channel) or not 0 <= channel <= 1):
        raise ValueError("invalid calibrated channel success probability")
    if not channel_ready or channel < thresholds["channel_success"]:
        return ask("PROBE", "CHANNEL_CHECK_REQUIRED")
    p = prediction.get("probabilities")
    if prediction.get("intent_calibrated") is not True or p is None:
        return ask("INTENT_QUERY", "INTENT_CALIBRATION_UNSUPPORTED")
    if (not isinstance(p, (list, tuple)) or len(p) != len(ids) + 1
            or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or v > 1 for v in p)
            or not math.isclose(sum(p), 1., abs_tol=1e-6)):
        raise ValueError("complete candidates-plus-outside calibrated probability axis required")
    events = [e for e in history if e["valid"] and e["role"] in ("TASK", "INTENT_QUERY")]
    if not events or public["as_of_s"] - events[-1]["ended_at_s"] >= config["selection_freshness_s"]:
        return ask("TASK", "FRESH_TASK_EVIDENCE_REQUIRED")
    if any(e["valid"] and e["role"] == "REJECT"
           and e["ended_at_s"] > events[-1]["ended_at_s"] for e in history):
        return ask("INTENT_QUERY", "REJECTED_PROPOSAL_REQUIRES_NEW_SELECTION")
    if p[-1] >= thresholds["outside_probability"]:
        return _result("WAIT", "TARGET_MAY_BE_OUTSIDE_CANDIDATES", refresh_required=True)
    source_index = max(range(len(ids)), key=lambda j: p[j])
    confidence = p[source_index]
    entropy = -sum(v * math.log(v) for v in p if v > 0) / math.log(len(p))
    unclear = (confidence < thresholds["intent_confidence"] if variant != "entropy_threshold"
               else entropy > thresholds["normalized_entropy"])
    if unclear:
        return ask("INTENT_QUERY", "INTENT_AMBIGUOUS")
    if variant == "auxiliary":
        suggestion = prediction.get("auxiliary_action")
        if suggestion == "ASK_INTENT":
            return ask("INTENT_QUERY", "AUXILIARY_INTENT_SUGGESTION")
        if suggestion == "CHECK_CHANNEL":
            return ask("PROBE", "AUXILIARY_CHANNEL_SUGGESTION")
        if suggestion == "WAIT":
            return _result("WAIT", "AUXILIARY_WAIT_SUGGESTION")
        if suggestion != "PROPOSE_TASK":
            return _result("WAIT", "AUXILIARY_SUGGESTION_UNSUPPORTED")
    source = ids[source_index]
    confirmed = public["public_state"]["confirmed_task"]
    if confirmed and confirmed["task"]["source_id"] == source:
        return _result("WAIT", "CONFIRMED_TASK_REQUIRES_SEPARATE_EXECUTION_AUTHORIZATION", source_id=source)
    from .evidence_gate import confirmation_evidence_gate
    gate = confirmation_evidence_gate(public, source, config)
    if gate is not None:
        if gate['action'] == 'PROBE':
            return ask('PROBE', gate['reason'])
        if gate['action'] == 'INTENT_QUERY':
            return ask('INTENT_QUERY', gate['reason'])
        return gate
    return _result("CONFIRM", "CONFIRM_EXACT_TASK_SUGGESTION", source_id=source)


def select_policy(rows, *, candidates, max_wrong_episode_rate=.1, min_resolved_fraction=0.,
                  resolution_field='resolved_correctly'):
    """Choose only on validation outcome summaries; policies never see truth.

    Rows: split, policy_id, group_id, episode_id, resolved_correctly(bool),
    wrong_submission(bool), interaction_cost>=0. One row per policy/episode,
    including unresolved/no-response/wait episodes. Risk uses any wrong
    submission in an episode. These are point estimates, not risk guarantees.
    """
    if not 0 <= max_wrong_episode_rate <= 1 or not 0 <= min_resolved_fraction <= 1:
        raise ValueError("selection constraints must lie in [0,1]")
    if resolution_field not in ('resolved_correctly', 'condition_resolution_fraction'):
        raise ValueError('unknown within-base resolution estimator')
    if not candidates:
        raise ValueError("at least one policy candidate required")
    grouped, seen = defaultdict(list), set()
    for row in rows:
        if row.get("split") != "validation":
            raise ValueError("policy thresholds may be selected on validation only")
        pid = row["policy_id"]
        if pid not in candidates:
            raise ValueError("unknown policy candidate")
        if any(not isinstance(row.get(k), str) or not row[k] for k in ("group_id", "episode_id")):
            raise ValueError("base episode and scene group required for policy selection")
        key = (pid, row["group_id"], row["episode_id"])
        if key in seen:
            raise ValueError("duplicate policy episode would inflate selection weight")
        seen.add(key)
        if any(type(row[k]) is not bool for k in ("resolved_correctly", "wrong_submission")):
            raise ValueError("episode outcome flags must be boolean")
        if resolution_field == 'condition_resolution_fraction':
            value = row.get(resolution_field)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('within-base condition resolution fraction must lie in [0,1]')
        if not math.isfinite(row["interaction_cost"]) or row["interaction_cost"] < 0:
            raise ValueError("all interaction costs must be finite and nonnegative")
        grouped[pid].append(row)
    expected = None
    scores = {}
    for pid, config in candidates.items():
        _validate_config(config)
        if config.get("variant") not in VARIANTS:
            raise ValueError("each candidate must declare its variant")
        items = grouped[pid]
        episodes = {(r["group_id"], r["episode_id"]) for r in items}
        if not episodes or (expected is not None and episodes != expected):
            raise ValueError("all policy candidates need the same complete validation episodes")
        expected = episodes
        groups = defaultdict(list)
        for row in items:
            groups[row["group_id"]].append(row)
        def macro(key):
            return sum(sum(float(r[key]) for r in group) / len(group) for group in groups.values()) / len(groups)
        scores[pid] = {"resolved_fraction": macro(resolution_field),
                       "all_conditions_resolved_fraction": macro("resolved_correctly"), "wrong_episode_rate": macro("wrong_submission"),
                       "interaction_cost": macro("interaction_cost"), "episodes": len(items), "groups": len(groups),
                       "variant": config["variant"], "config_sha256": policy_hash(config)}
    feasible = [pid for pid, s in scores.items() if s["wrong_episode_rate"] <= max_wrong_episode_rate and s["resolved_fraction"] >= min_resolved_fraction]
    ranking = lambda pid: (-scores[pid]["resolved_fraction"], scores[pid]["wrong_episode_rate"], scores[pid]["interaction_cost"],
                           VARIANTS.index(scores[pid]["variant"]), pid)
    chosen = min(feasible, key=ranking) if feasible else None
    fixed = [pid for pid in feasible if scores[pid]["variant"] != "auxiliary"]
    retained_fixed = False
    if chosen and scores[chosen]["variant"] == "auxiliary" and fixed:
        baseline = min(fixed, key=ranking)
        if scores[chosen]["resolved_fraction"] <= scores[baseline]["resolved_fraction"]:
            chosen, retained_fixed = baseline, True
    return {"split": "validation", "selected_policy_id": chosen, "scores": scores,
            "resolution_field": resolution_field,
            "retained_fixed_rule": retained_fixed, "risk_basis": "scene-macro episode point estimate; no statistical guarantee",
            "all_wait_counts_as_unresolved": True, "requires_rebinding_and_calibration_audit": True}
