"""CPU connector calibration and simulated interaction policy development.

Prerequisite: independently recorded complete choice-group and alias-pair VLM
binding gates from the selected Full seed0 repair checkpoint. This script never
loads a VLM, advances LIBERO dynamics, accesses formal test, or dispatches VLA
actions. Selected-policy online audit uses a separate strict audit entry point.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
VERSION = "V4-09-connector-calibration-policy/0.3"
SCENARIOS = ("stable", "goal_change", "channel_change", "both_change")
SPLITS = ("calibration_fit", "calibration_audit", "validation")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            result.update(chunk)
    return result.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def lines(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def binding_gate(job, training):
    if training.get("formal_test_accessed") is not False or training.get("calibration_accessed") is not False:
        raise ValueError("binding repair training must have kept calibration/test sealed")
    config = training["config"]
    limits = config["development_binding_gate"]
    for key in ("choice_group_all_correct_fraction", "joint_alias_pair_both_correct_fraction"):
        if not .9 <= limits[key] <= 1:
            raise ValueError("binding prerequisite thresholds must be at least .9")
    runs = [r for r in training.get("runs", []) if r.get("method") == "Full" and r.get("seed") == 0]
    if len(runs) != 1:
        raise ValueError("exactly one Full seed0 run required")
    run = runs[0]
    data = job / "training_data"
    manifest = load(data / "manifest.json")
    hashes = {r["path"]: r["sha256"] for r in manifest["files"]}
    names = ("choice_groups.jsonl", "joint_alias_pairs.jsonl", "supervision.jsonl")
    for name in names:
        relative = "validation/" + name
        if relative not in hashes or sha(data / relative) != hashes[relative]:
            raise ValueError("binding validation source file hash mismatch: " + relative)
    groups = lines(data / "validation/choice_groups.jsonl")
    pairs = lines(data / "validation/joint_alias_pairs.jsonl")
    if len({g["group_id"] for g in groups}) != len(groups) or len({p["pair_id"] for p in pairs}) != len(pairs):
        raise ValueError("duplicate binding choice-group or alias-pair records")
    if any(len(set(g["sample_ids"])) != len(g["sample_ids"]) for g in groups):
        raise ValueError("duplicate sample inside a complete choice group")
    supervision = {r["sample_id"]: r for r in lines(data / "validation/supervision.jsonl")}
    traces = run.get("selected_validation", {}).get("traces", [])
    by_id = {r["sample_id"]: r for r in traces}
    if len(by_id) != len(traces):
        raise ValueError("duplicate selected validation traces")
    def correct(sample):
        trace, target = by_id.get(sample), supervision.get(sample)
        if trace is None or target is None:
            return False
        expected = target["language"]["target"]
        return (trace.get("target") == expected and trace.get("generated") == expected
                and trace.get("EOS") is True and target["language"]["mask"] is True)
    choice = [{"group_id": g["group_id"], "samples": g["sample_ids"],
               "complete": bool(g.get("all_candidate_targets_covered")) and len(g["sample_ids"]) >= 2,
               "all_correct": bool(g.get("all_candidate_targets_covered")) and len(g["sample_ids"]) >= 2
                              and all(correct(sample) for sample in g["sample_ids"])} for g in groups]
    alias = [{"pair_id": p["pair_id"], "both_correct": correct(p["first"]) and correct(p["second"])} for p in pairs]
    choice_fraction = sum(g["all_correct"] for g in choice) / len(choice) if choice else 0.
    alias_fraction = sum(p["both_correct"] for p in alias) / len(alias) if alias else 0.
    passed = (run.get("status") == "DEVELOPMENT_TRAINING_COMPLETE" and bool(choice) and bool(alias)
              and all(g["complete"] for g in choice)
              and choice_fraction >= limits["choice_group_all_correct_fraction"]
              and alias_fraction >= limits["joint_alias_pair_both_correct_fraction"])
    return run, {"passed": passed, "choice_group_all_correct_fraction": choice_fraction,
                 "joint_alias_pair_both_correct_fraction": alias_fraction, "thresholds": limits,
                 "choice_groups": choice, "alias_pairs": alias, "trace_count": len(traces),
                 "source_manifest_sha256": sha(data / "manifest.json"),
                 "scope": "independent repair-job selected VLM validation traces; not re-generated here"}


class ConnectorPredictor:
    def __init__(self, model):
        self.model, self.cache = model, {}
        self.forward_calls = self.cache_hits = 0

    def __call__(self, public):
        import torch
        from bci_vla.interaction_contract_v4.validation import canonical_digest
        from bci_vla.neural_interaction_v4.tensorizer import collate_public
        key = canonical_digest(public)
        if key in self.cache:
            self.cache_hits += 1
            return deepcopy(self.cache[key])
        inputs, metadata = collate_public([public])
        with torch.inference_mode():
            output = self.model(inputs)
        indices = inputs["candidate_mask"][0].nonzero().reshape(-1).tolist()
        ids = [metadata[0]["instance_ids"][i] for i in indices]
        # The outside class is the LAST (17th) output, never index len(ids).
        logits = [float(output["candidate_logits"][0, i]) for i in indices]
        logits.append(float(output["candidate_logits"][0, -1]))
        supported = bool(output["reliability_valid"][0])
        actions = ("PROPOSE_TASK", "ASK_INTENT", "CHECK_CHANNEL", "WAIT")
        result = {"candidate_ids": ids, "intent_logits": logits,
                  "channel_supported": supported,
                  "channel_logit": float(output["reliability_logit"][0]) if supported else None,
                  "auxiliary_action": actions[int(output["interaction_logits"][0].argmax())]}
        self.cache[key] = result; self.forward_calls += 1
        return deepcopy(result)


class CalibratedPolicy:
    """Only public observations enter this wrapper; no world/scorer attribute."""
    def __init__(self, predictor, config, binding, intent, channel, name):
        self.predictor, self.config, self.binding = predictor, config, binding
        self.intent, self.channel, self.name = intent, channel, name
        self.counters = {"questions": 0, "probes": 0, "intent_queries": 0, "retries": 0}
        self.seen_events, self.decisions = set(), []

    def decide(self, public):
        from bci_vla.calibration_v4 import calibrate_prediction, decide_interaction
        from bci_vla.interaction_contract_v4.validation import canonical_digest
        from bci_vla.neural_interaction_v4.control import route_control
        route = route_control(public)
        if not route["call_neural_module"] or public["public_state"]["pending_question"] is not None:
            return decide_interaction(public, None, self.config, binding=self.binding, state=self.counters)
        for event in public["history"]:
            if event["event_id"] not in self.seen_events:
                self.seen_events.add(event["event_id"])
                if event["role"] in ("NO_RESPONSE", "REJECT"):
                    self.counters["retries"] += 1
        raw = self.predictor(public)
        intent = calibrate_prediction(raw["intent_logits"], self.intent, task="intent", binding=self.binding)
        channel = calibrate_prediction(raw["channel_logit"], self.channel, task="channel", binding=self.binding,
                                       supported=raw["channel_supported"])
        prediction = {"candidate_ids": raw["candidate_ids"], "probabilities": intent["probabilities"],
                      "intent_calibrated": intent["calibrated"], "channel_probability": channel["confidence"],
                      "channel_supported": raw["channel_supported"], "channel_calibrated": channel["calibrated"],
                      "input_sha256": canonical_digest(public), "context": public["context"], "as_of_s": public["as_of_s"],
                      "binding": self.binding, "auxiliary_action": raw["auxiliary_action"]}
        decision = decide_interaction(public, prediction, self.config, binding=self.binding, state=self.counters)
        action = decision["action"]
        if action != "WAIT":
            self.counters["questions"] += 1
            if action == "PROBE": self.counters["probes"] += 1
            if action == "INTENT_QUERY": self.counters["intent_queries"] += 1
        self.decisions.append({"input_sha256": prediction["input_sha256"], "decision": decision,
                               "counters_after_decision": dict(self.counters)})
        return decision


def episode_summary(result, world, group_id, scenario):
    from bci_vla.data_v4.channel import world_at
    interactions = [j for j in result["journal"] if j["kind"] == "INTERACTION"]
    roles = Counter(j["decision"]["action"] for j in interactions)
    waits = sum(j["kind"] == "WAIT" for j in result["journal"])
    no_response = sum(j["event"]["role"] == "NO_RESPONSE" for j in interactions)
    rejects = sum(j["event"]["role"] == "REJECT" for j in interactions)
    submissions = result["submissions"]
    final_goal, _ = world_at(world, result["horizon_s"])
    last = submissions[-1] if submissions else None
    goal_changed = world["initial_goal"] != world["later_goal"] and result["horizon_s"] >= world["change_at_s"]
    resolved = bool(last and last["task"]["source_id"] == final_goal
                    and (not goal_changed or last["submitted_at_s"] >= world["change_at_s"]))
    wrong = sum(not s["source_matches_latent_goal"] for s in result["private_scoring_only"])
    return {"split": result["partition"], "policy_id": result["method"], "group_id": group_id,
            "episode_id": result["base_episode_id"], "scenario": scenario,
            "resolved_correctly": resolved, "wrong_submission": wrong > 0, "wrong_submission_count": wrong,
            "submission_count": len(submissions), "questions": len(interactions), "probes": roles["PROBE"],
            "intent_queries": roles["INTENT_QUERY"], "task_queries": roles["TASK"], "confirmations": roles["CONFIRM"],
            "waits": waits, "no_responses": no_response, "rejects": rejects,
            "interaction_cost": 2. * (len(interactions) + waits), "cost_units": "simulated seconds; each question or wait slot charged 2s",
            "VLA_called": False, "real_dynamics_advanced": False}


def aggregate_base_summaries(condition_summaries, split):
    grouped = defaultdict(list)
    for summary in condition_summaries:
        if summary["split"] != split:
            raise ValueError("scenario summary partition mismatch")
        grouped[(summary["group_id"], summary["episode_id"])].append(summary)
    result = []
    for (group, episode), items in sorted(grouped.items()):
        if {r["scenario"] for r in items} != set(SCENARIOS) or len(items) != len(SCENARIOS):
            raise ValueError("incomplete four-condition base episode")
        if len({r["policy_id"] for r in items}) != 1:
            raise ValueError("one frozen policy required per base episode")
        result.append({"split": split, "policy_id": items[0]["policy_id"], "group_id": group, "episode_id": episode,
                       "resolved_correctly": all(r["resolved_correctly"] for r in items),
                       "wrong_submission": any(r["wrong_submission"] for r in items),
                       "interaction_cost": sum(r["interaction_cost"] for r in items),
                       "condition_resolution_fraction": sum(r["resolved_correctly"] for r in items) / len(items)})
    return result


def online_audit_rows(trajectories, predictor, binding):
    """Loss-only outcomes on selected-policy histories; never alter the policy.

    trajectories contains (result, private_environment_world, scene_group).
    Future success is a standardized +2s PROBE of an already-covered symbol,
    matching the offline calibration target. It is NOT a next-question oracle.
    Identical public observations across latent scenarios share one soft target.
    """
    from bci_vla.data_v4.channel import answer, world_at
    from bci_vla.evaluation_v4.runner import make_question
    from bci_vla.interaction_contract_v4.validation import canonical_digest
    from bci_vla.neural_interaction_v4.control import route_control
    pooled = {}
    for result, world, group in trajectories:
        if result["partition"] != "calibration_audit":
            raise ValueError("online probability audit permits calibration_audit only")
        for journal in result["journal"]:
            public = journal.get("input")
            if public is None or not route_control(public)["call_neural_module"]:
                continue
            # The predictor still receives exactly public, not world or outcome.
            key = canonical_digest(public)
            pool_key = (result["base_episode_id"], key)
            raw = pooled[pool_key]["raw"] if pool_key in pooled else predictor(deepcopy(public))
            ids = raw["candidate_ids"]
            entry = pooled.setdefault(pool_key, {
                "public": public, "raw": raw, "group": group, "episode": result["base_episode_id"],
                "intent_counts": [0.] * (len(ids) + 1), "observations": 0, "channel_successes": 0., "channel_observations": 0})
            if entry["raw"]["candidate_ids"] != ids:
                raise ValueError("same online audit observation has inconsistent candidate axis")
            goal, _ = world_at(world, public["as_of_s"])
            target = ids.index(goal) if goal in ids else len(ids)
            entry["intent_counts"][target] += 1.
            entry["observations"] += 1
            diagnostic = public["public_state"]["diagnostics"]
            target_time = public["as_of_s"] + 2.
            # Current support is insufficient if it expires before the specified
            # future outcome; no invented/expired channel target enters loss.
            future_supported = (raw["channel_supported"] and diagnostic["status"] == "SUFFICIENT"
                and diagnostic["covered_symbols"] and diagnostic["latest_ended_at_s"] is not None
                and target_time <= diagnostic["latest_ended_at_s"] + diagnostic["ttl_s"]
                and target_time <= public["as_of_s"] + diagnostic["window_s"])
            if future_supported:
                question = make_question(public, {"action": "PROBE"}, 100000)
                question.update(issued_at_s=target_time, expires_at_s=target_time + 1.5)
                symbol = diagnostic["covered_symbols"][0]
                known = next(o["option_id"] for o in question["options"] if o["symbol"] == symbol)
                question["diagnostic_option_id"] = known
                observed = answer(world, question, target_time)
                entry["channel_successes"] += int(observed["selected_option_id"] == known)
                entry["channel_observations"] += 1
    intent_rows, channel_rows = [], []
    for (_, key), entry in sorted(pooled.items()):
        raw = entry["raw"]
        common = {"split": "calibration_audit", "group_id": entry["group"], "episode_id": entry["episode"],
                  "metadata": {"binding": binding, "visible_input_sha256": key}, "target_mode": "PROBABILITY",
                  "label_scope": "online_simulated_outcome_distribution_not_identified_POINT"}
        intent_rows.append({**deepcopy(common), "logits": raw["intent_logits"], "mask": True, "supported": True,
                            "target": [v / entry["observations"] for v in entry["intent_counts"]]})
        supported = bool(entry["channel_observations"] and raw["channel_supported"])
        channel_rows.append({**deepcopy(common), "logits": raw["channel_logit"] if supported else None,
                             "target": entry["channel_successes"] / entry["channel_observations"] if supported else None,
                             "mask": supported, "supported": supported})
    return {"intent_rows": intent_rows, "channel_rows": channel_rows,
            "future_probe_definition": "private +2s already-covered symbol PROBE; never shown to policy",
            "policy_history_source": "selected policy online histories, not fixed historical schedule"}


def online_audit_acceptance(base_summaries, probability_audit, criteria):
    """Frozen development checks only, with no threshold/temperature retuning."""
    groups = defaultdict(list)
    for row in base_summaries:
        if row["split"] != "calibration_audit":
            raise ValueError("online acceptance requires calibration_audit episodes")
        groups[row["group_id"]].append(row)
    def macro(key):
        return sum(sum(float(r[key]) for r in values) / len(values) for values in groups.values()) / len(groups) if groups else None
    resolved = macro(criteria.get("resolution_field", "resolved_correctly"))
    wrong = macro("wrong_submission")
    checks = {"episodes_present": bool(base_summaries),
              "positive_resolution": resolved is not None and resolved > 0,
              "minimum_resolution": resolved is not None and resolved >= criteria["min_resolved_fraction"],
              "wrong_episode_constraint": wrong is not None and wrong <= criteria["max_wrong_episode_rate"]}
    diagnostics = {}
    for task in ("intent", "channel"):
        audit = probability_audit[task]
        checks[task + "_support"] = (audit["status"] == "AUDITED" and audit["samples"] >= criteria["min_probability_samples"]
                                     and audit["groups"] >= criteria["min_probability_groups"])
        diagnostics[task + "_nll_not_worse"] = bool(audit.get("after") and audit.get("before")
            and audit["after"]["nll"] <= audit["before"]["nll"] + criteria["nll_tolerance"])
    return {"passed": all(checks.values()), "checks": checks, "diagnostics": diagnostics, "criteria": criteria,
            "resolved_fraction": resolved, "wrong_episode_rate": wrong, "base_episodes": len(base_summaries),
            "scene_groups": len(groups), "risk_basis": "development point estimate only; no statistical safety guarantee",
            "thresholds_or_temperatures_refitted": False}


def candidates(template):
    result = {}
    def add(name, variant, intent, channel, entropy=.5):
        config = deepcopy(template)
        for field in ("scope", "formal_test_access", "calibration_run", "policy_validation_run"):
            config.pop(field, None)
        config.update(status="FROZEN_DEVELOPMENT_POLICY", variant=variant)
        config["policy_selection"]["require_positive_resolution"] = True
        config["policy_selection"]["resolution_field"] = "condition_resolution_fraction"
        config["interaction_protocol"] = {"version": "public-pick-object-confirmation/0.1", "declared_action": "pick",
                                           "scope": "object selection and exact confirmation only; no inferred action or missing-action claim"}
        config["implementation_binding"] = {relative: sha(ROOT / relative) for relative in (
            "bci_vla/calibration_v4/policy.py", "bci_vla/calibration_v4/calibration.py",
            "bci_vla/calibration_v4/evidence_gate.py",
            "bci_vla/evaluation_v4/runner.py", "bci_vla/data_v4/channel.py",
            "execution/v4/step09_calibrate_and_policy.py")}
        config["thresholds"].update(intent_confidence=intent, channel_success=channel, normalized_entropy=entropy)
        result[name] = config
    for intent in (.4, .6, .8):
        for channel in (.4, .6):
            label = str(int(round(intent * (100 if intent > .8 else 10))))
            add(f"fixed_i{label}_c{int(channel*10)}", "fixed_threshold", intent, channel)
    for entropy in (.4, .6):
        for channel in (.4, .6):
            add(f"entropy_e{int(entropy*10)}_c{int(channel*10)}", "entropy_threshold", .6, channel, entropy)
    add("auxiliary_i6_c4", "auxiliary", .6, .4)
    # Compare the evidence contract with the original gate under EXACTLY the
    # same question/probe budget and freshness; never tune from audit outcomes.
    expanded = {}
    for name, config in result.items():
        for suffix in ('plain', 'recheck'):
            candidate = deepcopy(config)
            candidate['selection_freshness_s'] = 8.
            candidate["budgets"] = {"questions":12,"probes":6,"intent_queries":4,"retries":4}
            candidate['confirmation_gate'] = ({'mode':'none'} if suffix=='plain' else
                {'mode':'consistent_recheck','consistent_selections':2,'correct_probes':3,'selection_max_age_s':12.})
            expanded[name + "_" + suffix] = candidate
    return expanded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding-job", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v4/calibration_and_policy.json")
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Append-only calibration job; choose a new output directory")
    args.output.mkdir(parents=True)
    start = time.monotonic()
    report = {"version": VERSION, "status": "STARTED", "step09_complete": False,
              "audit_role": "previously inspected development regression; not a fresh independent test",
              "remediation": "public selection consistency and post-selection known-target certificate; no hidden change-time trigger",
              "formal_test_accessed": False, "calibration_accessed": False, "VLA_called": False,
              "VLM_loaded": False, "real_dynamics_advanced": False, "physical_actions": False,
              "scope": "CPU learned connector outcome calibration and simulated interaction policy development; separate VLM binding gate evidence",
              "interaction_scope": "publicly declared fixed pick action over all visible candidates; object selection and exact confirmation only; no main-task/missing-action inference claim"}
    def event(name, **values):
        row = {"event": name, "elapsed_s": time.monotonic() - start, **values}
        with (args.output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print("CALIBRATION_STAGE=" + json.dumps(row), flush=True)
    exit_code = 1
    try:
        from bci_vla.calibration_v4 import (build_calibration_rows, fit_temperature, evaluate_calibration,
                                           policy_hash, select_policy)
        from bci_vla.calibration_v4.runtime import DEFAULT_VARIANTS, ROW_GENERATOR_VERSION
        from bci_vla.interaction_contract_v4.validation import canonical_digest
        training_path = args.binding_job / "results/training_result.json"
        training = load(training_path)
        run, gate = binding_gate(args.binding_job, training)
        report.update(binding_gate=gate, binding_training_report_sha256=sha(training_path))
        dump(args.output / "binding_gate.json", gate)
        event("binding_gate", passed=gate["passed"], choice=gate["choice_group_all_correct_fraction"], alias=gate["joint_alias_pair_both_correct_fraction"])
        if not gate["passed"]:
            report["status"] = "BLOCKED_BINDING_GATE"
        else:
            import torch
            from safetensors.torch import load_file
            from bci_vla.neural_interaction_v4.model import matched_models
            from bci_vla.evaluation_v4.runner import run_episode, run_calibration_audit_episode
            from execution.v4.build_step04_data import generate
            torch.set_num_threads(4)
            torch.manual_seed(0)
            chosen_dir = Path(run["selected_checkpoint"]).resolve()
            expected_dir = (args.binding_job / "results/Full_seed0" / f"step{run['selected_step']}").resolve()
            if chosen_dir != expected_dir:
                raise ValueError("selected checkpoint is outside this Full seed0 repair job")
            checkpoint = next(c for c in run["checkpoints"] if Path(c["path"]).resolve() == chosen_dir and c["step"] == run["selected_step"])
            file_hashes = {}
            for name in ("connector.safetensors", "language_lora.safetensors"):
                file_hashes[name] = sha(chosen_dir / name)
                if file_hashes[name] != checkpoint["files"][name]:
                    raise ValueError("selected repair checkpoint SHA mismatch: " + name)
            model_binding = {"checkpoint_files": file_hashes, "binding": training["config"]["binding"],
                             "binding_config_version": training["config"]["version"], "base_sha256": training["config"]["base_sha256"],
                             "connector_code_sha256": sha(ROOT / "bci_vla/neural_interaction_v4/model.py"),
                             "tensorizer_code_sha256": sha(ROOT / "bci_vla/neural_interaction_v4/tensorizer.py")}
            model_sha = canonical_digest(model_binding)
            capture_manifest_path = args.capture_root / "captures/capture_manifest.json"
            manifest = load(capture_manifest_path)
            capture_sets = {split: [c for c in manifest["episodes"] if c["partition"] == split] for split in SPLITS}
            if any(not caps for caps in capture_sets.values()):
                raise ValueError("fit/audit/validation actual captures all required")
            groups = {split: {c["group_id"] for c in caps} for split, caps in capture_sets.items()}
            if any(groups[a] & groups[b] for i, a in enumerate(SPLITS) for b in SPLITS[i+1:]):
                raise ValueError("calibration-fit/audit/validation scene-group leakage")
            frame_count = 0
            root = args.capture_root.resolve()
            for caps in capture_sets.values():
                for capture in caps:
                    for frame in capture["frames"]:
                        path = (root / frame["image_path"]).resolve()
                        if root not in path.parents or sha(path) != frame["image_sha256"]:
                            raise ValueError("allowed capture frame hash/path mismatch")
                        frame_count += 1
            data_binding = {"capture_manifest_sha256": sha(capture_manifest_path), "row_generator_version": ROW_GENERATOR_VERSION,
                            "variants": list(DEFAULT_VARIANTS), "splits": list(SPLITS),
                            "scope": "actual LIBERO captures; fixed historical questions; simulated latent-outcome calibration",
                            "row_generator_code_sha256": sha(ROOT / "bci_vla/calibration_v4/runtime.py"),
                            "dependency_sha256": {relative: sha(ROOT / relative) for relative in (
                                "execution/v4/build_step04_data.py", "bci_vla/data_v4/channel.py",
                                "bci_vla/evaluation_v4/runner.py", "bci_vla/neural_interaction_v4/tensorizer.py",
                                "bci_vla/training_v4/text_baseline.py")}}
            data_sha = canonical_digest(data_binding)
            schema_sha = sha(ROOT / "configs/v4/interfaces_data_4_1.json")
            model = matched_models()["Full"].eval().requires_grad_(False)
            model.load_state_dict(load_file(str(chosen_dir / "connector.safetensors"), device="cpu"), strict=True)
            predictor = ConnectorPredictor(model)
            report.update(calibration_accessed=True, checkpoint=str(chosen_dir), selected_step=run["selected_step"],
                          model_binding=model_binding, model_sha256=model_sha, data_binding=data_binding,
                          data_sha256=data_sha, verified_allowed_frames=frame_count,
                          source_partition_counts={k: len(v) for k, v in capture_sets.items()},
                          source_scene_groups={k: sorted(v) for k, v in groups.items()})
            configs = candidates(load(args.config))
            shared = next(iter(configs.values()))
            audit_criteria = {"max_wrong_episode_rate": shared["policy_selection"]["max_wrong_episode_rate"],
                              "min_resolved_fraction": shared["policy_selection"]["min_resolved_fraction"],
                              "resolution_field": shared["policy_selection"]["resolution_field"],
                              "require_positive_resolution": True,
                              "min_probability_samples": shared["calibration"]["min_samples"],
                              "min_probability_groups": shared["calibration"]["min_groups"], "nll_tolerance": 1e-8,
                              "probability_diagnostic": "report calibrated-vs-raw online NLL; negative result is not an implementation failure",
                              "selection_resolution_rule": "within-base condition mean, then scene/episode macro; any wrong submission per base; all-four secondary",
                              "no_audit_retuning": True}
            # This ledger is written before calibration/validation/audit outcomes.
            dump(args.output / "online_audit_preregistered_criteria.json", audit_criteria)
            report["online_audit_preregistered_criteria"] = audit_criteria
            summaries, calibrations, artifacts = [], {}, {}
            fixed_history_cache = {}
            for pid, config in configs.items():
                binding = {"model_sha256": model_sha, "data_sha256": data_sha,
                           "input_schema_sha256": schema_sha, "policy_sha256": policy_hash(config)}
                folder = args.output / "artifacts" / pid
                folder.mkdir(parents=True)
                by_split = {}
                for split in ("calibration_fit", "calibration_audit"):
                    if split not in fixed_history_cache:
                        fixed_history_cache[split] = build_calibration_rows(capture_sets[split], predictor, split=split, binding=binding)
                    # The historical question schedule and frozen predictor are
                    # policy-independent. Reuse those predictions, then bind a
                    # fresh loss-only row copy to each candidate's exact hash.
                    by_split[split] = deepcopy(fixed_history_cache[split])
                    by_split[split]["binding"] = deepcopy(binding)
                    for task in ("intent", "channel"):
                        for row in by_split[split][task + "_rows"]:
                            row["metadata"]["binding"] = deepcopy(binding)
                    dump(folder / f"{split}_rows.json", by_split[split])
                parameters = config["calibration"]
                temperatures, audit = {}, {}
                for task in ("intent", "channel"):
                    temperatures[task] = fit_temperature(by_split["calibration_fit"][task + "_rows"], task=task, binding=binding,
                        min_samples=parameters["min_samples"], min_groups=parameters["min_groups"], bounds=parameters["temperature_bounds"])
                    audit[task] = evaluate_calibration(by_split["calibration_audit"][task + "_rows"], temperatures[task], task=task,
                                                       binding=binding, bins=parameters["ece_bins"])
                artifacts[pid] = (config, binding, temperatures)
                calibrations[pid] = {"binding": binding, "temperatures": temperatures, "fixed_history_audit": audit,
                                     "scope": "historical schedule only; online policy distribution audit still required"}
                dump(folder / "policy.json", config); dump(folder / "temperatures.json", temperatures)
                dump(folder / "fixed_history_audit.json", audit)
                event("offline_calibration_complete", policy=pid, intent_temperature=temperatures["intent"]["temperature"],
                      channel_temperature=temperatures["channel"]["temperature"])
                condition_summaries = []
                for capture in capture_sets["validation"]:
                    for scenario in SCENARIOS:
                        _observations_not_replayed, world = generate(deepcopy(capture), scenario)
                        policy = CalibratedPolicy(predictor, config, binding, temperatures["intent"], temperatures["channel"], pid)
                        result = run_episode(capture, world, policy, args.capture_root, horizon=args.horizon, declared_action="pick")
                        summary = episode_summary(result, world, capture["group_id"], scenario)
                        condition_summaries.append(summary)
                        dump(folder / "validation" / f"{capture['base_episode_id']}_{scenario}.json",
                             {"summary": summary, "trace": result, "policy_decisions": policy.decisions})
                # Variants share a base capture, so do not inflate independent
                # selection N by treating four latent schedules as four episodes.
                base_summaries = aggregate_base_summaries(condition_summaries, "validation")
                summaries.extend(base_summaries)
                dump(folder / "validation_summary.json", {"conditions": condition_summaries, "base_episodes": base_summaries,
                      "selection_resolution_rule": "mean resolution over four conditions within each base episode, then scene/episode macro; any error counted per base episode; all-four is secondary",
                      "independent_base_episodes": len(base_summaries), "conditions_not_independent": True})
                event("validation_policy_complete", policy=pid, base_episodes=len(base_summaries),
                      all_conditions_resolved=sum(r["resolved_correctly"] for r in base_summaries))
            selection_limits = next(iter(configs.values()))["policy_selection"]
            selection = select_policy(summaries, candidates=configs,
                                      max_wrong_episode_rate=selection_limits["max_wrong_episode_rate"],
                                      min_resolved_fraction=max(selection_limits["min_resolved_fraction"], sys.float_info.epsilon),
                                      resolution_field=selection_limits["resolution_field"])
            report.update(calibrations=calibrations, validation_selection=selection,
                          predictor_forward_calls=predictor.forward_calls, predictor_cache_hits=predictor.cache_hits,
                          online_policy_audit={"status": "NOT_RUN_NO_SELECTED_POLICY", "split": "calibration_audit"},
                          status="NO_POLICY_MEETS_DEVELOPMENT_CONSTRAINTS")
            selected = selection["selected_policy_id"]
            if selected is None:
                report["status"] = "NO_POLICY_MEETS_DEVELOPMENT_CONSTRAINTS"
            else:
                config, binding, temperatures = artifacts[selected]
                frozen_before = canonical_digest({"config": config, "binding": binding, "temperatures": temperatures})
                audit_conditions, trajectories = [], []
                for capture in capture_sets["calibration_audit"]:
                    for scenario in SCENARIOS:
                        _ignored_fixed_history, world = generate(deepcopy(capture), scenario)
                        policy = CalibratedPolicy(predictor, config, binding, temperatures["intent"], temperatures["channel"], selected)
                        result = run_calibration_audit_episode(capture, world, policy, args.capture_root, horizon=args.horizon, declared_action="pick")
                        summary = episode_summary(result, world, capture["group_id"], scenario)
                        audit_conditions.append(summary)
                        trajectories.append((result, world, capture["group_id"]))
                        dump(args.output / "online_audit" / f"{capture['base_episode_id']}_{scenario}.json",
                             {"summary": summary, "trace": result, "policy_decisions": policy.decisions})
                audit_base = aggregate_base_summaries(audit_conditions, "calibration_audit")
                online_rows = online_audit_rows(trajectories, predictor, binding)
                online_metrics = {task: evaluate_calibration(online_rows[task + "_rows"], temperatures[task], task=task,
                                  binding=binding, bins=config["calibration"]["ece_bins"]) for task in ("intent", "channel")}
                acceptance = online_audit_acceptance(audit_base, online_metrics, audit_criteria)
                control_checks = {}
                audit_capture = capture_sets["calibration_audit"][0]
                _ignored, control_world = generate(deepcopy(audit_capture), "stable")
                for control in ("STOP", "CANCEL"):
                    control_policy = CalibratedPolicy(predictor, config, binding, temperatures["intent"], temperatures["channel"], selected)
                    control_trace = run_calibration_audit_episode(audit_capture, control_world, control_policy,
                        args.capture_root, horizon=args.horizon, control_at=(0., control), declared_action="pick")
                    control_checks[control] = (not control_policy.decisions and not control_trace["submissions"]
                        and len(control_trace["journal"]) == 1 and control_trace["journal"][0]["kind"] == "CONTROL")
                    dump(args.output / "online_audit" / f"control_{control.lower()}.json", control_trace)
                frozen_after = canonical_digest({"config": config, "binding": binding, "temperatures": temperatures})
                acceptance["checks"].update(STOP_bypass=control_checks["STOP"], CANCEL_bypass=control_checks["CANCEL"],
                                              frozen_policy_and_temperatures=frozen_before == frozen_after)
                acceptance["passed"] = all(acceptance["checks"].values())
                implementation_complete = (control_checks["STOP"] and control_checks["CANCEL"] and frozen_before == frozen_after)
                schedule_change = {"detected": True, "fixed_history_schedule_reused_as_online_audit": False,
                    "historical_distribution": "pre-generated TASK/PROBE/CONFIRM schedule across 11 noisy variants",
                    "audited_distribution": "selected calibrated policy generates online questions across four frozen world scenarios, with a public all-candidate pick menu",
                    "calibration_refit_after_shift": False, "policy_thresholds_retuned_on_audit": False}
                online_report = {"status": "PASSED" if acceptance["passed"] else "COMPLETED_WITH_NEGATIVE_FINDINGS", "split": "calibration_audit",
                    "selected_policy_id": selected, "binding": binding, "probability_metrics": online_metrics,
                    "acceptance": acceptance, "domain_shift": schedule_change, "condition_summaries": audit_conditions,
                    "base_episodes": audit_base, "independent_base_episode_count": len(audit_base),
                    "condition_count_not_independent": len(audit_conditions), "control_checks": control_checks,
                    "frozen_parameters_sha256_before": frozen_before, "frozen_parameters_sha256_after": frozen_after,
                    "VLA_called": False, "real_dynamics_advanced": False, "physical_actions": False}
                dump(args.output / "online_audit" / "probability_rows.json", online_rows)
                dump(args.output / "online_audit" / "audit_result.json", online_report)
                report.update(online_policy_audit=online_report, step09_complete=implementation_complete,
                              step09_implementation_complete=implementation_complete,
                              policy_operating_constraints_passed=acceptance["passed"],
                              simulated_interaction_policy_usable=acceptance["passed"],
                              policy_deployment_ready=False, research_conclusion_supported=False,
                              status=("STEP09_IMPLEMENTATION_AND_DEVELOPMENT_CONSTRAINTS_PASSED" if acceptance["passed"]
                                      else "STEP09_IMPLEMENTED_WITH_NEGATIVE_POLICY_FINDINGS") if implementation_complete else "ONLINE_POLICY_MECHANISM_FAILED",
                              predictor_forward_calls=predictor.forward_calls, predictor_cache_hits=predictor.cache_hits)
                dump(args.output / "selected_policy.json", {"policy_id": selected, "config": config, "binding": binding,
                     "temperatures": temperatures, "validation": selection["scores"][selected],
                     "online_policy_audit_passed": acceptance["passed"], "implementation_complete": implementation_complete,
                     "simulated_interaction_policy_usable": acceptance["passed"], "policy_deployment_ready": False,
                     "execution_authorized": False})
                event("online_policy_audit_complete", policy=selected, passed=acceptance["passed"], checks=acceptance["checks"])
            exit_code = 0 if report["step09_complete"] else 1
    except Exception as exc:
        report.update(status="FAILED", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        event("failed", error_type=type(exc).__name__, error=str(exc))
    report["elapsed_s"] = time.monotonic() - start
    dump(args.output / "calibration_result.json", report)
    print("CALIBRATION_RESULT=" + json.dumps({k: report[k] for k in ("status", "step09_complete", "elapsed_s")}), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
