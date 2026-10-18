"""Locally finalize measured Step08 evidence; never upgrades its restricted scope.

Run explicitly after training and selected-weight collection. No network, model
loading, optimizer, test/calibration access, or action dispatch exists here.
Existing finalization evidence is never overwritten. TXT uses UTF-8 BOM + CRLF.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "execution/v4"
METHODS = ("Full", "B3_aux", "B4_aux", "B2")
SEEDS = (0, 1, 2)
EXPECTED = {(m, s) for m in METHODS for s in SEEDS}
FAMILIES = ("noisy_ambiguous", "declared_pick", "missing_action", "missing_source",
            "conflicting_declarations", "no_response", "missing_destination", "next_pick_symbolic")
COMPLETE = "DEVELOPMENT_TRAINING_COMPLETE"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_text(path, value):
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    Path(path).write_bytes(("\ufeff" + normalized.replace("\n", "\r\n")).encode("utf-8"))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def number(value, digits=4):
    return f"{value:.{digits}f}" if finite(value) else "未记录"


def stats(values):
    values = [float(v) for v in values if finite(v)]
    return {"count": len(values), "mean": statistics.mean(values) if values else None,
            "sample_SD": statistics.stdev(values) if len(values) > 1 else None, "values": values}


def validation_summary(run, problems):
    label = f"{run.get('method')}/seed{run.get('seed')}"
    selected = run.get("selected_validation", {})
    traces = selected.get("traces", [])
    if not isinstance(traces, list) or not all(isinstance(t, dict) for t in traces):
        problems.append(label + ": validation traces格式错误")
        traces = []
    families, groups = defaultdict(list), defaultdict(list)
    for trace in traces:
        families[trace.get("family", "UNRECORDED")].append(trace)
        if trace.get("exact") is not (trace.get("generated") == trace.get("target")):
            problems.append(label + ": 原始文本与exact标记不符")
        if isinstance(trace.get("scene_group"), str) and finite(trace.get("language_CE")):
            groups[trace["scene_group"]].append(float(trace["language_CE"]))
    exact = sum(t.get("generated") == t.get("target") for t in traces)
    means = {k: statistics.mean(v) for k, v in sorted(groups.items())}
    scene = {"mean": statistics.mean(means.values()) if means else None,
             "group_means": means, "group_counts": {k: len(v) for k, v in sorted(groups.items())}}
    family = {}
    for key, rows in sorted(families.items()):
        matched = sum(r.get("generated") == r.get("target") for r in rows)
        pointers = [r["pointer_exact"] for r in rows if type(r.get("pointer_exact")) is bool]
        family[key] = {"exact": matched, "count": len(rows), "exact_rate": matched / len(rows),
                       "language_CE": stats(r.get("language_CE") for r in rows)["mean"],
                       "pointer_exact": sum(pointers), "pointer_count": len(pointers)}
    if run.get("status") == COMPLETE:
        if len(traces) != 32 or selected.get("count") != 32 or selected.get("exact") != exact:
            problems.append(label + ": 完整32条validation或exact汇总不符")
        if Counter(t.get("family") for t in traces) != Counter({f: 4 for f in FAMILIES}):
            problems.append(label + ": validation未覆盖8类各4条")
        recorded = selected.get("language_CE", {}).get("mean")
        if not finite(recorded) or not finite(scene["mean"]) or not math.isclose(recorded, scene["mean"], rel_tol=1e-6, abs_tol=1e-7):
            problems.append(label + ": 场景组等权CE无法从trace复核")
    proposal_targets = Counter(t.get("target") for t in traces if t.get("family") == "declared_pick")
    clarification_count = sum(isinstance(t.get("target"), str) and t["target"].endswith("?") for t in traces)
    return {"exact": exact, "count": len(traces), "exact_rate": exact / len(traces) if traces else None,
            "family": family, "scene_group_language_CE": scene,
            "initial_scene_group_language_CE": run.get("initial_validation", {}).get("language_CE"),
            "text_tokens": selected.get("text_tokens"),
            "target_composition": {"clarification_rows": clarification_count, "all_rows": len(traces),
                                   "declared_pick_target_counts": dict(proposal_targets)}}


def challenge_summary(path, training_runs):
    """Recompute paired diagnostics from real raw outputs; absent is not zero."""
    if path is None or not path.is_file():
        return {"status": "NOT_AVAILABLE", "path": str(path) if path else None,
                "runs": [], "methods": {}, "validation_errors": [],
                "note": "未提供真实challenge报告，不生成任何性能数值。"}
    report = read(path)
    errors, runs = [], []
    if report.get("status") != "DEVELOPMENT_CHALLENGES_COMPLETE":
        errors.append("challenge原始状态：" + str(report.get("status")))
    for flag in ("formal_test_accessed", "calibration_accessed", "action_sampling"):
        if report.get(flag) is not False:
            errors.append("challenge缺少排除范围确认：" + flag)
    if report.get("optimizer_steps") != 0 or report.get("no_weight_updates") is not True:
        errors.append("challenge不能证明未更新权重")
    seen = set()
    expected_steps = {(r["method"], r["seed"]): r["selected_step"] for r in training_runs}
    for raw in report.get("runs", []):
        key = (raw.get("method"), raw.get("seed"))
        label = f"{key[0]}/seed{key[1]}"
        if key not in EXPECTED or key in seen:
            errors.append(label + ": 非预期或重复challenge run")
        seen.add(key)
        if raw.get("selected_step") != expected_steps.get(key):
            errors.append(label + ": challenge未绑定训练所选step")
        pairs, unique = {}, set()
        for kind in ("paired_evidence", "joint_alias"):
            source = raw.get("pairs", {}).get(kind, {})
            traces = source.get("traces", [])
            families, validated = defaultdict(list), []
            for trace in traces:
                first, second, off = [trace.get(k, {}) for k in ("first", "second", "first_without_interaction")]
                for side_name, side in (("first", first), ("second", second), ("first_without_interaction", off)):
                    if not isinstance(side.get("generated"), str) or not isinstance(side.get("target"), str):
                        errors.append(label + "/" + kind + ": 缺少原始生成/目标 " + side_name)
                    elif side.get("exact") is not (side["generated"] == side["target"]):
                        errors.append(label + "/" + kind + ": 原始exact标记不符 " + side_name)
                a = first.get("generated") == first.get("target") and isinstance(first.get("target"), str)
                b = second.get("generated") == second.get("target") and isinstance(second.get("target"), str)
                without = off.get("generated") == first.get("target") and isinstance(first.get("target"), str)
                changed = first.get("generated") != second.get("generated")
                expected_changed = first.get("target") != second.get("target")
                if off.get("target") != first.get("target") or trace.get("both_correct") is not (a and b) or trace.get("output_changed") is not changed or not expected_changed:
                    errors.append(label + "/" + kind + ": pair控制/汇总绑定不符")
                for side in (first, second):
                    if isinstance(side.get("sample_id"), str):
                        unique.add(side["sample_id"])
                entry = {"family": trace.get("family"), "first_exact": a, "second_exact": b,
                         "both_correct": a and b, "output_changed": changed,
                         "first_without_interaction_exact": without,
                         "first_output_changes_when_removing_interaction": first.get("generated") != off.get("generated")}
                validated.append(entry)
                families[entry["family"]].append(entry)
            both = sum(t["both_correct"] for t in validated)
            changed = sum(t["output_changed"] for t in validated)
            if source.get("both_correct") != both or source.get("count") != len(validated) or source.get("output_changed") != changed:
                errors.append(label + "/" + kind + ": 报告计数无法由trace复核")
            pairs[kind] = {"both_correct": both, "count": len(validated),
                           "both_correct_rate": both / len(validated) if validated else None,
                           "output_changed": changed,
                           "first_without_interaction_exact": sum(t["first_without_interaction_exact"] for t in validated),
                           "first_output_changes_when_removing_interaction": sum(t["first_output_changes_when_removing_interaction"] for t in validated),
                           "family": {f: {"both_correct": sum(t["both_correct"] for t in rows), "count": len(rows),
                                          "output_changed": sum(t["output_changed"] for t in rows),
                                          "first_without_interaction_exact": sum(t["first_without_interaction_exact"] for t in rows)}
                                      for f, rows in sorted(families.items())},
                           "raw_traces": traces}
        aux = deepcopy(raw.get("auxiliary", {}))
        for name, metric in aux.items():
            if name == "train_frequency_channel_Brier_reference":
                if not finite(metric) or not 0 <= metric <= 1:
                    errors.append(label + ": 通道常量参照无效")
                continue
            if name == "confusion_at_uncalibrated_0_5":
                for head, cells in metric.items():
                    if set(cells) != {"TP", "TN", "FP", "FN"} or any(type(v) is not int or v < 0 for v in cells.values()):
                        errors.append(label + ": 辅助头混淆矩阵无效 " + head)
                        continue
                    tp, tn, fp, fn = (cells[k] for k in ("TP", "TN", "FP", "FN"))
                    recall = tp / (tp + fn) if tp + fn else None
                    specificity = tn / (tn + fp) if tn + fp else None
                    cells.update(precision=tp/(tp+fp) if tp+fp else None, recall=recall,
                                 F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                                 balanced_accuracy=(recall+specificity)/2 if recall is not None and specificity is not None else None)
                continue
            if not isinstance(metric, dict):
                errors.append(label + ": 未识别的辅助指标格式 " + name)
                continue
            count = metric.get("count")
            if type(count) is not int or count < 0:
                errors.append(label + ": 辅助指标count无效 " + name)
                continue
            if "correct" in metric:
                correct = metric["correct"]
                if type(correct) is not int or not 0 <= correct <= count:
                    errors.append(label + ": 辅助指标correct无效 " + name)
                metric["rate"] = correct / count if count else None
            if name == "channel_Brier" and count and (not finite(metric.get("mean")) or not 0 <= metric["mean"] <= 1):
                errors.append(label + ": 实测Brier无效")
        runs.append({"method": key[0], "seed": key[1], "selected_step": raw.get("selected_step"),
                     "unique_public_examples": len(unique), "pairs": pairs, "auxiliary": aux})
    if seen != EXPECTED:
        errors.append("challenge未实际覆盖全部12项训练run")
    counts = report.get("counts", {})
    for run in runs:
        if run["unique_public_examples"] != counts.get("unique_public_examples"):
            errors.append(f"{run['method']}/seed{run['seed']}: unique public计数与manifest不符")
        for kind in ("paired_evidence", "joint_alias"):
            if run["pairs"][kind]["count"] != counts.get(kind + "_pairs"):
                errors.append(f"{run['method']}/seed{run['seed']}: {kind} pairs计数不符")
    methods = {}
    for method in METHODS:
        selected = [r for r in runs if r["method"] == method]
        methods[method] = {
            kind: {"both_correct_rate_across_seeds": stats(r["pairs"][kind]["both_correct_rate"] for r in selected),
                   "family_both_correct_rate_across_seeds": {
                       family: stats(p["both_correct"] / p["count"] if p.get("count") else None
                                     for r in selected for p in [r["pairs"][kind]["family"].get(family, {})])
                       for family in ("declared_pick", "next_pick_symbolic")}}
            for kind in ("paired_evidence", "joint_alias")}
    return {"status": "VERIFIED_DEVELOPMENT_DIAGNOSTICS" if not errors else "DIAGNOSTIC_REPORT_WITH_GAPS",
            "source_status": report.get("status"), "path": str(path), "sha256": digest(path),
            "counts": counts, "elapsed_s": report.get("elapsed_s"), "error": report.get("error"),
            "runs": runs, "methods": methods, "validation_errors": list(dict.fromkeys(errors)),
            "scope": "post-training validation diagnostic; no weight update, no independent external test",
            "interpretation": "两侧同时正确才计绑定成功；仅output_changed不足以说明答案正确。三seed重复同一组public，不增加独立场景/回合样本。",
            "auxiliary_scope": "pointer与missing-slot/abstain是辅助头开发统计；0.5阈值未校准，少量模拟channel监督的Brier不证明真实BCI可靠性。"}


def verify_artifacts(run, manifest, manifest_path, problems):
    """Rehash selected safetensors against the actual training report, not itself."""
    method, seed, step = run.get("method"), run.get("seed"), run.get("selected_step")
    label = f"{method}/seed{seed}"
    selected = run.get("selected_checkpoint")
    matches = [c for c in run.get("checkpoints", []) if c.get("step") == step and c.get("path") == selected]
    if len(matches) != 1 or not isinstance(selected, str):
        problems.append(label + ": selected checkpoint缺少唯一hash记录")
        return []
    required = ["language_lora.safetensors"] + (["connector.safetensors"] if method != "B2" else [])
    result = []
    for name in required:
        expected = matches[0].get("files", {}).get(name)
        remote = str(PurePosixPath(selected) / name)
        entries = [e for e in manifest.get("files", []) if e.get("method") == method and e.get("seed") == seed
                   and e.get("step") == step and e.get("remote_path") == remote]
        detail = {"method": method, "seed": seed, "step": step, "name": name,
                  "expected_sha256_from_training_report": expected, "verified": False}
        if len(entries) != 1 or not isinstance(expected, str) or len(expected) != 64:
            problems.append(label + ": 缺少或重复artifact记录 " + name)
            result.append(detail)
            continue
        entry = entries[0]
        raw = entry.get("local_path", entry.get("path"))
        if not isinstance(raw, str) or not raw:
            problems.append(label + ": artifact本地路径缺失 " + name)
            result.append(detail)
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path if (ROOT / path).exists() else manifest_path.parent / path
        path = path.resolve()
        detail["local_path"] = str(path)
        if entry.get("sha256_expected", entry.get("sha256")) != expected:
            problems.append(label + ": artifact与report预期hash不符 " + name)
        elif not path.is_file():
            problems.append(label + ": 本地checkpoint不存在 " + name)
        else:
            actual = digest(path)
            detail.update(actual_sha256=actual, bytes=path.stat().st_size)
            if actual != expected or entry.get("sha256_actual", actual) != actual:
                problems.append(label + ": 本地SHA256复核失败 " + name)
            else:
                detail["verified"] = True
        result.append(detail)
    return result


def attempt_cost(attempt, path):
    report = read(path)
    costs = []
    for run in report.get("runs", []):
        smoke = run.get("single_batch_smoke", {})
        details = smoke.get("details", [])
        costs.append({"method": run.get("method"), "seed": run.get("seed"), "status": run.get("status"),
                      "error": run.get("error"), "training_optimizer_steps": run.get("optimizer_steps", 0),
                      "recorded_smoke_iterations": len(smoke.get("losses", [])),
                      "individually_verified_smoke_updates": sum(d.get("stepped") is True for d in details) if details else None,
                      "reported_elapsed_s": run.get("elapsed_s"), "peak_GPU_allocated_bytes": run.get("peak_GPU_allocated_bytes")})
    return {"attempt": attempt, "report": str(path.relative_to(ROOT)), "report_sha256": digest(path),
            "status": report.get("status"), "error": report.get("error"), "reported_total_wall_s": report.get("elapsed_s"),
            "training_optimizer_steps": sum(c["training_optimizer_steps"] for c in costs),
            "recorded_smoke_iterations": sum(c["recorded_smoke_iterations"] for c in costs), "runs": costs}


def build_audit(attempt, artifacts_path, challenges_path=None):
    path = OUT / f"08_real_attempt{attempt}.json"
    report = read(path)
    problems = []
    artifacts_manifest = read(artifacts_path) if artifacts_path.is_file() else {}
    if not artifacts_manifest:
        problems.append("本地checkpoint manifest尚不存在")
    if artifacts_manifest.get("attempt") != attempt or artifacts_manifest.get("source_report_sha256") != digest(path):
        problems.append("checkpoint manifest未绑定本轮真实report SHA256")
    if artifacts_manifest.get("errors"):
        problems.append("checkpoint收集存在errors，详见原manifest")
    config = report.get("config", {})
    if set(config.get("methods", [])) != set(METHODS) or set(config.get("seeds", [])) != set(SEEDS):
        problems.append("实际配置不完整覆盖四方法×三个种子")
    if config.get("optimizer_steps") != 200 or config.get("smoke_steps") != 20:
        problems.append("实际配置不是统一200训练步＋20烟测步")
    if report.get("status") != "DEVELOPMENT_PILOT_COMPLETE":
        problems.append("训练report状态：" + str(report.get("status")))
    if report.get("error"):
        problems.append("顶层错误：" + str(report["error"]))
    for flag in ("formal_test_accessed", "calibration_accessed", "action_sampling", "physical_actions"):
        if report.get(flag) is not False:
            problems.append("缺少排除范围确认：" + flag)
    for flag in ("frozen_base_unchanged", "action_expert_unchanged"):
        if report.get(flag) is not True:
            problems.append("缺少冻结参数复核：" + flag)
    by_key = defaultdict(list)
    for run in report.get("runs", []):
        by_key[(run.get("method"), run.get("seed"))].append(run)
    if set(by_key) != EXPECTED or any(len(v) != 1 for v in by_key.values()):
        problems.append("真实run集合缺失、重复或多于12项")
    runs, artifacts = [], []
    for method, seed in sorted(EXPECTED):
        candidates = by_key.get((method, seed), [])
        if len(candidates) != 1:
            problems.append(f"{method}/seed{seed}: 未提供唯一run")
            continue
        run = candidates[0]
        label = f"{method}/seed{seed}"
        if run.get("status") != COMPLETE:
            problems.append(label + ": " + str(run.get("status")) + "; " + str(run.get("error")))
        if run.get("optimizer_steps") != 200:
            problems.append(label + ": 实际训练步数=" + str(run.get("optimizer_steps")))
        history = run.get("history", [])
        if (len(history) != run.get("optimizer_steps", 0)
                or [h.get("step") for h in history] != list(range(1, len(history) + 1))
                or any(h.get("stepped") is not True or not finite(h.get("loss", {}).get("total")) for h in history)):
            problems.append(label + ": optimizer更新历史不连续或有无效值")
        smoke = run.get("single_batch_smoke", {})
        if (len(smoke.get("losses", [])) != 20 or smoke.get("loss_decreased") is not True
                or len(smoke.get("details", [])) != 20
                or any(d.get("stepped") is not True for d in smoke.get("details", []))):
            problems.append(label + ": 20步单批烟测未完整通过")
        changed = run.get("parameter_audit", {}).get("trainable_changed", [])
        if not any(not name.startswith("connector.") for name in changed) or (method != "B2" and not any(name.startswith("connector.") for name in changed)):
            problems.append(label + ": 缺少真实参数变化证据")
        metrics = validation_summary(run, problems)
        if run.get("status") == COMPLETE:
            artifacts += verify_artifacts(run, artifacts_manifest, artifacts_path, problems)
        runs.append({"method": method, "seed": seed, "status": run.get("status"), "error": run.get("error"),
                     "optimizer_steps": run.get("optimizer_steps", 0), "selected_step": run.get("selected_step"),
                     "selected_checkpoint": run.get("selected_checkpoint"), "schedule_sha256": run.get("schedule_sha256"),
                     "elapsed_s": run.get("elapsed_s"), "peak_GPU_allocated_bytes": run.get("peak_GPU_allocated_bytes"),
                     "trainable_parameters": run.get("trainable_parameters"), "validation": metrics,
                     "token_on_off_development_probe": run.get("token_on_off_development_probe", [])})
    for seed in SEEDS:
        schedules = {r["schedule_sha256"] for r in runs if r["seed"] == seed}
        if len(schedules) != 1 or None in schedules:
            problems.append(f"seed{seed}: 方法间训练schedule不一致或缺失")
    data_path = OUT / f"08_data_manifest{attempt}.json"
    data = read(data_path) if data_path.is_file() else {}
    for split, count in (("train", 128), ("validation", 32)):
        if data.get("counts", {}).get(split, {}).get("public_rows") != count:
            problems.append(f"{split}: 数据manifest缺失或行数不是{count}")
    methods = {}
    for method in METHODS:
        selected = [r for r in runs if r["method"] == method and r["status"] == COMPLETE]
        methods[method] = {"completed_seeds": [r["seed"] for r in selected],
                           "validation_exact_rate": stats(r["validation"]["exact_rate"] for r in selected),
                           "validation_scene_group_CE": stats(r["validation"]["scene_group_language_CE"]["mean"] for r in selected),
                           "family_exact_rate": {f: stats(r["validation"]["family"].get(f, {}).get("exact_rate") for r in selected) for f in FAMILIES}}
    costs = [attempt_cost(n, OUT / f"08_real_attempt{n}.json") for n in range(1, attempt + 1) if (OUT / f"08_real_attempt{n}.json").is_file()]
    problems = list(dict.fromkeys(problems))
    passed = not problems and len(runs) == 12 and len(artifacts) == 21 and all(a["verified"] for a in artifacts)
    challenges = challenge_summary(challenges_path, runs)
    cpu_log = OUT / "08_cpu_tests.log"
    cpu = {"status": "NOT_AVAILABLE", "tests_run": None}
    if cpu_log.is_file():
        raw_cpu = cpu_log.read_text(encoding="utf-8-sig")
        import re
        match = re.search(r"Ran\s+(\d+)\s+tests?\s+in\s+([\d.]+)s", raw_cpu)
        if match and re.search(r"(?m)^OK\s*$", raw_cpu) and not re.search(r"(?m)^FAILED\b", raw_cpu):
            cpu = {"status": "PASS_CPU_MECHANICS_ONLY", "tests_run": int(match[1]),
                   "elapsed_s": float(match[2]), "log": str(cpu_log.relative_to(ROOT)), "sha256": digest(cpu_log),
                   "new_challenge_generator_tests": 5,
                   "scope": "CPU数据/优化器/B2序列化/challenge生成机制；不是GPU语义或机器人任务性能证据。"}
        else:
            cpu = {"status": "LOG_NOT_PARSEABLE", "tests_run": None,
                   "log": str(cpu_log.relative_to(ROOT)), "sha256": digest(cpu_log)}
    return {"step_id": "V4-08", "attempt": attempt,
            "status": "PASS_RESTRICTED_DEVELOPMENT_TRAINING_ONLY" if passed else "PARTIAL_WITH_SPECIFIC_GAPS",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "restricted_development_training_complete": passed, "full_step08_complete": False,
            "source_report": str(path.relative_to(ROOT)), "source_report_sha256": digest(path),
            "artifact_manifest": str(artifacts_path), "artifact_manifest_sha256": digest(artifacts_path) if artifacts_path.is_file() else None,
            "config": config, "data": data, "data_manifest_sha256": digest(data_path) if data_path.is_file() else None,
            "runs": runs, "method_summaries": methods, "selected_local_checkpoint_verification": artifacts,
            "development_challenges": challenges, "local_cpu_tests": cpu,
            "missing_or_failed_requirements": problems, "attempt_costs": costs,
            "total_reported_attempt_wall_s": sum(c["reported_total_wall_s"] for c in costs if finite(c["reported_total_wall_s"])),
            "total_training_optimizer_steps_all_attempts": sum(c["training_optimizer_steps"] for c in costs),
            "total_recorded_smoke_iterations_all_attempts": sum(c["recorded_smoke_iterations"] for c in costs),
            "text_token_preflight": report.get("text_token_preflight"),
            "scope": "actual LIBERO images + simulated interactions; explicit public declaration teaching; development only",
            "general_hidden_intent_robustness_proven": False, "place_phase_ready": False,
            "external_generalization_proven": False, "real_EEG": False, "physical_robot": False,
            "formal_test_unsealed": False, "calibration_used": False, "full_VLA_closed_loop": False,
            "on_prefix_semantic_gate_passed": False, "historical_G2_G3_G4_upgraded": False,
            "selection_warning": "Validation参与checkpoint选优及开发修复；不是独立外部/test泛化结果。",
            "statistical_unit_warning": "32条validation来自8回合/2场景组；3seed重复同一批例子，不是96次独立机器人实验。",
            "artifact_scope": "仅本地选中safetensors重新计算SHA256；optimizer/RNG状态只保留远端report哈希，未下载或加载。",
            "auxiliary_scope": "交互头为辅助监督；未验收已部署的学得澄清策略或校准可靠性。"}


def document(audit):
    date = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
    outcome = "受限开发训练完成；步骤08整体仍未完成。" if audit["restricted_development_training_complete"] else "部分通过；存在真实训练或checkpoint核验缺项。"
    lines = [f"V4-08 交互模块与高层语言开发训练（{date}）", "", "一、结论", outcome,
             "本轮是模拟交互下的明确公开声明教学，不能称脑机隐藏意图推理已验证。",
             "真实放置阶段、真实含噪隐藏意图监督与外部泛化仍缺；旧G2/G3/G4不升级。", "",
             "二、数据与监督范围",
             "实际LIBERO图像＋特权投影框＋6维RGB统计；脑机输入模拟，Florence/SAM未接入。"]
    binding_lines = []
    for run in audit.get("development_challenges", {}).get("runs", []):
        if run.get("method") != "Full":
            continue
        pointer = run.get("auxiliary", {}).get("pointer", {})
        evidence = run.get("pairs", {}).get("paired_evidence", {})
        alias = run.get("pairs", {}).get("joint_alias", {})
        binding_lines.append(f"Full/seed{run['seed']}：声明对象指针{pointer.get('correct')}/{pointer.get('count')}；同场景证据切换双侧同时正确{evidence.get('both_correct')}/{evidence.get('count')}；别名重绑定双侧同时正确{alias.get('both_correct')}/{alias.get('count')}。")
    if binding_lines:
        lines[7:7] = ["关键语义复验：本轮尚未打通选中对象内部token到VLM目标文字的可靠绑定；不能仅用训练损失下降或对象指针通过替代此能力。"] + binding_lines
    for split in ("train", "validation"):
        lines.append(split + "实测清单：" + json.dumps(audit["data"].get("counts", {}).get(split, {}), ensure_ascii=False))
    lines += ["POINT仅表示受限公开声明对象绑定；相同public隐藏反事实仍歧义/澄清，并只计一次权重。",
              "主任务限公开动作菜单；缺动作/来源/目的、声明冲突或未响应时要求澄清/弃权。",
              "next_pick_symbolic是明确empty夹爪的纯文本符号干预，不伪装真实抓取/放置阶段帧。",
              "所有方法共用公开prompt和对象字典；Full/B3/B4经内部token，B2用完整同信息文本。", "",
              "三、实际配置", json.dumps(audit["config"], ensure_ascii=False, indent=2),
              "共同高层LoRA初始化/预算单列；B2没有连接器，参数量和文本token开销不同。",
              "烟测后重置共同初始化；烟测成本另计，不把过拟合权重带入比较。", "",
              "四、按方法与seed记录原始开发结果",
              "方法 | seed | 状态 | 实际步数 | 选中step | validation精确匹配 | 场景等权CE | run秒 | 峰值GPU字节"]
    for run in audit["runs"]:
        val = run["validation"]
        lines.append(f"{run['method']} | {run['seed']} | {run['status']} | {run['optimizer_steps']} | {run['selected_step']} | {val['exact']}/{val['count']} | {number(val['scene_group_language_CE']['mean'])} | {number(run['elapsed_s'],2)} | {run['peak_GPU_allocated_bytes']}")
        lines.append("  各family：" + "; ".join(f"{f}={s['exact']}/{s['count']}, CE={number(s['language_CE'])}, pointer={s['pointer_exact']}/{s['pointer_count']}" for f, s in val["family"].items()))
        lines.append("  场景CE：" + json.dumps(val["scene_group_language_CE"], ensure_ascii=False))
        lines.append("  原验证目标组成：" + json.dumps(val["target_composition"], ensure_ascii=False))
        if run["error"]:
            lines.append("  失败：" + run["error"])
    lines += ["", "五、三种子汇总（仅开发结果）"]
    for method, summary in audit["method_summaries"].items():
        exact, ce = summary["validation_exact_rate"], summary["validation_scene_group_CE"]
        lines.append(f"{method}：完成seed={summary['completed_seeds']}；精确率均值={number(exact['mean'])}，样本SD={number(exact['sample_SD'])}；CE均值={number(ce['mean'])}，样本SD={number(ce['sample_SD'])}。")
    lines += [audit["selection_warning"], audit["statistical_unit_warning"],
              "原验证集以澄清答案为主，declared_pick还可能集中于同一个OBJ别名；上面的组成由真实trace逐项统计。",
              "因此整体exact很高也不能单独证明模型读取了交互证据；需看以下双侧对象改变与联合别名重绑定。",
              "本轮不据此宣布基线优势、统计显著、真实脑机鲁棒或未知场景泛化。", "",
              "六、失败尝试与成本"]
    for cost in audit["attempt_costs"]:
        lines.append(f"attempt{cost['attempt']}：{cost['status']}；报告总耗时={number(cost['reported_total_wall_s'],2)}秒；训练optimizer={cost['training_optimizer_steps']}步；烟测记录={cost['recorded_smoke_iterations']}次。")
        if cost["error"]:
            lines.append("  原始失败：" + cost["error"])
        for run in cost["runs"]:
            lines.append(f"  {run['method']}/seed{run['seed']}：{run['status']}，训练{run['training_optimizer_steps']}步，烟测{run['recorded_smoke_iterations']}次，耗时{number(run['reported_elapsed_s'],2)}秒，错误={run['error']}。")
    lines += [f"历次报告总耗时合计={number(audit['total_reported_attempt_wall_s'],2)}秒；训练optimizer合计={audit['total_training_optimizer_steps_all_attempts']}步；烟测记录合计={audit['total_recorded_smoke_iterations_all_attempts']}次。",
              "attempt总耗时含加载/预检/验证，不再与run耗时相加。早期烟测逐步梯度缺日志时不编造更新次数。", "",
              "七、训练后对象绑定与别名诊断"]
    challenge = audit["development_challenges"]
    lines += ["真实challenge状态：" + challenge["status"]]
    if challenge["status"] == "NOT_AVAILABLE":
        lines.append(challenge["note"])
    else:
        lines += ["实际计数：" + json.dumps(challenge["counts"], ensure_ascii=False),
                  "额外GPU诊断耗时：" + number(challenge["elapsed_s"], 2) + "秒（与训练成本分列）。",
                  challenge["interpretation"], challenge["auxiliary_scope"]]
        for run in challenge["runs"]:
            lines.append(f"{run['method']}/seed{run['seed']}：实际unique public={run['unique_public_examples']}。")
            for kind, values in run["pairs"].items():
                lines.append(f"  {kind}：双侧同时正确={values['both_correct']}/{values['count']}；输出变化={values['output_changed']}/{values['count']}；移除交互后首侧正确={values['first_without_interaction_exact']}/{values['count']}。")
                lines.append("    按family：" + json.dumps(values["family"], ensure_ascii=False))
            lines.append("  原validation辅助头实测：" + json.dumps(run["auxiliary"], ensure_ascii=False))
        lines += ["诊断缺项：" + problem for problem in challenge["validation_errors"]]
        lines.append("按种子均值/样本SD：" + json.dumps(challenge["methods"], ensure_ascii=False))
    lines += ["", "八、本地CPU机制测试与checkpoint",
              "CPU实际日志汇总：" + json.dumps(audit["local_cpu_tests"], ensure_ascii=False),
              f"选中权重重新计算SHA256：{sum(a['verified'] for a in audit['selected_local_checkpoint_verification'])}/{len(audit['selected_local_checkpoint_verification'])}份通过。",
              audit["artifact_scope"], "各文件路径/SHA256见08_training_audit.json与checkpoint manifest。", "", "九、剩余工作"]
    lines += ["- " + p for p in audit["missing_or_failed_requirements"]]
    lines += ["- 步骤08整体未完成：补放置阶段、可辨识含噪隐藏意图监督及更充分分组开发验证。",
              "- 未做低层实例到达/抓取、完整VLA闭环或任务成功率评价。",
              "- 正式test封存；真实EEG/实体机器人排除；可靠性未校准，交互头不等于已部署澄清策略。",
              "原始report：" + audit["source_report"], "审计：execution/v4/08_training_audit.json。",
              "交接：execution/v4/handoff_step08.txt。", ""]
    return "\n".join(lines)


def finalize(attempt, artifacts_path, challenges_path=None):
    targets = [ROOT / "docs/v4/08_training.txt", OUT / "08_training_audit.json", OUT / "08_acceptance.json", OUT / "handoff_step08.txt"]
    if any(path.exists() for path in targets) or (OUT / "08_finalization_before.json").exists():
        raise FileExistsError("Preserve previous Step08 finalization evidence; do not overwrite")
    audit = build_audit(attempt, artifacts_path, challenges_path)
    progress_path = ROOT / "execution/v4_plan_progress.json"
    progress = read(progress_path)
    before = deepcopy(progress)
    complete = audit["restricted_development_training_complete"]
    summary = "受限开发训练完成（四方法×三种子及本地权重核验）；步骤08整体未完成。" if complete else "步骤08部分通过，训练或本地权重核验有具体缺项；整体未完成。"
    block = "\n".join(["V4-08最新状态：" + summary,
                       "实际LIBERO＋模拟交互；公开声明教学，128训练/32验证，结果按3seed分别记录。",
                       "报告docs/v4/08_training.txt；审计execution/v4/08_training_audit.json；交接execution/v4/handoff_step08.txt。",
                       "放置/真实含噪隐藏意图/外部泛化未完成；完整VLA闭环未运行；旧G2/G3/G4不升级。", ""])
    handoff = block + "\n".join([
        "POINT是公开声明绑定；同public隐藏反事实仍保留歧义并只计一次权重。",
        "empty夹爪子任务为纯文本符号干预，不是实际抓取/放置帧。",
        "原始报告：" + audit["source_report"] + "；本地checkpoint清单：" + str(artifacts_path),
        "历次失败及成本全部保留；烟测后重置共同03R初始化；validation参与checkpoint选优。",
        "原验证目标组成偏澄清且declared_pick对象集中；不能由整体exact声称理解意图。",
        "训练后challenge状态：" + audit["development_challenges"]["status"] + "；双侧正确与别名重绑定按family分别记录，三seed不增独立样本。",
        "CPU机制测试实际日志：" + json.dumps(audit["local_cpu_tests"], ensure_ascii=False),
        "具体缺项：" + ("；".join(audit["missing_or_failed_requirements"]) or "受限pilot记录完整；广义监督与外部验证仍缺"),
        "full_step08=false；不自动推进正式实验。正式test/校准未使用，真实EEG/实体机器人排除。",
        "动作专家结构不改，无新动作采样/完整VLA闭环；旧G2/G3/G4保持原结论。", ""])
    step = next(s for s in progress["steps"] if s["step_id"] == "V4-08")
    step.update(status="部分完成（受限开发训练完成）" if complete else "部分通过（具体缺项见审计）",
                full_step_complete=False, restricted_development_training_complete=complete,
                updated_at_utc=audit["completed_at_utc"], completion_scope=summary,
                actual_artifacts=["bci_vla/training_v4", "configs/v4/train_step08_pilot.json", "configs/v4/train_data_step08.json",
                                  "docs/v4/08_training.txt", "execution/v4/08_training_audit.json", "execution/v4/08_acceptance.json",
                                  "execution/v4/handoff_step08.txt", audit["source_report"]],
                verification={"status": audit["status"], "report": "execution/v4/08_training_audit.json",
                              "complete_runs": sum(r["status"] == COMPLETE for r in audit["runs"]), "full_step08_complete": False})
    any_training = any(r["optimizer_steps"] > 0 for r in audit["runs"])
    progress.update(current_stage="V4-08_restricted_training_complete_full_step_incomplete" if complete else "V4-08_partial_training_or_artifact_gaps",
                    next_step="V4-08", latest_handoff="execution/v4/handoff_step08.txt",
                    training_started=any_training, neural_baseline_training_run=any_training,
                    interaction_prefix_trained=any(r["method"] == "Full" and r["status"] == COMPLETE for r in audit["runs"]),
                    restricted_development_training_complete=complete, full_step08_complete=False,
                    training_data_ready=False, on_prefix_semantics_passed=False,
                    formal_experiments_run=False, full_VLA_closed_loop_run=False,
                    new_experiments_run_meaning="restricted public-declaration development training only; no hidden-intent/action/test acceptance",
                    step08_dependency="Remaining placement/hidden-intent supervision and external development validation")
    assert progress["historical_v2_gates"] == before["historical_v2_gates"]
    assert all(s == before["steps"][i] for i, s in enumerate(progress["steps"]) if s["step_id"] != "V4-08")
    acceptance = {k: audit[k] for k in ("step_id", "attempt", "status", "completed_at_utc", "restricted_development_training_complete", "full_step08_complete", "missing_or_failed_requirements")}
    acceptance.update(complete_runs=sum(r["status"] == COMPLETE for r in audit["runs"]),
                      verified_selected_local_weights=sum(a["verified"] for a in audit["selected_local_checkpoint_verification"]),
                      general_hidden_intent_robustness_proven=False, place_phase_ready=False,
                      external_generalization_proven=False, historical_gates_unchanged=True,
                      development_challenges_status=audit["development_challenges"]["status"],
                      local_cpu_tests=audit["local_cpu_tests"],
                      next_step="V4-08 remaining supervision/development checks")
    tracked = [progress_path, ROOT / "README.txt", ROOT / "execution/worklog.txt", ROOT / "execution/handoff.txt", OUT / "handoff.txt"]
    snapshot = {"created_at_utc": audit["completed_at_utc"], "historical_v2_gates": before["historical_v2_gates"],
                "files_before": [{"path": str(p), "sha256": digest(p)} for p in tracked if p.is_file()]}
    write_json(OUT / "08_finalization_before.json", snapshot)
    write_json(OUT / "08_training_audit.json", audit)
    write_json(OUT / "08_acceptance.json", acceptance)
    write_text(ROOT / "docs/v4/08_training.txt", document(audit))
    write_text(OUT / "handoff_step08.txt", handoff)
    write_json(progress_path, progress)
    for target in (ROOT / "README.txt", ROOT / "execution/handoff.txt", OUT / "handoff.txt"):
        old = target.read_text(encoding="utf-8-sig") if target.exists() else ""
        write_text(target, block + "\n历史记录（保留原范围）：\n" + old)
    worklog = ROOT / "execution/worklog.txt"
    old = worklog.read_text(encoding="utf-8-sig") if worklog.exists() else ""
    write_text(worklog, old.rstrip() + "\n\n工作记录：" + audit["completed_at_utc"] + "，V4-08受限开发训练收尾\n" + block +
               "实际complete_runs=" + str(acceptance["complete_runs"]) + "；verified_local_weights=" + str(acceptance["verified_selected_local_weights"]) +
               "；历次报告总耗时=" + number(audit["total_reported_attempt_wall_s"], 2) + "秒。\n" +
               "具体缺项：" + ("；".join(audit["missing_or_failed_requirements"]) or "受限pilot记录完整；步骤08整体仍有监督/外部验证缺口") + "。\n")
    assert read(progress_path)["historical_v2_gates"] == before["historical_v2_gates"]
    print(json.dumps({"status": audit["status"], "restricted_development_training_complete": complete,
                      "full_step08_complete": False, "complete_runs": acceptance["complete_runs"],
                      "verified_local_weights": acceptance["verified_selected_local_weights"],
                      "missing": audit["missing_or_failed_requirements"]}, ensure_ascii=False))
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=int, default=3)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--challenges", type=Path, help="Optional actual post-training GPU challenge report; missing is never treated as zero")
    args = parser.parse_args()
    if args.attempt != 3:
        raise ValueError("This scoped finalizer requires the actual three-seed attempt3 evidence")
    return finalize(args.attempt, args.artifacts.resolve(), args.challenges.resolve() if args.challenges else None)


if __name__ == "__main__":
    raise SystemExit(main())
