"""Annotation-isolated evaluation and scene-cluster bootstrap confidence intervals.

Correctness is scored on the structured task, never by an LLM judging prose.
The common renderer makes all ready instructions comparable to the same grammar.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Iterable

from .planning import candidate_key, compatible_candidates, generate_candidates, structural_clarification
from .rendering import render_instruction
from .schema import public_observation, validate_case


def _intent_key(intent: Any) -> str | None:
    if not isinstance(intent, dict):
        return None
    try:
        return candidate_key(intent)
    except (KeyError, TypeError, ValueError):
        return None


def _slot(prediction: dict[str, Any]) -> Any:
    if prediction.get("clarification_slot") is not None:
        return prediction["clarification_slot"]
    clarification = prediction.get("clarification")
    return clarification.get("slot") if isinstance(clarification, dict) else None


def _score(proposal: dict[str, Any], gold: dict[str, Any], grammar: set[str],
           compatible: set[str], unresolved: bool) -> dict[str, Any]:
    status = proposal.get("status", "invalid")
    intent = _intent_key(proposal.get("intent"))
    expected = _intent_key(gold.get("intent"))
    ready, gold_ready = status == "ready", gold["status"] == "ready"
    clarify, gold_clarify = status == "clarify", gold["status"] == "clarify"
    intent_correct = ready and gold_ready and intent is not None and intent == expected
    slot_correct = clarify and gold_clarify and _slot(proposal) == gold.get("clarification_slot")
    exact = bool(status == gold["status"] and (
        intent_correct if gold_ready else slot_correct if gold_clarify else True
    ))
    return {
        "status": status, "intent_key": intent, "clarification_slot": _slot(proposal),
        "status_correct": status == gold["status"], "decision_exact": exact,
        "ready": ready, "intent_correct": bool(intent_correct),
        "clarify": clarify, "clarification_correct": bool(slot_correct),
        "grammar_invalid_ready": bool(ready and intent not in grammar),
        "context_invalid_ready": bool(ready and intent not in compatible),
        "unresolved_ready": bool(ready and unresolved),
        "invalid_ready": bool(ready and (intent not in compatible or unresolved)),
        "ready_when_should_not": bool(ready and not gold_ready),
        "wrong_ready": bool(ready and not intent_correct),
    }


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    gold_ready = sum(row["gold_status"] == "ready" for row in rows)
    gold_clarify = sum(row["gold_status"] == "clarify" for row in rows)
    eligible = sum(row["candidate_eligible_gold"] for row in rows)
    output: dict[str, Any] = {
        "n_cases": n, "n_scenes": len({row["scene_id"] for row in rows}),
        "n_gold_ready": gold_ready, "n_gold_clarify": gold_clarify,
        "n_candidate_eligible_gold": eligible,
        "gold_candidate_coverage": _ratio(eligible, gold_ready),
        "prediction_error_rate": _ratio(sum(bool(row["error_type"]) for row in rows), n),
    }
    for prefix in ("raw", "guarded"):
        ready = sum(row[prefix + "_ready"] for row in rows)
        clarify = sum(row[prefix + "_clarify"] for row in rows)
        correct = sum(row[prefix + "_intent_correct"] for row in rows)
        clarification_correct = sum(row[prefix + "_clarification_correct"] for row in rows)
        output[prefix] = {
            "status_accuracy": _ratio(sum(row[prefix + "_status_correct"] for row in rows), n),
            "decision_exact_match": _ratio(sum(row[prefix + "_decision_exact"] for row in rows), n),
            "ready_coverage": _ratio(ready, n),
            "instruction_precision_on_ready": _ratio(correct, ready),
            "instruction_recall_on_gold_ready": _ratio(correct, gold_ready),
            "candidate_conditional_instruction_accuracy": _ratio(sum(
                row[prefix + "_intent_correct"] and row["candidate_eligible_gold"] for row in rows), eligible),
            "clarification_recall_on_gold_clarify": _ratio(clarification_correct, gold_clarify),
            "clarification_precision": _ratio(clarification_correct, clarify),
            "invalid_ready_rate_on_ready": _ratio(sum(row[prefix + "_invalid_ready"] for row in rows), ready),
            "grammar_invalid_ready_rate_on_ready": _ratio(sum(
                row[prefix + "_grammar_invalid_ready"] for row in rows), ready),
            "context_invalid_ready_rate_on_ready": _ratio(sum(
                row[prefix + "_context_invalid_ready"] for row in rows), ready),
            "unresolved_ready_rate_on_ready": _ratio(sum(row[prefix + "_unresolved_ready"] for row in rows), ready),
            "ready_when_should_not_rate": _ratio(sum(row[prefix + "_ready_when_should_not"] for row in rows), n),
            "wrong_ready_rate_on_ready": _ratio(sum(row[prefix + "_wrong_ready"] for row in rows), ready),
            "n_ready": ready,
            "n_clarify": clarify,
            "status_confusion": dict(Counter(
                row["gold_status"] + "->" + str(row[prefix + "_status"]) for row in rows)),
        }
    measured = [row["service_latency_ms"] for row in rows]
    live = [row["service_latency_ms"] for row in rows if not row["cache_hit"]]
    replay = [row["service_latency_ms"] for row in rows if row["cache_hit"]]
    remote = [row["recorded_remote_latency_ms"] for row in rows
              if isinstance(row["recorded_remote_latency_ms"], (int, float))]
    output["latency"] = {
        "service_median_ms": _quantile(measured, .5), "service_p95_ms": _quantile(measured, .95),
        "live_service_median_ms": _quantile(live, .5), "live_service_p95_ms": _quantile(live, .95),
        "replay_service_median_ms": _quantile(replay, .5),
        "recorded_remote_median_ms": _quantile(remote, .5),
        "n_cache_replay": len(replay), "n_live_or_local": len(live),
        "note": "Replay service latency is not LLM inference latency; includes failures and retries.",
    }
    rendered = [row["guarded_instruction_matches_renderer"] for row in rows
                if row.get("guarded_instruction_matches_renderer") is not None]
    output["guarded"]["renderer_match_rate_on_ready"] = _ratio(sum(rendered), len(rendered))
    return output


def _numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    flattened = {}
    for prefix in ("raw", "guarded"):
        for name, value in metrics[prefix].items():
            if isinstance(value, (float, int)) and not name.startswith("n_"):
                flattened[prefix + "." + name] = float(value)
    return flattened


def _bootstrap(rows: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["scene_id"]].append(row)
    scenes = sorted(grouped)
    rng = random.Random(seed)
    collected: dict[str, list[float]] = defaultdict(list)
    if scenes:
        for _ in range(samples):
            sampled = [row for _ in scenes for row in grouped[rng.choice(scenes)]]
            for key, value in _numeric_metrics(_aggregate(sampled)).items():
                collected[key].append(value)
    return {
        "method": "percentile scene-cluster bootstrap; resample whole scenes with replacement",
        "samples": samples, "seed": seed, "confidence_level": .95,
        "warning": "Intervals are unstable with few independent scenes; variants are not independent samples.",
        "intervals": {name: {"low": _quantile(values, .025), "high": _quantile(values, .975),
                              "defined_resamples": len(values)} for name, values in sorted(collected.items())},
    }


def evaluate(predictor: Any, cases: Iterable[dict[str, Any]], output_dir: str | Path | None = None,
             bootstrap_samples: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Evaluate every case; the predictor receives only ``public_observation``.

    API and parsing errors remain in the denominator. ``ready`` means an
    instruction proposal requiring confirmation, never an executed action.
    Optional reports contain no provider raw text, request headers, or credentials.
    """
    if not isinstance(bootstrap_samples, int) or not 0 <= bootstrap_samples <= 100000:
        raise ValueError("bootstrap_samples must be an integer between 0 and 100000")
    validated = [validate_case(case) for case in cases]
    if len({case["case_id"] for case in validated}) != len(validated):
        raise ValueError("Duplicate case IDs are not independent evaluation cases")
    rows: list[dict[str, Any]] = []
    for case in validated:
        obs = public_observation(case)
        gold = case["annotation"]
        grammar = {_intent_key(item) for item in generate_candidates(obs)} - {None}
        compatible = {_intent_key(item) for item in compatible_candidates(obs)} - {None}
        unresolved = structural_clarification(obs) is not None
        started = time.perf_counter()
        failure = None
        try:
            result = predictor.predict(obs)
            if not isinstance(result, dict):
                raise ValueError("Predictor did not return a dictionary")
        except Exception as exc:
            failure = type(exc).__name__
            result = {"status": "reject", "intent": None, "clarification_slot": None}
        elapsed = (time.perf_counter() - started) * 1000
        raw = result.get("raw_proposal", result)
        if not isinstance(raw, dict):
            raw = {"status": "invalid", "intent": None, "clarification_slot": None}
        metadata = result.get("llm_metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        row: dict[str, Any] = {
            "case_id": case["case_id"], "scene_id": case["scene_id"],
            "scenario": case.get("scenario"), "provenance": case.get("provenance"),
            "gold_status": gold["status"], "gold_intent_key": _intent_key(gold.get("intent")),
            "gold_clarification_slot": gold.get("clarification_slot"),
            "candidate_eligible_gold": bool(gold["status"] == "ready" and _intent_key(gold.get("intent")) in grammar),
            "service_latency_ms": elapsed, "cache_hit": bool(metadata.get("cache_hit", False)),
            "recorded_remote_latency_ms": metadata.get("remote_latency_ms"),
            "error_type": failure or result.get("error") or metadata.get("error_type"),
            "request_sha256": metadata.get("request_sha256"),
            "model_version": metadata.get("model_version"),
            "attempts": metadata.get("attempts"),
        }
        for prefix, prediction in (("raw", raw), ("guarded", result)):
            row.update({prefix + "_" + key: value for key, value in
                        _score(prediction, gold, grammar, compatible, unresolved).items()})
            if row["error_type"]:
                # A provider outage or malformed completion is a failed case,
                # even if its rejection happens to match a gold reject label.
                for key in ("status_correct", "decision_exact", "intent_correct", "clarification_correct"):
                    row[prefix + "_" + key] = False
        # Rendering is checked only for readiness and never used as a free-text
        # semantic judge. All systems share this renderer after their proposal.
        row["guarded_instruction_matches_renderer"] = None
        if result.get("status") == "ready":
            try:
                row["guarded_instruction_matches_renderer"] = (
                    result.get("instruction") == render_instruction(obs, result["intent"]))
            except (ValueError, TypeError, KeyError):
                row["guarded_instruction_matches_renderer"] = False
        rows.append(row)
    settings = predictor.reproducibility_settings() if hasattr(predictor, "reproducibility_settings") else {
        "predictor": type(predictor).__name__}
    dataset_digest = hashlib.sha256(json.dumps(validated, sort_keys=True, ensure_ascii=False,
                                               separators=(",", ":")).encode("utf-8")).hexdigest()
    report = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "dataset_sha256": dataset_digest, "predictor": settings,
              "provenance": sorted({str(row["provenance"]) for row in rows if row["provenance"]}),
              "metrics": _aggregate(rows), "bootstrap": _bootstrap(rows, bootstrap_samples, seed),
              "interpretation": {
                  "raw": "Predictor proposal before shared compatibility, ambiguity and grounding checks.",
                  "guarded": "Output after shared guard; guard corrections are not learned reasoning gains.",
                  "instruction_scoring": "Exact action/source/destination; no free-text LLM judging.",
                  "error_scoring": "API, parse and predictor failures count as incorrect even on gold-reject cases.",
                  "candidate_conditional": "Gold-ready cases whose gold intent exists in grammar candidates; also report unconditional recall.",
                  "generalization": "Scene-disjoint test results only; synthetic smoke scores are not empirical evidence.",
              }, "per_case": rows}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["scenario"] is not None:
            groups[str(row["scenario"])].append(row)
    report["by_scenario_point_estimates"] = {name: _aggregate(group) for name, group in sorted(groups.items())}
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        aggregate = {key: value for key, value in report.items() if key != "per_case"}
        (directory / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
        (directory / "per_case.jsonl").write_text("".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        if rows:
            with (directory / "per_case.csv").open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    return report


def paired_scene_comparison(report_a: dict[str, Any], report_b: dict[str, Any], *,
                            bootstrap_samples: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Paired bootstrap of B minus A on exactly the same cases and scene clusters."""
    if not isinstance(bootstrap_samples, int) or not 0 <= bootstrap_samples <= 100000:
        raise ValueError("bootstrap_samples must be an integer between 0 and 100000")
    if report_a.get("dataset_sha256") != report_b.get("dataset_sha256"):
        raise ValueError("Paired comparison requires the same frozen dataset")
    a = {row["case_id"]: row for row in report_a["per_case"]}
    b = {row["case_id"]: row for row in report_b["per_case"]}
    if not a or set(a) != set(b):
        raise ValueError("Paired comparison requires identical nonempty case IDs")
    grouped: dict[str, list[str]] = defaultdict(list)
    for case_id in a:
        if a[case_id]["scene_id"] != b[case_id]["scene_id"]:
            raise ValueError("Scene mismatch")
        grouped[a[case_id]["scene_id"]].append(case_id)
    scenes = sorted(grouped)
    point_a, point_b = _numeric_metrics(report_a["metrics"]), _numeric_metrics(report_b["metrics"])
    names = sorted(set(point_a) & set(point_b))
    rng = random.Random(seed)
    collected: dict[str, list[float]] = defaultdict(list)
    for _ in range(bootstrap_samples):
        keys = [case_id for _ in scenes for case_id in grouped[rng.choice(scenes)]]
        sample_a = _numeric_metrics(_aggregate([a[key] for key in keys]))
        sample_b = _numeric_metrics(_aggregate([b[key] for key in keys]))
        for name in names:
            if name in sample_a and name in sample_b:
                collected[name].append(sample_b[name] - sample_a[name])
    return {"direction": "B minus A", "method": "paired scene-cluster percentile bootstrap",
            "samples": bootstrap_samples, "seed": seed,
            "differences": {name: {"difference": point_b[name] - point_a[name],
                                    "low": _quantile(collected[name], .025),
                                    "high": _quantile(collected[name], .975),
                                    "defined_resamples": len(collected[name])} for name in names}}
