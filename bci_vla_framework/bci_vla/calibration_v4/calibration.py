"""Scalar temperature calibration with explicit split, support and hash gates.

Rows: split, logits (intent class list, channel scalar logit), target (class
index/bool), mask, supported, group_id, episode_id, metadata={binding: four SHA
fields} (or the four fields directly), and target_mode=POINT for identified
intent or PROBABILITY for an explicit simulated outcome distribution. No
public observation or future labels are opened here. A caller constructs rows
at the loss/evaluation boundary from actual frozen predictions and labels.
"""
from __future__ import annotations

import math
import re

BINDING_KEYS = ("model_sha256", "data_sha256", "input_schema_sha256", "policy_sha256")
VERSION = "bci-vla-temperature/0.2"


def validate_binding(binding):
    if not isinstance(binding, dict) or set(binding) != set(BINDING_KEYS):
        raise ValueError("exact model/data/input-schema/policy SHA256 binding required")
    if any(not isinstance(binding[k], str) or not re.fullmatch("[0-9a-f]{64}", binding[k]) for k in BINDING_KEYS):
        raise ValueError("invalid SHA256 binding")
    return dict(binding)


def _probabilities(logits, temperature, task):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("positive finite temperature required")
    if task == "channel":
        if isinstance(logits, (list, tuple)):
            if len(logits) != 1:
                raise ValueError("channel requires one Bernoulli logit")
            logits = logits[0]
        value = float(logits) / temperature
        if not math.isfinite(value):
            raise ValueError("finite channel logit required")
        p = 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))
        return [1 - p, p]
    if task != "intent" or not isinstance(logits, (list, tuple)) or len(logits) < 2:
        raise ValueError("intent requires at least two class logits, including outside")
    values = [float(v) / temperature for v in logits]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("finite intent logits required")
    maximum = max(values)
    exp = [math.exp(v - maximum) for v in values]
    total = sum(exp)
    return [v / total for v in exp]


def _target(row, task):
    value = row.get("target")
    if task == "channel":
        if row.get("target_mode") == "PROBABILITY":
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("future-PROBE outcome probability must lie in [0,1]")
            return float(value)
        if type(value) not in (bool, int) or value not in (0, 1):
            raise ValueError("channel target must be observed next-PROBE bool/0/1")
        return int(value)
    if row.get("target_mode") == "PROBABILITY":
        if (not isinstance(value, (list, tuple)) or len(value) != len(row["logits"])
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in value)
                or not math.isclose(sum(value), 1., abs_tol=1e-8)):
            raise ValueError("intent outcome probability target must match class axis and sum to one")
        return list(value)
    if type(value) is not int:
        raise ValueError("POINT intent target must be a class index")
    if not 0 <= value < len(row["logits"]):
        raise ValueError("POINT target outside class axis")
    return value


def _prepare(rows, split, task, binding):
    binding = validate_binding(binding)
    if task not in ("intent", "channel"):
        raise ValueError("unknown calibration task")
    rows = list(rows)
    selected, skipped = [], {"masked": 0, "unsupported": 0, "nonpoint": 0}
    for row in rows:
        if row.get("split") != split:
            raise ValueError("calibration partition leak: expected " + split)
        metadata = row.get("metadata", {})
        if validate_binding(metadata.get("binding", metadata)) != binding:
            raise ValueError("prediction row model/data/schema/policy hash mismatch")
        if any(not isinstance(row.get(k), str) or not row[k] for k in ("group_id", "episode_id")):
            raise ValueError("scene group and base episode provenance required")
        if type(row.get("mask")) is not bool or type(row.get("supported")) is not bool:
            raise ValueError("independent boolean mask and support required")
        if not row["mask"]:
            skipped["masked"] += 1
            continue
        if not row["supported"]:
            skipped["unsupported"] += 1
            continue
        if task == "intent" and row.get("target_mode") not in ("POINT", "PROBABILITY"):
            skipped["nonpoint"] += 1
            continue
        probabilities = _probabilities(row["logits"], 1., task)
        target = _target(row, task)
        selected.append((row, target, len(probabilities)))
    return rows, selected, skipped


def _weights(selected):
    groups = {}
    for row, _, _ in selected:
        episodes = groups.setdefault(row["group_id"], {})
        episodes[row["episode_id"]] = episodes.get(row["episode_id"], 0) + 1
    return [1 / len(groups) / len(groups[r["group_id"]]) / groups[r["group_id"]][r["episode_id"]]
            for r, _, _ in selected] if groups else []


def _target_distribution(target, task, classes):
    if task == "channel":
        return [1 - float(target), float(target)]
    return target if isinstance(target, list) else [float(i == target) for i in range(classes)]


def _nll(selected, weights, temperature, task):
    result = 0.
    for (row, target, classes), weight in zip(selected, weights):
        p = _probabilities(row["logits"], temperature, task)
        q = _target_distribution(target, task, classes)
        result += weight * -sum(v * math.log(max(p[i], 1e-15)) for i, v in enumerate(q))
    return result


def fit_temperature(rows, *, task, binding, min_samples=2, min_groups=1, bounds=(.05, 20.)):
    """Fit only calibration_fit; group/episode macro weights avoid row inflation.

    A deterministic bounded log-temperature grid plus local golden refinement
    minimizes weighted NLL. Insufficient support returns an unfitted artifact,
    whose inference probabilities are null, rather than an invented confidence.
    """
    if type(min_samples) is not int or min_samples < 1 or type(min_groups) is not int or min_groups < 1:
        raise ValueError("positive support limits required")
    if len(bounds) != 2 or not (0 < bounds[0] <= 1 <= bounds[1]) or not all(math.isfinite(v) for v in bounds):
        raise ValueError("finite positive temperature bounds must contain 1")
    all_rows, selected, skipped = _prepare(rows, "calibration_fit", task, binding)
    groups = sorted({r["group_id"] for r in all_rows})
    episodes = sorted({r["group_id"] + "::" + r["episode_id"] for r in all_rows})
    result = {"version": VERSION, "task": task, "binding": validate_binding(binding),
              "fit_split": "calibration_fit", "fit_groups": groups, "fit_episodes": episodes,
              "samples": len(selected), "skipped": skipped, "temperature": None,
              "status": "INSUFFICIENT_SUPPORT", "scope": "synthetic_BCI_domain_only",
              "weighting": "equal scene groups, then equal base episodes, then rows",
              "bounds": list(bounds), "nll_before": None, "nll_after": None}
    result["total_rows"] = len(all_rows)
    result["support_fraction"] = len(selected) / len(all_rows) if all_rows else 0.
    result["target_modes"] = sorted({r.get("target_mode", "OBSERVED_BINARY") for r, _, _ in selected})
    result["outcome_calibration_not_identifiability_supervision"] = any(r.get("target_mode") == "PROBABILITY" for r, _, _ in selected)
    support_groups = {r["group_id"] for r, _, _ in selected}
    if len(selected) < min_samples or len(support_groups) < min_groups:
        return result
    weights = _weights(selected)
    low, high = map(math.log, bounds)
    grid = [low + (high - low) * i / 128 for i in range(129)] + [0.]
    scored = [(_nll(selected, weights, math.exp(t), task), t) for t in grid]
    _, best = min(scored)
    width = (high - low) / 128
    left, right = max(low, best - width), min(high, best + width)
    ratio = (math.sqrt(5) - 1) / 2
    for _ in range(56):
        a, b = right - ratio * (right - left), left + ratio * (right - left)
        if _nll(selected, weights, math.exp(a), task) <= _nll(selected, weights, math.exp(b), task):
            right = b
        else:
            left = a
    choices = scored + [(_nll(selected, weights, math.exp((left + right) / 2), task), (left + right) / 2)]
    after, optimum = min(choices)
    temperature = math.exp(optimum)
    result.update(status="FITTED", temperature=temperature,
                  nll_before=_nll(selected, weights, 1., task), nll_after=after,
                  optimum_at_boundary=abs(optimum - low) < 1e-6 or abs(optimum - high) < 1e-6)
    return result


def _check_artifact(artifact, task, binding):
    if artifact.get("version") != VERSION or artifact.get("task") != task or artifact.get("fit_split") != "calibration_fit":
        raise ValueError("calibration artifact version/task/split mismatch")
    if validate_binding(artifact.get("binding")) != validate_binding(binding):
        raise ValueError("stale model/data/schema/policy calibration binding")


def calibrate_prediction(logits, artifact, *, task, binding, supported=True):
    _check_artifact(artifact, task, binding)
    if type(supported) is not bool:
        raise ValueError("boolean public evidence support required")
    if not supported or artifact.get("status") != "FITTED":
        return {"probabilities": None, "confidence": None, "calibrated": False,
                "reason": "UNSUPPORTED" if not supported else "UNFITTED", "binding": dict(binding)}
    probabilities = _probabilities(logits, artifact["temperature"], task)
    return {"probabilities": probabilities, "confidence": max(probabilities) if task == "intent" else probabilities[1],
            "calibrated": True, "reason": "SUPPORTED", "binding": dict(binding)}


def evaluate_calibration(rows, artifact, *, task, binding, bins=10):
    """Audit only an independent calibration_audit partition; never refit here."""
    _check_artifact(artifact, task, binding)
    if type(bins) is not int or bins < 1:
        raise ValueError("positive integer ECE bin count required")
    all_rows, selected, skipped = _prepare(rows, "calibration_audit", task, binding)
    if {r["group_id"] for r in all_rows} & set(artifact["fit_groups"]):
        raise ValueError("calibration-fit/audit scene groups overlap")
    if {r["group_id"] + "::" + r["episode_id"] for r in all_rows} & set(artifact["fit_episodes"]):
        raise ValueError("calibration-fit/audit episodes overlap")
    result = {"split": "calibration_audit", "task": task, "binding": dict(binding),
              "samples": len(selected), "skipped": skipped, "before": None, "after": None,
              "scope": "synthetic_BCI_domain_only", "fit_performed": False,
              "weighting": "equal scene groups, then equal base episodes, then rows"}
    result["total_rows"] = len(all_rows)
    result["support_fraction"] = len(selected) / len(all_rows) if all_rows else 0.
    result["groups"] = len({r["group_id"] for r, _, _ in selected})
    result["episodes"] = len({(r["group_id"], r["episode_id"]) for r, _, _ in selected})
    if not selected or artifact.get("status") != "FITTED":
        result["status"] = "INSUFFICIENT_SUPPORT"
        return result
    weights = _weights(selected)
    def metrics(temperature):
        brier = distribution_brier = nll = 0.
        bucket = [[0., 0., 0.] for _ in range(bins)]
        for (row, y, classes), weight in zip(selected, weights):
            p = _probabilities(row["logits"], temperature, task)
            q = _target_distribution(y, task, classes)
            # Expected proper outcome score retains irreducible uncertainty of
            # a latent mixture; squared error to q is separately named.
            brier += weight * (p[1] ** 2 - 2 * p[1] * q[1] + q[1] if task == "channel"
                               else sum(v * v - 2 * v * q[j] + q[j] for j, v in enumerate(p)))
            distribution_brier += weight * ((p[1] - q[1]) ** 2 if task == "channel"
                                            else sum((v - q[j]) ** 2 for j, v in enumerate(p)))
            nll += weight * -sum(v * math.log(max(p[j], 1e-15)) for j, v in enumerate(q))
            confidence = p[1] if task == "channel" else max(p)
            outcome = q[1] if task == "channel" else q[max(range(len(p)), key=p.__getitem__)]
            cell = bucket[min(bins - 1, int(confidence * bins))]
            cell[0] += weight; cell[1] += weight * confidence; cell[2] += weight * outcome
        ece = sum(abs(conf - outcome) for mass, conf, outcome in bucket if mass)
        return {"brier": brier, "distribution_brier": distribution_brier, "nll": nll, "ece": ece, "ece_bins": bins,
                "ece_kind": "success_probability" if task == "channel" else "top_label_confidence"}
    result.update(status="AUDITED", before=metrics(1.), after=metrics(artifact["temperature"]))
    return result
