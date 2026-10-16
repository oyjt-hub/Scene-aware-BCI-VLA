"""Probe discrimination and independent holdout diagnostics, never decoder fitting."""
import math

from ..protocols import fields, integer, number, require, text
from .channel import NO_RESPONSE, probability_row


def probe_discrimination(model, *, prompt_symbol, mode_pair, compliance_id, scope, duration_s, assumption):
    require(isinstance(mode_pair, (list, tuple)) and len(mode_pair) == 2, "invalid_mode_pair")
    require(mode_pair[0] != mode_pair[1] and all(z in model.data["mode_ids"] for z in mode_pair), "invalid_mode_pair")
    tv = 0.0
    for output in (*model.data["scope"]["symbols"], NO_RESPONSE):
        values = model.likelihoods("CHANNEL_PROBE", prompt_symbol, output, scope=scope, duration_s=duration_s, compliance_id=compliance_id, assumption=assumption)
        tv += abs(values[mode_pair[0]] - values[mode_pair[1]]) / 2
    return {"total_variation": tv, "status": "simulation_only", "interpretation": "specified compliance model only; not empirical identifiability"}


def transfer_error(predicted_task_rows, observed_counts, *, provenance, source_ref, split):
    """Compare probe-derived task predictions to an independent task holdout.

    Caller supplies the transfer prediction; identity C_probe=C_task is not assumed.
    No confidence interval or acceptable-error threshold is invented from few rows.
    """
    require(provenance in {"synthetic_fixture", "independent_calibration"}, "unknown_evidence_source")
    require(split == "calibration_holdout", "independent_calibration_holdout_required")
    text(source_ref)
    require(isinstance(predicted_task_rows, dict) and bool(predicted_task_rows), "missing_prediction")
    fields(observed_counts, predicted_task_rows.keys())
    rows = {}; weighted = 0.0; n_total = 0
    for key, predicted in predicted_task_rows.items():
        text(key)
        require(isinstance(predicted, dict) and bool(predicted), "invalid_prediction")
        for output in predicted: text(output)
        probability_row(predicted, predicted.keys())
        counts = observed_counts[key]; fields(counts, predicted.keys())
        for n in counts.values(): integer(n)
        n = sum(counts.values())
        require(n <= 2**63 - 1, "count_overflow")
        empirical = {y: counts[y] / n for y in counts} if n else None
        tv = math.fsum(abs(predicted[y] - empirical[y]) for y in predicted) / 2 if n else None
        rows[key] = {"n": n, "empirical_total_variation": tv}
        if n: weighted += n * tv; n_total += n
    return {"status": "simulation_only" if provenance == "synthetic_fixture" else "holdout_estimate_not_validated_for_deployment", "source_ref": source_ref, "split": split, "rows": rows, "n": n_total, "weighted_empirical_total_variation": weighted / n_total if n_total else None, "uncertainty_interval": None, "acceptance_threshold": None, "reason": "sample counts and uncertainty must be reviewed independently; no automatic transfer approval"}
