"""Actual-capture simulated-outcome calibration rows, with a public predictor.

No file I/O or model is supplied here. A caller loads capture metadata, provides
a frozen predictor, and owns image loading on that callback's public path. The
original pure step04 generator creates current observations and private scorer
outcomes. Controls bypass prediction. Latent twins share ONE prediction and
become a probability target, never two contradictory identified labels.
"""
from __future__ import annotations

from copy import deepcopy
import math

from bci_vla.data_v4.contracts import validate_private, validate_public
from bci_vla.interaction_contract_v4.validation import canonical_digest
from bci_vla.neural_interaction_v4.control import route_control
from .calibration import validate_binding

ALLOWED_SPLITS = ("calibration_fit", "calibration_audit", "validation")
DEFAULT_VARIANTS = ("stable", "goal_change", "channel_change", "both_change", "no_response",
                    "feature_dropout", "mapping_change", "candidate_miss", "cancel", "stop", "ambiguous")
ROW_GENERATOR_VERSION = "libero-simulated-outcome-rows/0.1"


def _default_generator(capture, variant):
    # Importing this module does not execute its filesystem-writing main().
    from execution.v4.build_step04_data import generate
    return generate(capture, variant)


def _visible_key(public):
    # The old neural_view validates only the frozen 4.0 provenance enum. Use the
    # actual 4.1 tensorizer view, without relabelling real capture provenance.
    from bci_vla.training_v4.text_baseline import compact_model_view
    # IDs are provenance only. The public image path selects an actual image;
    # all event ages, ROI values, aliases and public session state stay visible.
    dictionary = [{key: value for key, value in obj.items() if key != "instance_id"}
                  for obj in public["scene"]["objects"]]
    return canonical_digest({"view": compact_model_view(public), "image": public["scene"]["image_ref"],
                             "dictionary": dictionary})


def _prediction(raw, ids, public):
    if not isinstance(raw, dict) or raw.get("candidate_ids") != ids:
        raise ValueError("predictor must bind logits to the exact valid public candidate axis")
    logits = raw.get("intent_logits")
    if (not isinstance(logits, (list, tuple)) or len(logits) != len(ids) + 1
            or any(type(v) not in (float, int) or not math.isfinite(v) for v in logits)):
        raise ValueError("raw intent logits require finite candidates-plus-outside values")
    support = raw.get("channel_supported")
    if type(support) is not bool:
        raise ValueError("predictor channel support must be an explicit bool")
    if support and public["public_state"]["diagnostics"]["status"] != "SUFFICIENT":
        raise ValueError("predictor claimed channel support without sufficient public diagnostics")
    channel = raw.get("channel_logit")
    if support and (type(channel) not in (float, int) or not math.isfinite(channel)):
        raise ValueError("supported channel requires a finite raw Bernoulli logit")
    return {"intent_logits": list(logits), "channel_logit": float(channel) if support else None,
            "channel_supported": support}


def _future_success(future, channel, public):
    if future is None or type(future.get("success")) is not bool or future["success"] != channel["next_probe_success"]:
        raise ValueError("supported channel outcome is missing its private future PROBE receipt")
    question = future.get("question", {})
    diagnostic = public["public_state"]["diagnostics"]
    known = question.get("diagnostic_option_id")
    options = question.get("options", [])
    symbol = next((o.get("symbol") for o in options if o.get("option_id") == known), None)
    if (question.get("role") != "PROBE" or question.get("context") != public["context"]
            or question.get("layout_id") != diagnostic["layout_id"]
            or question.get("issued_at_s") != channel["target_time_s"]
            or known is None or symbol not in diagnostic["covered_symbols"]):
        raise ValueError("future PROBE mapping/layout/time/known-symbol is not supported by current public diagnostics")
    answer = future.get("answer", {})
    observed = answer.get("selected_option_id")
    if observed is not None and observed not in {o["option_id"] for o in options}:
        raise ValueError("future PROBE answer outside its actual stimulus options")
    if future["success"] != (observed == known):
        raise ValueError("future PROBE outcome contradicts its observed answer")
    return int(future["success"])


def build_calibration_rows(captures, predictor, *, split, binding, variants=None, max_episodes=None, generator=None):
    """Return intent_rows/channel_rows/counts/scope from allowed capture metadata.

    predictor(public) -> candidate_ids, intent_logits[N+1 incl outside],
    channel_logit (scalar/None), channel_supported (bool). No label, latent goal,
    future PROBE or world object is passed to it. Input captures may be a full
    manifest or list; partition filtering occurs before any generation.

    These are outcome calibration targets under the declared simulated variant
    mixture and fixed historical question schedule, NOT identified POINT intent
    supervision, policy-closed-loop calibration, EEG or manipulation success.
    """
    if split not in ALLOWED_SPLITS:
        raise ValueError("runtime rows permit calibration_fit/calibration_audit/validation only; test stays sealed")
    binding = validate_binding(binding)
    if not callable(predictor):
        raise TypeError("a public-only frozen model predictor is required")
    variants = tuple(DEFAULT_VARIANTS if variants is None else variants)
    if not variants or len(set(variants)) != len(variants) or any(v not in DEFAULT_VARIANTS for v in variants):
        raise ValueError("explicit unique supported noisy variants required")
    if max_episodes is not None and (type(max_episodes) is not int or max_episodes < 1):
        raise ValueError("positive optional base-episode limit required")
    episodes = captures["episodes"] if isinstance(captures, dict) else captures
    selected = [capture for capture in episodes if capture.get("partition") == split]
    # No frame, label or source task from an excluded capture is accessed.
    selected.sort(key=lambda c: (c["group_id"], c["base_episode_id"]))
    if max_episodes is not None:
        selected = selected[:max_episodes]
    if not selected:
        raise ValueError("no actual captures in requested allowed partition")
    if len({c["base_episode_id"] for c in selected}) != len(selected):
        raise ValueError("duplicate base capture would inflate outcome weight")
    generate = generator or _default_generator
    cache, accumulators = {}, {}
    counts = {"base_episodes": len(selected), "scene_groups": len({c["group_id"] for c in selected}),
              "generated_public_records": 0, "control_records_bypassed": 0, "predictor_calls": 0,
              "latent_twin_pairs": 0, "private_outcomes": 0, "deduplicated_public_records": 0}
    for capture in selected:
        for variant in variants:
            records, _environment_world = generate(deepcopy(capture), variant)
            for public, private, future in records:
                counts["generated_public_records"] += 1
                validate_public(public); validate_private(private, public)
                if (private["split"] != split or public["context"]["scene_id"] != capture["group_id"]
                        or public["context"]["session_id"] != "session-" + capture["base_episode_id"]
                        or public["scene"]["image_ref"] not in {f["image_path"] for f in capture["frames"]}):
                    raise ValueError("generated observation provenance differs from the filtered capture")
                if not route_control(public)["call_neural_module"]:
                    counts["control_records_bypassed"] += 1
                    continue
                ids = [o["instance_id"] for o in public["scene"]["objects"] if o["object_valid"]]
                if not ids:
                    continue
                visible = _visible_key(public)
                # Cache per base episode so raw instance IDs never migrate across
                # scenes that happen to share assets or rendered appearances.
                key = (capture["base_episode_id"], visible)
                if key not in cache:
                    raw = predictor(deepcopy(public))
                    cache[key] = _prediction(raw, ids, public)
                    counts["predictor_calls"] += 1
                else:
                    counts["deduplicated_public_records"] += 1
                raw = cache[key]
                if key not in accumulators:
                    accumulators[key] = {"raw": raw, "ids": ids, "public": public,
                                         "group_id": capture["group_id"], "episode_id": capture["base_episode_id"],
                                         "intent": [0.] * (len(ids) + 1), "intent_weight": 0.,
                                         "channel_success": 0., "channel_weight": 0., "latent_variants": 0,
                                         "source_hashes": set(), "variants": set()}
                accumulator = accumulators[key]
                if accumulator["ids"] != ids:
                    raise ValueError("same visible input has inconsistent candidate instance binding")
                outcomes = [private]
                if variant == "ambiguous":
                    twin = deepcopy(private)
                    other = next((oid for oid in ids if oid != private["hidden_truth"]["goal"]["source_id"]), None)
                    if other is not None:
                        twin["variant_id"] = "latent-1"
                        twin["hidden_truth"]["goal"]["source_id"] = other
                        validate_private(twin, public)
                        outcomes.append(twin)
                        counts["latent_twin_pairs"] += 1
                weight = 1. / len(outcomes)
                for outcome in outcomes:
                    goal = outcome["hidden_truth"].get("goal")
                    if goal is not None:
                        target = ids.index(goal["source_id"]) if goal["source_id"] in ids else len(ids)
                        accumulator["intent"][target] += weight
                        accumulator["intent_weight"] += weight
                    channel = outcome["labels"]["channel"]
                    if channel["mask"]:
                        success = _future_success(future, channel, public)
                        # Future is loss-only; public support was independently
                        # checked before this outcome was inspected.
                        accumulator["channel_success"] += weight * success
                        accumulator["channel_weight"] += weight
                    counts["private_outcomes"] += 1
                accumulator["latent_variants"] += len(outcomes)
                accumulator["source_hashes"].add(canonical_digest(public))
                accumulator["variants"].add(variant)
    intent_rows, channel_rows = [], []
    for (_episode, visible), item in sorted(accumulators.items()):
        raw = item["raw"]
        common = {"split": split, "group_id": item["group_id"], "episode_id": item["episode_id"],
                  "metadata": {"binding": dict(binding), "visible_input_sha256": visible,
                               "source_public_sha256": sorted(item["source_hashes"]),
                               "latent_outcomes": item["latent_variants"], "variants": sorted(item["variants"])},
                  "target_mode": "PROBABILITY", "label_scope": "simulated_outcome_distribution_not_identified_POINT"}
        weight = item["intent_weight"]
        intent_rows.append({**deepcopy(common), "logits": raw["intent_logits"],
                            "target": [v / weight for v in item["intent"]] if weight else None,
                            "mask": weight > 0, "supported": True, "candidate_ids": item["ids"]})
        channel_weight = item["channel_weight"]
        supported = raw["channel_supported"] and channel_weight > 0
        channel_rows.append({**deepcopy(common), "logits": raw["channel_logit"] if supported else None,
                             "target": item["channel_success"] / channel_weight if supported else None,
                             "mask": supported, "supported": supported})
    counts.update(intent_rows=len(intent_rows), channel_rows=len(channel_rows),
                  channel_supported_rows=sum(r["supported"] for r in channel_rows))
    return {"version": ROW_GENERATOR_VERSION, "split": split, "binding": dict(binding), "intent_rows": intent_rows, "channel_rows": channel_rows,
            "counts": counts, "scope": "actual LIBERO capture observations; simulated noisy outcome calibration under fixed historical questions",
            "variant_mixture": "equal configured variant contributions; each ambiguous latent pair shares that contribution equally",
            "row_weighting": "one unique visible snapshot per base episode; latent twins do not multiply prediction/sample weight",
            "policy_closed_loop": False, "identified_POINT_supervision": False, "real_eeg": False,
            "formal_test_accessed": False, "VLA_called": False}
