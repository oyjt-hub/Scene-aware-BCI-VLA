"""Finite categorical likelihoods, with explicit applicability and provenance."""
from __future__ import annotations

from pathlib import Path
import math

from ..protocols import ProtocolError, fields, freeze, json_object, number, require, text, unique_strings

NO_RESPONSE = "NO_RESPONSE"
MODES = {"TASK_QUERY", "CHANNEL_PROBE", "CONFIRM"}


def probability_row(row, keys):
    fields(row, set(keys))
    for v in row.values(): number(v, 0, 1)
    require(abs(math.fsum(row.values()) - 1) <= 1e-9, "row_not_normalized")


class ChannelModel:
    """v1 deliberately refuses live inference and all uncalibrated profiles.

    Matrix orientation is [channel mode][duration string][intended symbol][output].
    A score vector is never accepted as a categorical observation or likelihood.
    """

    def __init__(self, config):
        v = json_object(config)
        require(isinstance(v, dict), "expected_object")
        require(v.get("status") != "uncalibrated", "real_calibration_not_available")
        fields(v, {"schema_version", "profile_id", "status", "provenance", "deployment_allowed", "scope", "mode_ids", "matrices", "compliance"})
        require(v["schema_version"] == "channel-model/1.0", "unsupported_channel_schema")
        text(v["profile_id"])
        require(v["status"] == "simulation_only" and v["provenance"] == "illustrative_not_fitted" and v["deployment_allowed"] is False, "real_calibration_not_available")
        unique_strings(v["mode_ids"])
        require(2 <= len(v["mode_ids"]) <= 8, "unsupported_mode_count")
        s = v["scope"]
        fields(s, {"mapping_id", "stimulus_version", "layout_id", "sample_rate_hz", "duration_s", "symbols", "max_stable_window_s", "observation_assumption"})
        for k in ["mapping_id", "stimulus_version", "layout_id"]: text(s[k])
        number(s["sample_rate_hz"], 1); number(s["max_stable_window_s"], 0)
        unique_strings(s["symbols"])
        require(NO_RESPONSE not in s["symbols"], "reserved_symbol")
        require(isinstance(s["duration_s"], list) and bool(s["duration_s"]), "missing_durations")
        for d in s["duration_s"]:
            number(d, 0); require(0 < d <= s["max_stable_window_s"], "invalid_duration")
        require(len(set(s["duration_s"])) == len(s["duration_s"]), "duplicate_duration")
        require(s["observation_assumption"] == "state_constant_within_window", "unsupported_observation_model")
        outputs = s["symbols"] + [NO_RESPONSE]
        duration_keys = [str(float(d)) for d in s["duration_s"]]
        fields(v["matrices"], MODES)
        for mode in MODES:
            fields(v["matrices"][mode], v["mode_ids"])
            for z in v["mode_ids"]:
                fields(v["matrices"][mode][z], duration_keys)
                for d in duration_keys:
                    matrix = v["matrices"][mode][z][d]
                    fields(matrix, s["symbols"])
                    for row in matrix.values(): probability_row(row, outputs)
        require(isinstance(v["compliance"], dict) and bool(v["compliance"]), "missing_compliance")
        for key, model in v["compliance"].items():
            text(key); fields(model, s["symbols"])
            for row in model.values(): probability_row(row, outputs + ["UNKNOWN"])
        self.data = freeze(v)

    @classmethod
    def load(cls, path):
        return cls(Path(path).read_text(encoding="utf-8-sig"))

    def ensure_applicable(self, scope, duration_s, *, execution_mode="offline", assumption=None):
        require(execution_mode == "offline", "simulation_profile_cannot_run_live")
        require(assumption == "state_constant_within_window", "window_stability_not_established")
        fields(scope, {"mapping_id", "stimulus_version", "layout_id", "sample_rate_hz"})
        require(all(scope[k] == self.data["scope"][k] for k in scope), "outside_calibration_scope")
        number(duration_s, 0)
        require(duration_s in self.data["scope"]["duration_s"], "unsupported_duration_no_interpolation")
        return str(float(duration_s))

    def _matrix(self, mode, intended_symbol, output, scope, duration_s, execution_mode, assumption):
        require(isinstance(mode, str) and mode in MODES, "unsupported_likelihood_mode")
        require(isinstance(intended_symbol, str) and intended_symbol in self.data["scope"]["symbols"], "unknown_intended_symbol")
        require(isinstance(output, str) and output in (*self.data["scope"]["symbols"], NO_RESPONSE), "categorical_output_required")
        d = self.ensure_applicable(scope, duration_s, execution_mode=execution_mode, assumption=assumption)
        return d

    def likelihoods(self, mode, intended_symbol, output, *, scope, duration_s, compliance_id=None, execution_mode="offline", assumption=None):
        d = self._matrix(mode, intended_symbol, output, scope, duration_s, execution_mode, assumption)
        require(mode == "CHANNEL_PROBE" or compliance_id is None, "compliance_only_for_probe")
        if mode == "CHANNEL_PROBE":
            bounds = self.probe_bounds(intended_symbol, output, scope=scope, duration_s=duration_s, compliance_id=compliance_id, execution_mode=execution_mode, assumption=assumption)
            require(all(abs(lo - hi) <= 1e-12 for lo, hi in bounds.values()), "unknown_compliance_use_bounds")
            return {z: lo for z, (lo, hi) in bounds.items()}
        return {z: self.data["matrices"][mode][z][d][intended_symbol][output] for z in self.data["mode_ids"]}

    def probe_bounds(self, prompt_symbol, output, *, scope, duration_s, compliance_id, execution_mode="offline", assumption=None):
        d = self._matrix("CHANNEL_PROBE", prompt_symbol, output, scope, duration_s, execution_mode, assumption)
        require(isinstance(compliance_id, str) and compliance_id in self.data["compliance"], "unknown_compliance_model")
        a = self.data["compliance"][compliance_id][prompt_symbol]
        result = {}
        for z in self.data["mode_ids"]:
            possibilities = [self.data["matrices"]["CHANNEL_PROBE"][z][d][u][output] for u in self.data["scope"]["symbols"]]
            # Omission is a latent user response, separate from decoder omissions.
            possibilities.append(float(output == NO_RESPONSE))
            known = math.fsum(a[u] * self.data["matrices"]["CHANNEL_PROBE"][z][d][u][output] for u in self.data["scope"]["symbols"])
            known += a[NO_RESPONSE] * float(output == NO_RESPONSE)
            result[z] = (known + a["UNKNOWN"] * min(possibilities), known + a["UNKNOWN"] * max(possibilities))
        return result

    def accepted_result_likelihoods(self, result, *, intended_symbol, layout_id, assumption=None, execution_mode="offline", compliance_id=None, nonresponse_duration_s=None):
        """Bridge a step-05 result, preserving status and score semantics."""
        require(result.accepted and result.request is not None and result.response is not None, "accepted_response_required")
        q, r = result.request.data, result.response.data
        require(q["channel_profile_id"] == self.data["profile_id"], "profile_mismatch")
        require(q["mode"] == r["mode"] and r["response_to"] == q["event_id"], "response_binding_mismatch")
        require(set(o["symbol"] for o in q["payload"]["options"]) == set(self.data["scope"]["symbols"]), "outside_calibration_scope")
        status = r["payload"]["status"]
        require(status in {"OBSERVATION", NO_RESPONSE}, "status_not_in_likelihood_model")
        w = r["payload"]["sample_window"]
        if status == "OBSERVATION":
            require(result.likelihood_kind == {"TASK_QUERY": "TASK", "CHANNEL_PROBE": "PROBE", "CONFIRM": "CONFIRM"}.get(q["mode"]), "likelihood_kind_mismatch")
            require(r["payload"]["symbol"] is not None, "score_likelihood_not_calibrated")
            duration = (w["sample_end"] - w["sample_start"]) / w["sample_rate_hz"]
            sample_rate = w["sample_rate_hz"]
            output = r["payload"]["symbol"]
        else:
            require(nonresponse_duration_s is not None, "nonresponse_duration_required")
            duration = nonresponse_duration_s
            sample_rate = self.data["scope"]["sample_rate_hz"]
            output = NO_RESPONSE
        if q["mode"] == "CHANNEL_PROBE":
            require(q["payload"]["probe_prompt"]["presentation_ack"], "probe_not_presented")
            require(intended_symbol == q["payload"]["probe_prompt"]["symbol"], "probe_prompt_mismatch")
        scope = {"mapping_id": q["mapping_id"], "stimulus_version": q["stimulus_version"], "layout_id": layout_id, "sample_rate_hz": sample_rate}
        return self.likelihoods(q["mode"], intended_symbol, output, scope=scope, duration_s=duration, compliance_id=compliance_id, execution_mode=execution_mode, assumption=assumption)


class JointTransitionModel:
    """Two binary labels: independent goal/channel flips plus a common shock.

    This is a CTMC prediction kernel, not a likelihood for within-window changes.
    """
    def __init__(self, config):
        v = json_object(config)
        fields(v, {"status", "provenance", "goal_ids", "mode_ids", "goal_rate_per_s", "channel_rate_per_s", "joint_rate_per_s"})
        require(v["status"] == "simulation_only" and v["provenance"] == "illustrative_not_fitted", "real_transition_calibration_unavailable")
        for k in ["goal_ids", "mode_ids"]:
            unique_strings(v[k]); require(len(v[k]) == 2, "binary_transition_only")
        for k in ["goal_rate_per_s", "channel_rate_per_s", "joint_rate_per_s"]: number(v[k], 0, 1e6)
        self.data = freeze(v)

    def matrix(self, elapsed_s, *, execution_mode="offline"):
        require(execution_mode == "offline", "simulation_profile_cannot_run_live")
        number(elapsed_s, 0, 1e9)
        v = self.data
        p = [-math.expm1(-2 * v[k] * elapsed_s) / 2 for k in ["goal_rate_per_s", "channel_rate_per_s", "joint_rate_per_s"]]
        def bern(bit, prob): return prob if bit else 1 - prob
        states = [(g, z) for g in range(2) for z in range(2)]
        return [[math.fsum(bern(b, p[2]) * bern(g0 ^ g1 ^ b, p[0]) * bern(z0 ^ z1 ^ b, p[1]) for b in [0, 1]) for g1, z1 in states] for g0, z0 in states]

    def any_change_probability(self, elapsed_s):
        number(elapsed_s, 0, 1e9)
        rate = math.fsum(self.data[k] for k in ["goal_rate_per_s", "channel_rate_per_s", "joint_rate_per_s"])
        return -math.expm1(-rate * elapsed_s)
