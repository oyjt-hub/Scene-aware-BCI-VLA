"""Deterministic model identities and deployment guards, not performance tests."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from bci_vla.models.channel import ChannelModel, JointTransitionModel, NO_RESPONSE
from bci_vla.models.probe_transfer import probe_discrimination, transfer_error
from bci_vla.models.evidence import FinalPrefixLedger, summarize_timing
from bci_vla.protocols import ProtocolError
from tests.test_protocols import adapter, BASE, example, ingest

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "configs/channels/simulation_binary_v1.json").read_text(encoding="utf-8"))


def args():
    return {"scope": {k: config()["scope"][k] for k in ["mapping_id", "stimulus_version", "layout_id", "sample_rate_hz"]}, "duration_s": 1.0, "assumption": "state_constant_within_window"}


def accepted(mode="TASK_QUERY", **payload_changes):
    a = adapter(); name = mode.lower()
    q = example(name + "_request"); r = example(name + "_response")
    q["channel_profile_id"] = r["channel_profile_id"] = "simulation-binary-v1"
    context = a.current_context; context["channel_profile_id"] = "simulation-binary-v1"; a.update_context(context)
    a.register_request(q, now=BASE, monotonic_now=0)
    r["payload"].update(payload_changes)
    result = ingest(a, r)
    assert result.accepted, result.reason
    return result


class ChannelTests(unittest.TestCase):
    def test_all_generative_rows_and_effective_probe_outputs_sum_to_one(self):
        model = ChannelModel(config())
        for mode in ["TASK_QUERY", "CHANNEL_PROBE", "CONFIRM"]:
            for u in ["0", "1"]:
                values = [model.likelihoods(mode, u, y, compliance_id="imperfect" if mode == "CHANNEL_PROBE" else None, **args()) for y in ["0", "1", NO_RESPONSE]]
                for z in ["normal", "flip"]:
                    self.assertAlmostEqual(sum(v[z] for v in values), 1)

    def test_task_probe_confirm_are_not_silently_equated(self):
        model = ChannelModel(config())
        self.assertAlmostEqual(model.likelihoods("TASK_QUERY", "0", "0", **args())["normal"], 0.85)
        self.assertAlmostEqual(model.likelihoods("CHANNEL_PROBE", "0", "0", compliance_id="compliant", **args())["normal"], 0.90)
        self.assertAlmostEqual(model.likelihoods("CONFIRM", "0", "0", **args())["normal"], 0.80)

    def test_compliance_includes_wrong_response_and_omission(self):
        model = ChannelModel(config())
        p = model.likelihoods("CHANNEL_PROBE", "0", "0", compliance_id="imperfect", **args())
        n = model.likelihoods("CHANNEL_PROBE", "0", NO_RESPONSE, compliance_id="imperfect", **args())
        self.assertAlmostEqual(p["normal"], 0.7675); self.assertAlmostEqual(p["flip"], 0.0875)
        self.assertAlmostEqual(n["normal"], 0.145)

    def test_unknown_compliance_produces_bounds_not_fictional_point(self):
        model = ChannelModel(config())
        bounds = model.probe_bounds("0", "0", compliance_id="unknown", **args())
        self.assertEqual(bounds["normal"], (0.0, 0.9)); self.assertEqual(bounds["flip"], (0.0, 0.9))
        with self.assertRaisesRegex(ProtocolError, "unknown_compliance_use_bounds"):
            model.likelihoods("CHANNEL_PROBE", "0", "0", compliance_id="unknown", **args())
        with self.assertRaisesRegex(ProtocolError, "unknown_compliance_model"):
            model.likelihoods("CHANNEL_PROBE", "0", "0", **args())

    def test_probe_can_be_uninformative_despite_accurate_decoder(self):
        model = ChannelModel(config())
        value = probe_discrimination(model, prompt_symbol="0", mode_pair=["normal", "flip"], compliance_id="uninformative", **args())
        self.assertAlmostEqual(value["total_variation"], 0)
        useful = probe_discrimination(model, prompt_symbol="0", mode_pair=["normal", "flip"], compliance_id="imperfect", **args())
        self.assertAlmostEqual(useful["total_variation"], 0.68)

    def test_simulation_never_accepts_live_or_missing_stability_assumption(self):
        model = ChannelModel(config())
        with self.assertRaisesRegex(ProtocolError, "cannot_run_live"):
            model.likelihoods("TASK_QUERY", "0", "1", execution_mode="live", **args())
        a = args(); a["assumption"] = None
        with self.assertRaisesRegex(ProtocolError, "window_stability_not_established"):
            model.likelihoods("TASK_QUERY", "0", "1", **a)

    def test_each_applicability_dimension_is_enforced_no_interpolation(self):
        model = ChannelModel(config())
        for key, value in [("mapping_id", "new"), ("stimulus_version", "new"), ("layout_id", "new"), ("sample_rate_hz", 1000)]:
            a = args(); a["scope"][key] = value
            with self.subTest(key=key):
                with self.assertRaisesRegex(ProtocolError, "outside_calibration_scope"): model.likelihoods("TASK_QUERY", "0", "1", **a)
        a = args(); a["duration_s"] = 1.5
        with self.assertRaisesRegex(ProtocolError, "no_interpolation"): model.likelihoods("TASK_QUERY", "0", "1", **a)

    def test_uncalibrated_real_template_and_fake_calibrated_metadata_are_blocked(self):
        with self.assertRaisesRegex(ProtocolError, "real_calibration_not_available"):
            ChannelModel.load(ROOT / "configs/channels/real_ssvep_uncalibrated.template.json")
        v = config(); v["status"] = "calibrated"
        with self.assertRaisesRegex(ProtocolError, "real_calibration_not_available"): ChannelModel(v)

    def test_invalid_rows_nan_and_private_fields_rejected(self):
        for value in [float("nan"), -0.1, True, 2.0]:
            v = config(); v["matrices"]["TASK_QUERY"]["normal"]["1.0"]["0"]["0"] = value
            with self.subTest(value=value):
                with self.assertRaises(ProtocolError): ChannelModel(v)
        v = config(); v["true_current_mode"] = "normal"
        with self.assertRaisesRegex(ProtocolError, "unsupported_field"): ChannelModel(v)

    def test_raw_scores_are_not_a_likelihood_and_config_is_immutable(self):
        v = config(); model = ChannelModel(v); v["matrices"].clear()
        self.assertAlmostEqual(model.likelihoods("TASK_QUERY", "0", "0", **args())["normal"], 0.85)
        with self.assertRaisesRegex(ProtocolError, "categorical_output_required"):
            model.likelihoods("TASK_QUERY", "0", {"0": 0.9}, **args())
        with self.assertRaises(TypeError): model.data["status"] = "calibrated"

    def test_step05_bridge_preserves_probe_prompt_and_does_not_change_task(self):
        model = ChannelModel(config()); result = accepted("CHANNEL_PROBE")
        before = result.request.to_dict()
        p = model.accepted_result_likelihoods(result, intended_symbol="0", layout_id="fixture-two-markers", compliance_id="imperfect", assumption="state_constant_within_window")
        self.assertAlmostEqual(p["normal"], 0.0875)
        self.assertEqual(result.request.to_dict(), before)
        with self.assertRaisesRegex(ProtocolError, "probe_prompt_mismatch"):
            model.accepted_result_likelihoods(result, intended_symbol="1", layout_id="fixture-two-markers", compliance_id="imperfect", assumption="state_constant_within_window")

    def test_step05_bridge_refuses_score_only_or_invalid_signal(self):
        model = ChannelModel(config())
        for result, error in [(accepted(symbol=None), "score_likelihood_not_calibrated"), (accepted(status="INVALID_SIGNAL", symbol=None), "status_not_in_likelihood_model")]:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ProtocolError, error): model.accepted_result_likelihoods(result, intended_symbol="0", layout_id="fixture-two-markers", assumption="state_constant_within_window")

    def test_explicit_no_response_is_not_timeout_or_selection(self):
        model = ChannelModel(config())
        result = accepted(status="NO_RESPONSE", symbol=None, raw_scores=None, score_kind=None, calibration_id=None, sample_window=None)
        with self.assertRaisesRegex(ProtocolError, "nonresponse_duration_required"):
            model.accepted_result_likelihoods(result, intended_symbol="0", layout_id="fixture-two-markers", assumption="state_constant_within_window")
        p = model.accepted_result_likelihoods(result, intended_symbol="0", layout_id="fixture-two-markers", nonresponse_duration_s=1.0, assumption="state_constant_within_window")
        self.assertEqual(p, {"normal": 0.05, "flip": 0.05})


class TransferAndTimeTests(unittest.TestCase):
    def test_holdout_transfer_error_and_missing_counts_remain_estimates(self):
        v = transfer_error({"normal|0": {"0": 0.9, "1": 0.1}}, {"normal|0": {"0": 8, "1": 2}}, provenance="synthetic_fixture", source_ref="fixture-only", split="calibration_holdout")
        self.assertAlmostEqual(v["weighted_empirical_total_variation"], 0.1)
        self.assertEqual(v["status"], "simulation_only"); self.assertIsNone(v["acceptance_threshold"])
        v = transfer_error({"normal|0": {"0": 0.9, "1": 0.1}}, {"normal|0": {"0": 0, "1": 0}}, provenance="synthetic_fixture", source_ref="fixture-only", split="calibration_holdout")
        self.assertIsNone(v["weighted_empirical_total_variation"])
        with self.assertRaisesRegex(ProtocolError, "holdout_required"):
            transfer_error({}, {}, provenance="independent_calibration", source_ref="id", split="test")

    def test_final_prefix_only_and_repeated_same_prefix_not_multiplied(self):
        ledger = FinalPrefixLedger(); result = accepted()
        with self.assertRaisesRegex(ProtocolError, "intermediate_prefix_not_evidence"): ledger.claim(result, finalized=False)
        token = ledger.claim(result, finalized=True)
        self.assertEqual((token.sample_start, token.sample_end), (250, 500))
        with self.assertRaisesRegex(ProtocolError, "prefix_already_consumed"): ledger.claim(result, finalized=True)
        other = accepted("CONFIRM")
        with self.assertRaisesRegex(ProtocolError, "overlapping_prefix_evidence"): ledger.claim(other, finalized=True)

    def test_timing_counts_elapsed_once_and_union_busy_intervals(self):
        segments = [{'phase':'reading','clock_id':'local','start_s':0,'end_s':2,'user_busy':True}, {'phase':'switching','clock_id':'local','start_s':1,'end_s':3,'user_busy':True}, {'phase':'acquisition','clock_id':'local','start_s':3,'end_s':5,'user_busy':True}, {'phase':'decoding','clock_id':'local','start_s':4,'end_s':6,'user_busy':False}, {'phase':'return_to_task','clock_id':'local','start_s':6,'end_s':7,'user_busy':True}, {'phase':'confirmation','clock_id':'local','start_s':7,'end_s':8,'user_busy':True}]
        v = summarize_timing(segments, required_phases=[s['phase'] for s in segments], clock_id='local', window_start_s=0, window_end_s=8)
        self.assertEqual(v['total_elapsed_s'], 8); self.assertEqual(v['user_busy_s'], 7)
        self.assertEqual(sum(v['phase_seconds'].values()), 10)
        v = summarize_timing(segments, required_phases=['retry'], clock_id='local', window_start_s=0, window_end_s=8)
        self.assertFalse(v['decomposition_complete']); self.assertIsNone(v['user_busy_s'])
        segments[0]['clock_id']='remote-monotonic'
        with self.assertRaisesRegex(ProtocolError, 'incompatible_clock'): summarize_timing(segments, required_phases=[], clock_id='local', window_start_s=0, window_end_s=8)


class TransitionTests(unittest.TestCase):
    def model(self):
        return JointTransitionModel(json.loads((ROOT / 'configs/channels/simulation_transition_v1.json').read_text(encoding='utf-8')))

    def test_zero_time_identity_stochasticity_and_simultaneous_changes(self):
        m=self.model(); t=m.matrix(0)
        self.assertEqual(t, [[float(i==j) for j in range(4)] for i in range(4)])
        t=m.matrix(2)
        self.assertTrue(t[0][1]>0 and t[0][2]>0 and t[0][3]>0)
        for row in t: self.assertAlmostEqual(sum(row),1)
        self.assertGreater(m.any_change_probability(2),m.any_change_probability(1))
        with self.assertRaisesRegex(ProtocolError,'cannot_run_live'): m.matrix(1,execution_mode='live')

    def test_elapsed_time_semigroup_not_one_transition_per_question(self):
        m=self.model(); a,b,c=m.matrix(1),m.matrix(2),m.matrix(3)
        for i in range(4):
            for j in range(4): self.assertAlmostEqual(sum(a[i][k]*b[k][j] for k in range(4)),c[i][j])
        self.assertNotEqual(a,b)
        with self.assertRaises(ProtocolError): m.matrix(-1)


class ModelImportTests(unittest.TestCase):
    def test_models_import_without_network_gpu_or_eeg_libraries(self):
        script='''
import builtins,socket
old=builtins.__import__
def guard(n,*a,**k):
 if n.split('.')[0] in {'torch','jax','numpy','scipy','mne','pygame','requests','paramiko','cv2'}: raise AssertionError('external runtime import')
 return old(n,*a,**k)
builtins.__import__=guard
socket.socket=lambda *a,**k: (_ for _ in ()).throw(AssertionError('network forbidden'))
import bci_vla.models.channel,bci_vla.models.probe_transfer,bci_vla.models.evidence
'''
        v=subprocess.run([sys.executable,'-B','-c',script],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(v.returncode,0,v.stderr)
