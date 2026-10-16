"""Delivery checks, not a real-BCI calibration acceptance test."""
from pathlib import Path
import hashlib
import json
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from bci_vla.models.channel import ChannelModel, JointTransitionModel
from bci_vla.protocols import ProtocolError

def load(p): return json.loads((root / p).read_text(encoding="utf-8-sig"))
checks = []
def check(name, condition):
    if not condition: raise AssertionError(name)
    checks.append(name)

p = load("execution/progress.json")
t = load("execution/06_test_result.json")
real = load("configs/channels/real_ssvep_uncalibrated.template.json")
simulation = load("configs/channels/simulation_binary_v1.json")
check("01-06 remain complete", all(s["status"] == "已完成" for s in p["steps"][:6]))
check("later progress preserved", int(p["current_step"]) >= 7)
check("step06 limited to offline branch", p["steps"][5]["execution_scope"] == "offline_simulation_branch" and p["steps"][5]["real_calibration"]["status"] == "待外部条件")
check("G1 scope unchanged and G2-G4 not started", p["research_gates"]["G1"]["scope"] == "conditional_theory_and_protocol_design" and all(p["research_gates"][g]["status"] == "未开始" for g in ["G2", "G3", "G4"]))
check("47 deterministic software checks pass", t["result"] == "PASS" and t["tests_run"] == 47 and t["failures"] == t["errors"] == t["skipped"] == 0)
check("real profile parameters absent and blocked", real["status"] == "uncalibrated" and not real["deployment_allowed"] and real["matrices"] is None and real["scope"] is None)
try: ChannelModel(real)
except ProtocolError as e: check("loader refuses uncalibrated real profile", str(e) == "real_calibration_not_available")
else: raise AssertionError("real profile admitted")
check("simulation provenance explicit", simulation["status"] == "simulation_only" and simulation["provenance"] == "illustrative_not_fitted" and simulation["deployment_allowed"] is False)
ChannelModel(simulation); JointTransitionModel(load("configs/channels/simulation_transition_v1.json"))
check("model configurations validate", True)
acceptance = (root / "execution/06_acceptance.txt").read_text(encoding="utf-8-sig")
check("06.1-06.9 mapped", all(f"06.{i}" in acceptance for i in range(1, 10)))
check("input/software fingerprints match", all(hashlib.sha256(Path(f["path"]).read_bytes()).hexdigest() == f["sha256"] for f in load("execution/06_source_manifest.json")["files"]))
check("TXT encoding intact", all(q.read_bytes().startswith(b"\xef\xbb\xbf") and b"\n" not in q.read_bytes().replace(b"\r\n", b"") for q in root.rglob("*.txt")))
v = {"scope": "offline_model_delivery_integrity", "result": "PASS", "count": len(checks), "checks": checks, "real_calibration_status": "待外部条件", "not_evaluated": ["real EEG likelihoods", "real probe compliance or transfer", "sampling pipeline replay", "real user timing", "G2-G4"]}
text = json.dumps(v, ensure_ascii=False, indent=2) + "\n"
output_name="06_integrity_result.json" if int(p["current_step"])==7 else "06_integrity_recheck_after_step07.json"
(root / "execution" / output_name).write_text(text, encoding="utf-8")
print(text)
