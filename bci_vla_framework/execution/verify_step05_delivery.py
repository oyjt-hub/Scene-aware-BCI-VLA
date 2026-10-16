"""Check software delivery against the step-05 inventory; no device execution."""
import ast
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from bci_vla.protocols import Event, ProtocolError

def load(p):
    return json.loads((root / p).read_text(encoding="utf-8-sig"))

checks = []
def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)

p = load("execution/progress.json")
t = load("execution/05_test_result.json")
check("01-05 remain complete", all(s["status"] == "已完成" for s in p["steps"][:5]))
check("later progress and gates unchanged", int(p["current_step"]) >= 6 and p["research_gates"]["G1"]["scope"] == "conditional_theory_and_protocol_design" and all(p["research_gates"][g]["status"] == "未开始" for g in ["G2", "G3", "G4"]))
check("software regressions pass without skips", t["result"] == "PASS" and t["tests_run"] == 28 and t["failures"] == t["errors"] == t["skipped"] == 0)
names = ["task_query_request", "task_query_response", "channel_probe_request", "channel_probe_response", "confirm_request", "confirm_response", "stop", "cancel", "execution_event"]
for name in names:
    Event.parse(load("examples/protocols/" + name + ".json"))
check("nine legal fixtures parse", True)
for case in load("examples/protocols/invalid_cases.json"):
    try:
        Event.parse(case["event"])
    except ProtocolError as exc:
        assert str(exc) == case["expected_error"]
    else:
        raise AssertionError("illegal fixture accepted")
check("eight illegal fixtures rejected", len(load("examples/protocols/invalid_cases.json")) == 8)
source = root / "bci_vla/protocols.py"
symbols = {n.name for n in ast.parse(source.read_text(encoding="utf-8")).body if isinstance(n, ast.ClassDef)}
check("seven domain records declared", {"SceneState", "JointBelief", "ChannelProfile", "ProbeModel", "TaskContract", "ConsequenceModel", "ExecutionEvent"} <= symbols)
check("05.1-05.8 acceptance mapped", all(f"05.{i}" in (root / "execution/05_acceptance.txt").read_text(encoding="utf-8-sig") for i in range(1, 9)))
check("source snapshots unchanged", all(hashlib.sha256(Path(f["path"]).read_bytes()).hexdigest() == f["sha256"] for f in load("execution/05_source_manifest.json")["files"]))
check("TXT encoding maintained", all(q.read_bytes().startswith(b"\xef\xbb\xbf") and b"\n" not in q.read_bytes().replace(b"\r\n", b"") for q in root.rglob("*.txt")))
result = {"scope": "software_delivery_integrity", "result": "PASS", "count": len(checks), "checks": checks, "limitations": ["No live transport or physical stopping validated", "No empirical calibration/EEG/VLA results", "No systemwide covert-label leakage guarantee", "G2-G4 not evaluated"]}
out = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
output_name = "05_integrity_result.json" if int(p["current_step"]) == 6 else "05_integrity_recheck_after_step06.json"
(root / "execution" / output_name).write_text(out, encoding="utf-8")
print(out)
