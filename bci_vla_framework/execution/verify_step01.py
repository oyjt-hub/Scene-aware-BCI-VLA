"""Check documentation integrity only; no model, EEG, or robot evaluation."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
read = lambda p: (root / p).read_text(encoding="utf-8-sig")
claim = read("docs/01_claim_map.txt")
scope = read("docs/01_scope.txt")
progress = json.loads(read("execution/progress.json"))
sources = json.loads(read("execution/source_manifest.json"))
checks = []

def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)

check("17 unique ordered steps", [s["step_id"] for s in progress["steps"]] == [f"{i:02}" for i in range(1, 18)])
check("step 01 remains complete", progress["steps"][0]["status"] == "已完成")
check("progress preserves later execution", int(progress["current_step"]) >= 2)
check("research gates recorded separately", set(progress["research_gates"]) == {"G1", "G2", "G3", "G4"})
check("dependency order", all(d in [f"{i:02}" for i in range(1, int(s["step_id"]))] for s in progress["steps"] for d in s["depends_on"]))
check("10 nearest-neighbor rows", all(f"R{i:02}｜" in claim for i in range(1, 11)))
check("mandatory related work", all(t in claim for t in ["RBC", "POMDP-BCI", "动态停时", "任务VoI", "自重校准", "BAL", "ActivAsk", "NOIR 2.0", "直接BCI—VLA"]))
check("claims and nonclaims", all(t in claim for t in ["C1", "C2", "C3", "不主张首次联合估计意图/通道", "首次主动校准", "首次BCI接VLA"]))
check("scope and gate criteria", all(t in scope for t in ["SSVEP", "固定VLA", "有限离散语义目标", "少量有限", "G1", "G2", "G3", "G4", "失败", "处理"]))
check("fair baselines and oracle", all(t in claim for t in ["B4", "B5", "B8", "B9", "B10", "O1", "O2"]))
required = ["README.txt", "docs/01_claim_map.txt", "docs/01_scope.txt", "execution/progress.json", "execution/source_manifest.json", "execution/worklog.txt", "execution/handoff.txt", "execution/01_acceptance.txt"]
check("required records readable", all((root / p).is_file() and "\ufffd" not in read(p) for p in required))
check("TXT UTF8 BOM and CRLF", all((root / p).read_bytes().startswith(b"\xef\xbb\xbf") and b"\n" not in (root / p).read_bytes().replace(b"\r\n", b"") for p in required if p.endswith(".txt")))
import hashlib
check("source files unchanged", all(hashlib.sha256(Path(s["path"]).read_bytes()).hexdigest() == s["sha256"] for s in sources["sources"]))
print(json.dumps({"scope": "documentation", "result": "PASS", "count": len(checks), "checks": checks, "not_evaluated": ["G1-G4", "algorithm", "EEG", "robot", "innovation validity"]}, ensure_ascii=False, indent=2))
