"""Validate inventory consistency, not model or robot performance."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
def load(p):
    return json.loads((root / p).read_text(encoding="utf-8-sig"))

checks = []
def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)

p = load("execution/progress.json")
m = load("configs/baseline_manifest.json")
local = load("execution/02_local_inventory.json")
remote = load("execution/02_remote_inspection.json")
assets = load("execution/02_remote_candidates.json")
software = load("execution/02_software_check.json")
check("01 and 02 remain complete", all(s["status"] == "已完成" for s in p["steps"][:2]))
check("progress preserves later execution", int(p["current_step"]) >= 3)
check("research gates recorded separately", set(p["research_gates"]) == {"G1", "G2", "G3", "G4"})
check("local frozen files unchanged", all(hashlib.sha256(Path(f["path"]).read_bytes()).hexdigest() == f["sha256"] for f in local["files"]))
check("prototype Python AST passed", sum(f.get("syntax") == "PASS" for f in local["files"]) == 18)
check("remote read-only success and matching commit", remote["status"] == "read_only_success" and m["server"]["commit"] == remote["evidence"]["head"]["stdout"])
check("remote before and after clean", remote["evidence"]["status"]["stdout"] == assets["final_git_status"] == "")
check("unknown deployment remains explicit", m["server"]["selected_paper_checkpoint"] is None and m["server"]["actual_launch_command"] is None and not m["server"]["runtime_and_hardware_verified"])
check("software outcome not overclaimed", software["tests_run"] == software["passed"] + software["skipped"] == 53 and software["skipped"] == 9)
check("all baselines labelled", all(b["status"] and b["unresolved"] for b in m["baselines"]))
check("baseline source evidence exists", all((root / s).is_file() for s in m["source_evidence"]))
issues = (root / "docs/02_issues.txt").read_text(encoding="utf-8-sig")
check("14 tracked issues", all(f"I{i:02}｜" in issues for i in range(1, 15)))
check("02.1-02.8 acceptance", all(f"02.{i}" in (root / "execution/02_acceptance.txt").read_text(encoding="utf-8-sig") for i in range(1, 9)))
check("TXT encoding", all(q.read_bytes().startswith(b"\xef\xbb\xbf") and b"\n" not in q.read_bytes().replace(b"\r\n", b"") for q in root.rglob("*.txt")))
result = {"scope": "inventory_and_documentation", "result": "PASS", "count": len(checks), "checks": checks, "not_evaluated": ["weight integrity", "live services", "EEG replay", "robot capability", "G1-G4"]}
out = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
output_name = "02_integrity_result.json" if int(p["current_step"]) == 3 else "02_integrity_recheck_after_step03.json"
(root / "execution" / output_name).write_text(out, encoding="utf-8")
print(out)
