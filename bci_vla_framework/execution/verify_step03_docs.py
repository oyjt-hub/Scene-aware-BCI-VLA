"""Cross-check step 03 records; mathematical validity also requires independent review."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
def read(p):
    return (root / p).read_text(encoding="utf-8-sig")
def load(p):
    return json.loads(read(p))
checks = []
def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)

progress = load("execution/progress.json")
math = load("execution/03_math_result.json")
doc = read("docs/03_identifiability.txt")
timeline = read("docs/03_protocol_timeline.txt")
review = read("execution/03_review.txt")
acceptance = read("execution/03_acceptance.txt")
examples = load("examples/alias_cases.json")
sources = load("execution/03_source_manifest.json")
check("01-03 remain completed", all(s["status"] == "已完成" for s in progress["steps"][:3]))
check("progress preserves later execution", int(progress["current_step"]) >= 4)
check("G1 design scope explicit", progress["research_gates"]["G1"]["status"] == "已完成" and progress["research_gates"]["G1"]["scope"] == "conditional_theory_and_protocol_design")
check("G2-G4 not started", all(progress["research_gates"][g]["status"] == "未开始" for g in ["G2", "G3", "G4"]))
check("all 9 substeps mapped", all(f"03.{i}" in acceptance for i in range(1, 10)))
check("math checks recorded", math["result"] == "PASS" and math["count"] == 21 and math["scope"] == "analytic_identity_check")
check("independent review provenance retained", "/root/step03_independent_review" in review and "独立审查原文" in review)
check("conditional boundaries in proof", all(t in doc for t in ["分析窗口内g,f固定", "同一策略", "alpha=eta+epsilon_p-2*eta*epsilon_p", "混合权重", "不能未经验证令C_probe=C_task", "K_a", "模型外", "后验赔率"]))
check("protocol source and noisy confirmation", all(t in doc for t in ["prompt_source=display_controller", "presentation_ack", "response_to", "无控制", "重叠EEG前缀", "CONFIRM", "不能定位", "F12"]))
check("timeline branches and private truth", all(t in timeline for t in ["无法判断分支", "任务变更/取消/迟到分支", "不可见变化分支", "评价私有数据"]))
check("examples are private analytical fixtures", examples["visibility"] == "evaluation_private_example_only_not_online_protocol" and len(examples["cases"]) == 13)
check("design and input hashes unchanged since freeze", all(hashlib.sha256(Path(f["path"]).read_bytes()).hexdigest() == f["sha256"] for f in sources["files"]))
check("TXT encoding preserved", all(q.read_bytes().startswith(b"\xef\xbb\xbf") and b"\n" not in q.read_bytes().replace(b"\r\n", b"") for q in root.rglob("*.txt")))
out = json.dumps({"scope": "documentation_and_record_consistency", "result": "PASS", "count": len(checks), "checks": checks, "not_evaluated": ["physiological cause", "empirical probe validity", "novelty sufficiency", "G2-G4"]}, ensure_ascii=False, indent=2) + "\n"
output_name = "03_integrity_result.json" if int(progress["current_step"]) == 4 else "03_integrity_recheck_after_step04.json"
(root / "execution" / output_name).write_text(out, encoding="utf-8")
print(out)
