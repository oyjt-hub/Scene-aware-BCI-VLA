"""Lint design records and leakage boundaries; does not run a benchmark."""
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

private = load("examples/benchmark_v2/private_change_plan.template.json")
public = load("examples/benchmark_v2/public_event.template.json")
outcome = load("examples/benchmark_v2/outcome_record.template.json")
progress = load("execution/progress.json")
protocol = read("docs/04_benchmark_protocol.txt")
fields = read("docs/04_evaluation_fields.txt")
fmt = read("docs/04_change_plan_format.txt")
acceptance = read("execution/04_acceptance.txt")
check("four unique 2x2 design cells", {(c["goal_change_planned"], c["channel_change_planned"]) for c in private["planned_cells"]} == {(False, False), (True, False), (False, True), (True, True)} and len({c["cell_id"] for c in private["planned_cells"]}) == 4)
check("no fabricated sample sizes or schedules", all(c["intended_episode_count"] is None and c["schedule"] is None for c in private["planned_cells"]) and all(v is None for v in private["parameters"].values()))
check("collection blocked", private["ready_for_collection"] is False and private["status"] == "draft_not_executable" and len(private["collection_blockers"]) >= 1)
check("12 stress conditions", [s["id"] for s in private["stress_conditions"]] == [f"S{i:02}" for i in range(1, 13)])
check("public allowlist carries no private plan fields", set(public["event_fields"]).isdisjoint(private["required_private_episode_fields"]) and all(v is None for v in public["event_fields"].values()))
forbidden = {"true_goal", "true_channel", "ground_truth", "factor_cell", "factor_cell_intended", "prescribed_goal", "future_schedule", "injection_seed", "evaluation_seed", "comprehension_annotation"}
def keys(value):
    if isinstance(value, dict):
        return set(value).union(*(keys(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(keys(v) for v in value))
    return set()
check("no private keys anywhere in public template", keys(public).isdisjoint(forbidden))
check("no outcome observations invented", outcome["is_collected_record"] is False and outcome["result_summary"] is None and all(v is None for v in outcome["fields"].values()))
check("all evidence layers separated", all(f"L{i}" in protocol for i in range(1, 6)))
check("truth boundaries explicit", all(t in protocol for t in ["cue_presented并非意图精确切换时刻", "不称生理漂移", "not_exposed", "不使用任务完成、算法后验", "相机", "跨会话", "unknown"]))
check("denominators and missing bounds defined", all(t in fields for t in ["N=S+F+U", "S/N", "(S+U)/N", "(W+U_w)/N", "not_applicable", "不能默认false", "区间并集"]))
check("split and budget fairness documented", all(t in protocol for t in ["同一EEG原始段", "连通分量", "校准预算", "标准一步任务VoI", "测试不用于选择"]))
check("9 substeps mapped", all(f"04.{i}" in acceptance for i in range(1, 10)))
check("01-04 remain complete", all(s["status"] == "已完成" for s in progress["steps"][:4]))
check("later progress and gates preserved", int(progress["current_step"]) >= 5 and progress["research_gates"]["G1"]["scope"] == "conditional_theory_and_protocol_design" and all(progress["research_gates"][g]["status"] == "未开始" for g in ["G2", "G3", "G4"]))
check("input and design hashes match freeze", all(hashlib.sha256(Path(f["path"]).read_bytes()).hexdigest() == f["sha256"] for f in load("execution/04_source_manifest.json")["files"]))
check("all TXT UTF8 BOM CRLF", all(p.read_bytes().startswith(b"\xef\xbb\xbf") and b"\n" not in p.read_bytes().replace(b"\r\n", b"") for p in root.rglob("*.txt")))
result = {"scope": "protocol_design_lint", "result": "PASS", "count": len(checks), "checks": checks, "limitations": ["Template allowlist checks do not prove runtime leakage prevention", "No actual observations, participants or outcomes", "No empirical truth, sampling, statistical power or device validation", "Formal protocol implementation deferred to step 05", "G2-G4 not evaluated"]}
text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
output_name = "04_integrity_result.json" if int(progress["current_step"]) == 5 else "04_integrity_recheck_after_step05.json"
(root / "execution" / output_name).write_text(text, encoding="utf-8")
print(text)
