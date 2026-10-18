"""Finalize V4-02 only after its reproducible local checks pass."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from bci_vla.interaction_contract_v4 import load_json
from verify_step02 import main as verify


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def write_txt(path,value):
    path.write_bytes(("\ufeff"+value.replace("\r\n","\n").replace("\n","\r\n")).encode("utf-8"))


def main():
    if verify():return 1
    folder=ROOT/"execution/v4"
    progress_path=ROOT/"execution/v4_plan_progress.json"
    progress=load_json(progress_path)
    previous=load_json(folder/"02_progress_before.json")
    assert progress["historical_v2_gates"]==previous["historical_v2_gates"]
    assert progress["implementation_started"] is False and progress["training_started"] is False
    assert progress["new_experiments_run"] is False
    assert all(s["status"]=="未开始" for s in progress["steps"][2:])
    checks=load_json(folder/"02_verification.json")
    now=datetime.now(timezone.utc).isoformat()
    handoff="""V4-02交接：输入输出与两类澄清协议
日期：2026-10-17（Asia/Hong_Kong）。
已完成：schema、11个公开输入、2个私有标签、4个模块输出夹具、标准库参考校验器及中文协议说明。
复验：在D:\\Scene_Aware\\bci_vla_framework运行 python execution/v4/verify_step02.py。
结果见execution/v4/02_verification.json；本轮128/128项通过，含历史来源文件保护检查，仅为合成协议/软件检查。
协议：D:\\Scene_Aware\\bci_vla_framework\\docs\\v4\\02_protocol.txt。
冻结来源：execution/v4/02_source_manifest.json；验收：execution/v4/02_acceptance.json。
分数按选项ID绑定；PROBE不改任务；NO_RESPONSE保留问题角色；确认绑定精确任务/版本但不授予执行权限。
空输入哨兵、候选之外、候选歧义、缺槽和弃权分别表示。示例前缀为全零占位，不是神经预测。
新Transformer未实现/训练；真实特征维数和π0.5文本/子任务能力未验证；旧G2/G3/G4结论不升级。
下一步V4-03核实主干能力，本步没有执行步骤03、模型加载、EEG或实体动作。
原步骤01交接execution/v4/handoff.txt及其冻结来源未改。不要重跑build_step02.py覆盖接口；有意更新需修改版本、样例、校验和冻结清单。
"""
    assert checks["checks"]==128, "Update handoff count if checks change"
    write_txt(folder/"handoff_step02.txt",handoff)
    sources=[ROOT/"configs/v4/interfaces.json",ROOT/"docs/v4/02_protocol.txt"]
    sources+=sorted((ROOT/"examples/v4").glob("*.json"))
    sources+=sorted((ROOT/"bci_vla/interaction_contract_v4").glob("*.py"))
    sources+=[folder/name for name in ["build_step02.py","build_step02_edgecases.py","verify_step02.py","complete_step02.py","handoff_step02.txt"]]
    manifest={"step":"V4-02","version":"V4-02-contract-1","files":[{"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources]}
    write_json(folder/"02_source_manifest.json",manifest)
    acceptance={"step_id":"V4-02","status":"PASS","scope":"synthetic interface/protocol/software acceptance only","completed_at_utc":now,
        "verification_report":"execution/v4/02_verification.json","checks":checks["checks"],"passed":checks["passed"],"failed":checks["failed"],
        "plan_requirements":{"objects_evidence_time_versions_unknown":True,"current_event_once_confirmation_bound":True,"intent_query_vs_probe_no_response":True,"tensor_shapes_and_masks_specified_not_implemented":True,"public_private_examples":True,"persistent_state_invalidation":True,"channel_estimand_and_supervision_masks":True,"empty_outside_ambiguity_missing_abstain_distinct":True,"alias_binding_and_permutation":True},
        "historical_files_unchanged":True,"source_manifest_files":len(sources),"neural_module_implemented":False,"models_loaded":False,"training_run":False,"real_eeg":False,"physical_robot":False,"research_gates_upgraded":False,"next_step":"V4-03"}
    write_json(folder/"02_acceptance.json",acceptance)
    step=next(s for s in progress["steps"] if s["step_id"]=="V4-02")
    step.update(status="已完成",completed_at_utc=now,actual_artifacts=["configs/v4/interfaces.json","examples/v4/fixture_index.json","docs/v4/02_protocol.txt","bci_vla/interaction_contract_v4/validation.py","execution/v4/02_verification.json","execution/v4/02_source_manifest.json","execution/v4/02_acceptance.json","execution/v4/handoff_step02.txt"],verification={"scope":acceptance["scope"],"status":"PASS","report":"execution/v4/02_verification.json","checks":checks["checks"]})
    progress.update(current_stage="V4-02_protocol_complete",next_step="V4-03",latest_handoff="execution/v4/handoff_step02.txt",review_status="V4-02_contract_verified_no_neural_implementation")
    progress["implementation_started_meaning"]="Neural/model implementation only; contract/reference validation work does not set this flag."
    write_json(progress_path,progress)
    write_txt(ROOT.parent/"BCI_VLA当前计划入口_V4.txt","""当前研究计划：V4-plan-2。
2026-10-17：V4-01、V4-02已完成；下一步V4-03未开始。
主计划：D:\\Scene_Aware\\BCI_VLA框架_意图更新与澄清_执行计划_V4.txt。
最新交接：D:\\Scene_Aware\\bci_vla_framework\\execution\\v4\\handoff_step02.txt。
当前进度：D:\\Scene_Aware\\bci_vla_framework\\execution\\v4_plan_progress.json。
接口说明：D:\\Scene_Aware\\bci_vla_framework\\docs\\v4\\02_protocol.txt。
步骤02验收：D:\\Scene_Aware\\bci_vla_framework\\execution\\v4\\02_acceptance.json。
本轮128/128项合成协议/软件检查通过，新Transformer未实现/训练，高层文本与子任务能力未验证；真实EEG与实体动作排除。
计划正文的初始状态是制定时快照，当前实施状态以进度及最新交接为准。
""")
    for row in manifest["files"]:
        assert hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()==row["sha256"]
    for path in [ROOT/"docs/v4/02_protocol.txt",folder/"handoff_step02.txt",ROOT.parent/"BCI_VLA当前计划入口_V4.txt"]:
        raw=path.read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf") and b"\n" not in raw.replace(b"\r\n",b"")
    print(json.dumps({"step":"V4-02","status":"PASS","checks":checks["checks"],"frozen_files":len(sources),"next_step":"V4-03","neural_implemented":False}))
    return 0


if __name__=="__main__":raise SystemExit(main())
