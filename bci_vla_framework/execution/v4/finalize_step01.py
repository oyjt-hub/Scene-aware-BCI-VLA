"""Assemble V4-01 inventory and documentation acceptance from captured evidence.

No network, model import, model load, dependency install, or experiment is run.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


BASE = Path(__file__).resolve().parent
PROJECT = BASE.parents[1]
ROOT = PROJECT.parent


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def txt(path: Path, value: str) -> None:
    path.write_bytes(b"\xef\xbb\xbf" + value.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))


def main() -> None:
    if (BASE / "01_inventory.json").exists():
        raise SystemExit("Preserve completed inventory; do not overwrite a prior delivery.")
    now = datetime.now(timezone.utc).isoformat()
    remote = read(BASE / "01_remote_inventory.json")
    supplement = read(BASE / "01_remote_supplement.json")
    paths = read(BASE / "01_remote_paths.json")
    workspace = read(BASE / "01_workspace_weights.json")
    candidates = read(BASE / "01_candidate_weights.json")
    local = read(BASE / "01_local_sources.json")
    environment = read(BASE / "01_local_environment.json")
    before = read(BASE / "01_before_manifest.json")
    historical = read(PROJECT / "execution/15C_current_assets.json")
    progress_path = PROJECT / "execution/v4_plan_progress.json"
    progress = read(progress_path)
    assert progress["plan_version"] == "V4-plan-2"
    assert progress["steps"][0]["status"] == "进行中"
    assert all(s["status"] == "未开始" for s in progress["steps"][1:])

    all_path_rows = remote["paths"] + supplement["paths"] + paths["frontend_path_checks"] + workspace["weight_and_config_matches"] + candidates["paths"]
    by_path = {r["path"]: r for r in all_path_rows}
    sources = {r["path"]: r for r in remote["source_files"] + supplement["frontend_source_files"] + candidates["small_sources"]}
    old_hashes = {r["path"]: r["sha256"] for r in historical["files"]}
    weight = "/home/oyjt/workspace/RLinf-RECAP/models/baseline_sft/RLinf-Pi05-LIBERO-SFT/model.safetensors"
    star = "/home/oyjt/workspace/Qwen2.5-VL-GR00T-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt"
    florence = "/home/oyjt/Grounded-SAM-2/Florence-2-large/model.safetensors"
    sam = "/home/oyjt/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
    assets = [
        {"id": "A01", "name": "original server repository", "path": "/home/oyjt/Scene-aware-BCI-VLA", "status": "current_git_and_source_inventory_verified", "evidence": remote["git"]["/home/oyjt/Scene-aware-BCI-VLA"]},
        {"id": "A02", "name": "existing modified OpenPI source", "path": "/home/oyjt/openpi", "status": "present_with_existing_changes_preserve_and_isolate", "evidence": remote["git"]["/home/oyjt/openpi"]},
        {"id": "A03", "name": "historical pi05 LIBERO reference", "path": weight, "status": "file_and_size_present_current_full_hash_unverified", "metadata": by_path[weight], "sha256_current": None, "sha256_historical": old_hashes[weight], "historical_hash_evidence": "execution/15C_current_assets.json", "new_v4_model_selected": False},
        {"id": "A04", "name": "base/droid/fold/uniform pi05 candidates", "status": "directories_located_not_loaded_or_selected", "paths": [r for r in candidates["paths"] if "/workspace/checkpoints/" in r["path"]], "full_shard_integrity_verified": False},
        {"id": "A05", "name": "Paligemma tokenizer", "status": "file_and_current_sha256_verified_decode_capability_pending", **paths["tokenizer"]},
        {"id": "A06", "name": "Florence visual weights", "path": florence, "status": "actual_file_located_elsewhere_frontend_binding_unverified", "metadata": by_path[florence], "sha256_current": None, "former_repository_relative_path_missing": True},
        {"id": "A07", "name": "SAM2.1 visual weights", "path": sam, "status": "actual_file_located_elsewhere_frontend_binding_unverified", "metadata": by_path[sam], "sha256_current": None, "former_repository_relative_path_missing": True},
        {"id": "A08", "name": "LIBERO environments and historical adapters", "status": "paths_and_distribution_metadata_present_no_import_or_rollout_this_step", "paths": [r for r in all_path_rows if r["path"] in ("/home/oyjt/LIBERO", "/home/oyjt/workspace/miniconda3/envs/libero", "/home/oyjt/openpi/.venv")], "environments": remote["environments"]},
        {"id": "A09", "name": "historical Qwen GR00T reference", "path": star, "status": "file_and_size_present_current_hash_unverified", "metadata": by_path[star], "sha256_current": None, "sha256_historical": old_hashes[star]},
        {"id": "A10", "name": "early context_intent prototype", "path": str(ROOT / "context_intent"), "status": "source_only_untrained_not_v4_internal_connector", "trained_weight_files": local["legacy_context_intent_weight_files"]},
        {"id": "A11", "name": "V2 interaction and authorization code", "status": "local_source_available_historical_research_gates_unchanged", "source_files": [r for r in local["files"] if any(k in r["path"].replace("\\", "/") for k in ("/core/", "/contracts/", "/adapters/"))]},
        {"id": "A12", "name": "new neural interaction V4", "status": "not_implemented_not_trained", "directory_exists": local["v4_neural_module_directory_exists"], "high_level_text_verified": False, "semantic_subtask_verified": False, "prefix_injection_verified": False},
        {"id": "A13", "name": "current local inspection interpreter", "status": "metadata_only_not_a_verified_training_environment", "evidence": environment},
    ]

    literature_rows = [
        ("L01", "BLIP-2", "https://arxiv.org/abs/2301.12597", "abstract", "查询网络桥接冻结视觉与语言主干已有先例", "普通查询连接器必须作为对照"),
        ("L02", "Perceiver", "https://arxiv.org/abs/2103.03206", "abstract", "固定潜在瓶颈压缩可变输入已有先例", "K个query本身不作为原创"),
        ("L03", "Robot Confirmation Generation and Action Planning Using Long-context Q-Former Integrated with Multimodal LLM", "https://arxiv.org/html/2511.17335v1", "method HTML", "上下文查询token生成确认描述和动作步骤，结构/功能接近", "验证交互来源处理与在线含噪意图更新的额外作用"),
        ("L04", "NOIR", "https://arxiv.org/abs/2311.01454", "abstract", "脑机目标对象和动作控制机器人已有先例", "不能以脑机机器人链路本身主张原创"),
        ("L05", "NOIR 2.0", "https://arxiv.org/html/2511.20848v1", "method HTML", "基础模型辅助预测对象/技能与减少交互", "单纯场景任务建议不足以区分"),
        ("L06", "Gaze2Act", "https://zuo-kuangji.github.io/Gaze2Act/", "abstract and author project", "人类目标与位置线索条件化VLA并支持动态交互", "不能仅把注视换成脑机作为方法创新"),
        ("L07", "pi0.5", "https://arxiv.org/abs/2504.16054", "abstract refreshed; method checked in previous turn", "语义子任务和动作专家是原框架机制", "真实检查点能力另由V4-03核实"),
        ("L08", "When to Act, Ask, or Learn: Uncertainty-Aware Policy Steering", "https://arxiv.org/abs/2602.22474", "v2 abstract", "语义歧义和动作能力影响执行/询问/干预", "不确定时询问已有先例"),
        ("L09", "MIGU: Multimodal Instruction Grounding under Uncertainty for Manipulation Planning", "https://multimodal-instruction.github.io/", "author project only", "不确定语言/指向证据融合及执行/澄清选择", "补充证据融合近邻，全文尚未审计"),
        ("L10", "Bayesian Active Learning for Intent Disambiguation in Interactive Robot Planning", "https://arxiv.org/abs/2609.34270", "abstract", "任务候选上的主动澄清已有先例", "信息增益选问题不单列原创"),
        ("L11", "Knowing When to Ask: Resolving Uncertainty in Human-Robot Joint Planning via Explicit Dialogue and Implicit Intent Cues", "https://arxiv.org/abs/2603.07822", "v2 abstract", "显式对话与隐式线索用于不确定人机规划", "需验证通道诊断与内部token的实质收益"),
    ]
    literature = {"checked_at_utc": now, "scope": "targeted primary-source comparison, not exhaustive novelty certification", "local_implementations_of_papers_claimed": False,
                  "papers": [{"id": a, "title": b, "url": d, "reading_depth": e, "observed_overlap": f, "research_implication": g} for a,b,d,e,f,g in literature_rows]}
    save(BASE / "01_literature.json", literature)
    gaps = [
        {"id": "I01", "gap": "high-level text and semantic subtask capability unverified", "owner_step": "V4-03", "blocks_schema_step02": False},
        {"id": "I02", "gap": "new connector, object binding and gradient path not implemented", "owner_step": "V4-02/06/07"},
        {"id": "I03", "gap": "large current checkpoint integrity not reverified; candidate shards not audited", "owner_step": "V4-03"},
        {"id": "I04", "gap": "actual Florence/SAM weights located; frontend path/runtime binding pending", "owner_step": "V4-12"},
        {"id": "I05", "gap": "multiple source copies and existing OpenPI changes require isolation", "owner_step": "V4-03"},
        {"id": "I06", "gap": "local default interpreter lacks torch metadata; runtime compatibility pending", "owner_step": "V4-06"},
        {"id": "I07", "gap": "new visual interaction data, subtask labels and formal splits not built", "owner_step": "V4-04"},
        {"id": "I08", "gap": "diagnostic conditions, masks and risk rules require protocol freeze", "owner_step": "V4-02/09/10"},
        {"id": "I09", "gap": "compute costs and reservations not verified by pilot", "owner_step": "V4-03/08/12"},
        {"id": "I10", "gap": "method advantage and novelty not established", "owner_step": "V4-13"},
    ]
    inventory = {"step_id": "V4-01", "plan_version": "V4-plan-2", "completed_at_utc": now,
        "scope": "read-only inventory, research scope and nearest-neighbor comparison only",
        "assets": assets, "remote_source_hashes": [{"path": k, "sha256": v.get("sha256"), "exists": v["exists"]} for k,v in sources.items()],
        "gpu_snapshot": {"observed_at_utc": remote["checked_at_utc"], **remote["gpu"], "resource_reserved": False, "hardware_memory_capacity_independently_verified": False},
        "capability_claims": {"new_neural_network_implemented": False, "trained_v4_weights_available": False, "pi05_high_level_generation_verified": False, "subtask_planning_verified": False, "current_full_pi05_weight_hash_verified": False, "actual_visual_weight_files_located": True, "real_eeg_calibrated": False},
        "historical_gates": {"G2": "未通过", "G3": "未验证", "G4": "未通过"},
        "gaps": gaps, "next_step": "V4-02", "new_v4_model_selected": False,
        "credentials_recorded": False, "model_loads": 0, "service_launches": 0, "training_runs": 0, "EEG_connections": 0, "robot_connections": 0,
        "raw_evidence": ["01_git_before.json", "01_remote_inventory.json", "01_remote_supplement.json", "01_remote_paths.json", "01_workspace_weights.json", "01_candidate_weights.json", "01_local_environment.json", "01_local_sources.json", "01_literature.json", "01_command_log.json"]}
    save(BASE / "01_inventory.json", inventory)

    command_log = read(BASE / "01_command_log.json")
    command_log["attempts"] += [
        {"command": "inspect_assets_step01.py --workspace (initial recursive version)", "result": "SSH read timed out; no result used; scan narrowed before successful retry", "artifact": None},
        {"command": "inspect_assets_step01.py --workspace (bounded top-level version)", "result": "succeeded, 162 names plus exact known weight metadata", "artifact": "01_workspace_weights.json"},
        {"command": "inspect_assets_step01.py --candidates", "result": "succeeded; base/droid/fold/uniform and actual Florence/SAM files located", "artifact": "01_candidate_weights.json"},
    ]
    save(BASE / "01_command_log.json", command_log)
    report = PROJECT / "docs/v4/01_scope_and_overlap.txt"
    txt(report, report.read_text(encoding="utf-8-sig"))
    txt(BASE / "handoff.txt", "V4-01已完成：2026-10-17，资产/范围/近邻对照已交付。\n"
        "先读execution/v4/01_inventory.json及docs/v4/01_scope_and_overlap.txt。\n"
        "下一步V4-02，未开始。只冻结公开/私有schema、角色、OBJ_i、窗口外状态和诊断估计量。\n"
        "权重已定位：workspace/checkpoints有base/droid/fold/uniform；LIBERO SFT和Qwen-GR00T存在。\n"
        "Florence/SAM实际权重在/home/oyjt/Grounded-SAM-2，原仓库相对路径不能直接代表它们。\n"
        "本轮大权重完整SHA256未复核，高层文本/语义子任务未验证，新模块未实现/训练。\n"
        "V2/V3结果及G2/G3/G4保留。计划正文中的初始状态是制定时快照，当前进度以execution/v4_plan_progress.json为准。\n")

    immutable = [Path(r["path"]) for r in before["files"] if Path(r["path"]).resolve() != progress_path.resolve()]
    before_hash = {str(Path(r["path"]).resolve()): r["sha256"] for r in before["files"]}
    changed = [str(f) for f in immutable if hashlib.sha256(f.read_bytes()).hexdigest() != before_hash[str(f.resolve())]]
    checks = {
        "authorized_step_only": inventory["step_id"] == "V4-01" and all(s["status"] == "未开始" for s in progress["steps"][1:]),
        "required_scope_report_exists": report.is_file() and all(s in report.read_text(encoding="utf-8-sig") for s in ("候选贡献", "否证", "G2未通过", "V4-02")),
        "primary_sources_registered": len(literature["papers"]) == 11 and all(r["url"].startswith("https://") and r["reading_depth"] for r in literature["papers"]),
        "remote_inspection_success_evidence": bool(remote["source_files"]) and bool(remote["git"]),
        "pi05_file_present_at_workspace_path": by_path[weight]["exists"] and by_path[weight]["bytes"] == 7473091464,
        "base_and_droid_params_present": all(by_path[f"/home/oyjt/workspace/checkpoints/{name}/params"]["exists"] for name in ("pi05_base", "pi05_droid")),
        "visual_weights_located_not_declared_missing": by_path[florence]["exists"] and by_path[sam]["exists"] and inventory["capability_claims"]["actual_visual_weight_files_located"],
        "current_vs_historical_hash_separated": assets[2]["sha256_current"] is None and len(assets[2]["sha256_historical"]) == 64 and not inventory["capability_claims"]["current_full_pi05_weight_hash_verified"],
        "tokenizer_current_hash_recorded": len(paths["tokenizer"]["sha256"]) == 64,
        "no_v4_network_claim": not inventory["capability_claims"]["new_neural_network_implemented"] and not local["v4_neural_module_directory_exists"],
        "no_model_or_device_operations": all(inventory[k] == 0 for k in ("model_loads", "service_launches", "training_runs", "EEG_connections", "robot_connections")),
        "high_level_gate_not_upgraded": not inventory["capability_claims"]["pi05_high_level_generation_verified"] and not inventory["capability_claims"]["subtask_planning_verified"],
        "historical_source_files_unchanged": not changed,
        "gaps_have_owner_steps": len(gaps) == 10 and all(r["owner_step"].startswith("V4-") for r in gaps),
        "read_only_inventory_utility_parses": isinstance(ast.parse((BASE / "inspect_assets_step01.py").read_text(encoding="utf-8")), ast.Module),
        "utf8_bom_crlf_reports": all(f.read_bytes().startswith(b"\xef\xbb\xbf") and b"\n" not in f.read_bytes().replace(b"\r\n", b"") for f in (report, BASE / "handoff.txt")),
    }
    assert all(checks.values()), checks

    manifest_paths = [BASE / name for name in inventory["raw_evidence"]] + [BASE / "01_inventory.json", report, BASE / "handoff.txt", BASE / "inspect_assets_step01.py", Path(__file__)]
    save(BASE / "01_source_manifest.json", {"scope": "V4-01 captured evidence and delivery snapshot; mutable global progress is excluded", "files": [{"path": str(f.resolve()), "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "bytes": f.stat().st_size} for f in manifest_paths]})
    acceptance = {"step_id": "V4-01", "status": "PASS", "scope": "inventory/documentation acceptance only", "completed_at_utc": now, "checks": checks, "historical_files_compared": len(immutable), "historical_files_changed": changed, "full_weight_hash_verified": False, "next_step": "V4-02", "research_gates_upgraded": False}
    save(BASE / "01_acceptance.json", acceptance)
    step = progress["steps"][0]
    step["status"] = "已完成"
    step["completed_at_utc"] = now
    step["actual_artifacts"] = ["execution/v4/01_inventory.json", "docs/v4/01_scope_and_overlap.txt", "execution/v4/01_literature.json", "execution/v4/01_source_manifest.json", "execution/v4/01_acceptance.json", "execution/v4/handoff.txt"]
    step["verification"] = {"scope": "inventory/documentation only", "status": "PASS", "report": "execution/v4/01_acceptance.json", "checks": len(checks)}
    progress.update(current_stage="V4-01_completed", next_step="V4-02", execution_started=True, implementation_started=False, training_started=False, new_experiments_run=False)
    progress["implementation_started_meaning"] = "Neural/model code implementation; inventory-only work does not set this flag."
    progress["authorization_this_turn"] = "Complete V4-01 read-only inventory including user-requested workspace weight-location check; no V4-02 implementation, model loads, training, EEG or robot operations."
    save(progress_path, progress)
    txt(ROOT / "BCI_VLA当前计划入口_V4.txt", "当前研究计划：V4-plan-2。\n"
        "2026-10-17：V4-01资产/范围/近邻对照已完成，下一步V4-02未开始。\n"
        "主计划：D:\\Scene_Aware\\BCI_VLA框架_意图更新与澄清_执行计划_V4.txt。\n"
        "最新交接：D:\\Scene_Aware\\bci_vla_framework\\execution\\v4\\handoff.txt。\n"
        "当前进度：D:\\Scene_Aware\\bci_vla_framework\\execution\\v4_plan_progress.json。\n"
        "盘点报告：D:\\Scene_Aware\\bci_vla_framework\\docs\\v4\\01_scope_and_overlap.txt。\n"
        "本轮仅完成盘点，新网络未实现/训练，高层文本与子任务能力未验证；真实EEG与实体动作继续排除。\n"
        "计划正文的初始状态是制定时快照，当前实施状态以进度及最新交接为准。\n")
    print(json.dumps({"step": "V4-01", "status": "PASS", "assets": len(assets), "primary_sources": len(literature_rows), "checks": len(checks), "historical_files_changed": len(changed), "next_step": "V4-02"}, ensure_ascii=True))


if __name__ == "__main__":
    main()
