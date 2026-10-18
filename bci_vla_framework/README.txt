关键语义缺口：Full三个种子的证据切换与别名重绑定均0/8（双侧同时正确）；原声明对象指针均16/16。下一步先修内部token→目标语言绑定，不能进入动作闭环或宣称意图推理已通过。

V4-08最新状态：受限开发训练完成（四方法×三种子及本地权重核验）；步骤08整体未完成。
实际LIBERO＋模拟交互；公开声明教学，128训练/32验证，结果按3seed分别记录。
报告docs/v4/08_training.txt；审计execution/v4/08_training_audit.json；交接execution/v4/handoff_step08.txt。
放置/真实含噪隐藏意图/外部泛化未完成；完整VLA闭环未运行；旧G2/G3/G4不升级。

历史记录（保留原范围）：
2026-10-16 V3设计计划入口：根目录BCI_VLA框架_神经交互模块方案_V3.txt与BCI_VLA通用交互框架_逐步执行手册_V3.txt。
V3为显式双路查询网络设计，尚未实现/训练。新编号V3-01起，进度在execution/v3_plan_progress.json；历史V2步骤16及G2/G3/G4状态不变。

2026-10-16 步骤16：本地协议冻结后完成6272个非真实合成episode及409项测试；结果/区间/消融见docs/16_G4_report.txt。正式G4未通过，G2/G3不升级。

2026-10-16 步骤15C：非真实完整框架迁移工程分支完成，401测试PASS。四个软件编码/VLA单元各10场景；真实2×2未测试，失败结果及G2/G3限制见docs/15C_full_transfer_report.txt。

2026-10-16 步骤14C更新：非真实完整框架工程闭环完成，398测试PASS；5个功能抓取成功。执行中换目标控制恢复通过但抓取失败，留出0/2。真实BCI/机器人未开展；G2/G3不升级。最新报告docs/14C_full_simulation_report.txt，历史14/15报告仅保留原范围。

BCI—VLA框架V2；更新2026-10-16。
最新步骤15：冻结核心后，完成新回放传输的核心等价检查，以及第二VLA π0.5的LIBERO仿真接入。
π0.5最终两个相同任务完成：抽屉139步、碗放炉子102步；PROBE/STOP故障试次通过，终点清队列。
385项完整软件回归PASS；冻结25份来源变更0。先读docs/15_transfer_matrix.txt。
新增传输仅软件回放，不是第二真实BCI；完整联合诊断/原授权与实际LIBERO仍未整合，15整体进行中。
真实BCI、真实机器人、完整2×2与独立后果校准均未做；G2未通过、G3未验证。
最终π0.5证据runs/transfer_step15_pi05_final/v4/results/；原StarVLA证据保持步骤14有效v3，开发/故障版本均保留。
配置configs/transfer/step15_binding.json；冻结execution/15_source_manifest.json；验收verify_step15_delivery.py。
旧01—14结果不覆盖；交接execution/handoff.txt；进度execution/progress.json。下一步补齐14真实仿真后端与原核心整合。

2026-10-18 V4-09修复更新：V4-09：校准和绑定已修复，策略运行门槛未通过。
CPU 79项通过；对象组24/24、编号对24/24；语言112/120，声明冲突8条仍失败。
选中策略：fixed_i88_c4_b8；策略可用：False；禁止把校准指标改善称为完整BCI→VLA成功。
报告docs/v4/09_policy_validation.txt；状态execution/v4/09_repair_audit.json。
下一步按实际策略失败与缺失监督处理；不从正式test调参；steps08广义监督/放置仍缺，旧G2/G3/G4保留。
config实际启用状态见configs/v4/policy_activation.json。真实EEG/实体动作排除。

2026-10-18 V4-09修复更新：V4-09：受限开发策略通过。
CPU 88项通过；对象组24/24、编号对24/24；语言112/120，声明冲突8条仍失败。
选中策略：fixed_i6_c4_recheck；策略可用：True；禁止把校准指标改善称为完整BCI→VLA成功。
报告docs/v4/09_policy_validation.txt；状态execution/v4/09_repair_audit.json。
下一步按实际策略失败与缺失监督处理；不从正式test调参；steps08广义监督/放置仍缺，旧G2/G3/G4保留。
config实际启用状态见configs/v4/policy_activation.json。真实EEG/实体动作排除。

2026-10-18 V4-10更新：V4-10受限开发验收完成；正式评价预案仍草案，正式test未访问。
冻结Full在validation复跑13/32解决、错误0/8；去门控21/32解决但错误7/8；Wait0/32，不宣称架构增益。
96个主比较条件+32个控制运行、6类非法提交拒绝、64歧义和64候选外快照检查完成；新增7项机制测试通过。
跨平台32条离散轨迹复现，原始浮点末位差以1e-12容差核验，178对实际模型张量完全相同；第一次失败保留。
报告docs/v4/10_development_report.txt；协议execution/v4/10_evaluation_protocol_draft.json；验收execution/v4/10_acceptance.json。
可进入V4-11适配/授权开发；步骤08含噪/放置/完整基线仍缺，声明冲突语言8条失败保留为拒绝路径。
仅固定pick、模拟BCI、LIBERO冻结观察；VLM/动作专家未加载，没有物理动力学或实体执行，旧G2/G3/G4不升级。

V4-11部分完成：按用户要求采用LisavilaLee/openpi_with_subtask，固定99c738a。
复用上游方法体+当前PyTorch同VLM互操作；没有把上游能力算作自有创新。
本地28项授权/上游接口机制测试通过，均为显式替身，不是真实VLM能力。
主任务确认仅授权规划，子任务需单独确认；STOP/CANCEL/版本变更撤销并清队列缓存。
35文件约103KB源码包已准备：execution/v4/11_code_manifest1.json；无本地数据/权重/凭据。
自动审批拒绝向现有服务器新隔离目录上传源码；未上传、未运行实权重探针。
取得明确上传授权后执行 python execution/v4/run_step11_upstream.py 1 --execute-prepared。
当前LoRA旧提示格式不代表已通过上游Task/Subtask格式，真实复验可能需要适配训练。
步骤11整体未完成，旧08缺口/09冻结/10证据/正式test封存/EEG和实体排除保持。
