"""Finalize the explicitly non-real Step14 branch; keep historical evidence intact."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'execution/14C_source_manifest.json').exists():
 raise SystemExit('Delivery already frozen; use verify_step14C_delivery.py. Do not overwrite evidence.')
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def save(p,d):(ROOT/p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def txt(p,s):(ROOT/p).write_bytes(b'\xef\xbb\xbf'+s.replace('\r\n','\n').replace('\n','\r\n').encode())
base='runs/pilot_robot_14C_full_pipeline/v3/results/'
s=read(base+'summary.json');assert s['all_checks_passed']
assert read('execution/14C_test_result.json')['tests_run']==398
coverage=dict(scope='nonreal_complete_framework_functional_LIBERO_pick',nonreal_closed_loop_completed=True,
 full_framework_simulation_completed=True,real_step14_completed=False,real_BCI_trials=0,physical_robot_commands=0,
 software_tests=398,functional_cases=10,engineering_checks_passed=True,completed_pick_cases=5,
 active_goal_change_control_passed=True,active_goal_change_grasp_success=False,
 independent_calibration_records=4,independent_holdout_records=2,holdout_grasp_successes=0,
 formal_success_rate_claim=False,strategy_superiority_claim=False,G2_passed=False,G3_passed=False,
 scope_limitations=['two pick intents in LIBERO goal6 scene','simulator privileged object grounding',
 'scripted BCI with logical query clock','fixed four-probe supervisor mapping repair outside planner',
 'small calibration; both heldout grasps failed','active goal-change recovered command but grasp failed'])
save('execution/14C_coverage.json',coverage)
records=read(base+'independent_trajectories.json')
save('execution/14C_calibration_audit.json',dict(records=records,fit_record_ids=[r['record_id'] for r in records if r['split']=='calibration'],
 holdout_record_ids=[r['record_id'] for r in records if r['split']=='calibration_holdout'],
 prior='predeclared Beta(1,1); posterior mean (1+failures)/(2+n)',human_time_measured=False,
 user_time_s=2,user_time_source='declared synthetic confirmation allowance',
 recovery_s_source='measured simulator reset wall time, not human recovery time',
 remaining_policy_coefficients='unchanged illustrative weights; not empirically fitted',holdout_used_in_fit=False))
report='''步骤14补全验收报告：非真实脑机完整框架闭环（14C）
日期：2026-10-16

一、结论与范围
非真实脑机分支的完整闭环工程验收完成。真实EEG试次0，物理机器人命令0。
当前链路：脚本化BCI响应→原联合信念滤波与runtime_v2诊断规划→原确认/版本授权规则→实际StarVLA推理→LIBERO动作→独立模拟器状态谓词→会话终态与离线日志重建。
工程完成不等于抓取可靠性达标。执行中换目标的控制恢复通过，但后续抓取失败；独立留出2条抓取全部失败。
原步骤14/15报告和冻结数据保留为历史。本报告补全原14仅后端烟测的缺口，只覆盖StarVLA这一后端。

二、实际实现
新代码放在bci_vla/simulation_v3；冻结的core、contracts、runtime_v2和旧pilot文件未修改。
session的authorize、dispatch、ingest、begin_interaction、request_policy、revise_context方法与原会话AST一致（仅仿真scope字符串适配）。门槛0.8、确认有效期2秒保持。
新适配器绑定checkpoint、实际双目图像/状态摘要，执行STARTED只发一次，完成反馈与独立终点判定分离。STOP或上下文变化清空动作队列；未知推理故障不自动重复提交。
独立终点使用模拟器接触与抬升超过3cm判定holding，连续3步满足后结束。它与pick契约对应，不使用官方goal6放入碗中任务成功指标冒充pick结果。
场景只含奶油奶酪、酒瓶两种pick意图；物体身份和空间框来自模拟器特权状态，未验证视觉检测或任意任务迁移。

三、诊断与恢复
保留联合目标/通道信念和原runtime_v2。初次规划暂停时，额外执行固定4次已知目标PROBE：0、1、0、1；映射不一致则暂停。
仅根据公开probe提示与响应判定编码翻转，显式修订通道profile、mapping、task、scene版本；交换通道标签时保持完整目标边缘分布，再执行新的TASK与CONFIRM。
私有目标仅供外部合成响应与事后评价使用，不用于重置先验或直接选择执行目标。
该固定监督修复流程位于原规划器之外，属于工程补全，尚未公平比较其额外探测成本或证明算法创新与优越性。

四、实测功能场景（10条，均无未处理异常）
1 正常奶油奶酪：SUCCEEDED，51步。
2 正常酒瓶：SUCCEEDED，86步。
3 目标变化：新诊断/确认后SUCCEEDED，90步。
4 通道翻转：公开probe修复、重新确认后SUCCEEDED，54步。
5 目标/通道联合混淆：修复、重新确认后SUCCEEDED，91步。
6 确认过期：拒绝授权，0步，保留AWAITING_AUTHORIZATION状态。
7 执行中STOP：CANCELLED，8步，之后无动作。
8 执行中目标变化：停止旧动作、旧授权失效、同环境重新确认和派发均通过；总168步，后续160步未抓取成功，FAILED。不能记成任务恢复成功。
9 人为限制1步导致执行失败：FAILED；不将执行失败反写为通道错误。
10 注入推理故障：UNKNOWN后停止，CANCELLED、0步；不重复提交、不反写通道。
10条的每次派发均与当时独立意图一致；这是固定开发轨迹观察，不是错误率统计保证。

五、校准与负结果
先独立运行init0/1×2目标共4条校准，init2×2目标共2条留出；功能测试只使用init3。v2/v3复用v1原始校准文件并校验SHA256，未择优重采样。
校准3条抓取成功、1条失败；留出2条均失败，完整保留。
后果表task_failure使用预声明Beta(1,1)后验均值；wrong_start来自契约目标与独立目标身份，recovery_s来自实际模拟器reset耗时；user_time_s=2为声明的合成确认开销，非人体测量。
只用校准记录拟合后果表；留出不参与。原代价权重及其他噪声参数仍为示例配置，不代表真实BCI校准或后果模型充分校准。
交互预算使用逻辑时间；实际推理/仿真另有墙钟日志，尚未验证真人实时交互预算。

六、验证与复现
398项软件测试全部通过，0失败、0错误、0跳过。所有v3会话日志禁用执行器后逐条独立重建，不依赖保存的checkpoint冒充重建。
本地跨进程重建仅在验收工具中对浮点比较采用绝对误差1e-12；实际最大差异记录在14C_integrity_result.json。标识、摘要、版本、状态仍严格相等，运行时滤波/授权/回放源码未放宽。服务器同进程运行时重建已严格相等通过。
v1首次完整测试暴露重复STARTED回放错误；v2暴露目标变化恢复暂停。两版日志、源码包及失败结果均保留，v3修复后复验。
模型checkpoint SHA256：26b36799a702ca920467c987096a8ccb5f476c6965af6c7f2fd419ecb6893bf2。
远程任务：/home/oyjt/Scene-aware-BCI-VLA-runs/step14C_full_20261016_v3。
临时模型服务仅本机16816，运行结束已停止。没有修改原服务器仓库或训练环境。
本地证据：runs/pilot_robot_14C_full_pipeline；独立验收：python -B execution/verify_step14C_delivery.py。
部署与恢复：使用隔离任务目录和sources.zip，按14C_run_plan_v3.json复现；不要直接重跑带防覆盖断言的旧采集/终结脚本。回退时停止自有临时服务并使用原冻结入口，新namespace不影响旧版本。

七、后续边界
步骤14非真实分支完成；步骤14整体真实闭环仍未开展。G2研究效用未通过，G3真人门槛未验证。
步骤15还需把π0.5接入同一完整会话/契约与独立校准，完成公平迁移评估；原步骤15第二后端独立烟测不代表本轮完整框架迁移。
论文不能据此声称普适BCI接口、真实脑机效果、跨后端优越性或可靠抓取泛化。
'''
txt('docs/14C_full_simulation_report.txt',report)
txt('execution/14C_acceptance.txt','步骤14非真实分支验收：完整框架实际模型/LIBERO闭环完成。\n详见docs/14C_full_simulation_report.txt及14C_integrity_result.json。\n控制恢复通过、执行中换目标抓取失败、留出0/2成功。真实EEG/机器人未开展，G2/G3不升级。\n')
p=read('execution/progress.json');p.update(current_step='14',current_step_status='步骤14非真实完整闭环工程验收完成；真实分支未开展',authorization='继续完成14；限脚本化BCI与LIBERO仿真',updated_at='2026-10-16')
step=p['steps'][13];step.update(nonreal_closed_loop_completed=True,nonreal_scope_status='已完成',full_framework_simulation_completed=True,
 real_closed_loop_completed=False,checks=['398项软件测试PASS','10场景工程检查PASS；5个常规/诊断抓取完成','执行中换目标控制恢复通过但抓取失败；留出0/2'],
 unresolved=['real BCI/robot excluded','heldout grasp reliability failed; no research performance claim'],
 next_action='非真实分支已完成；步骤15接入π0.5完整会话后做独立校准与迁移验证')
step['evidence_paths']+=['docs/14C_full_simulation_report.txt','execution/14C_coverage.json','execution/14C_integrity_result.json',base+'summary.json']
step['changed_files']+=['bci_vla/simulation_v3/*.py','execution/*14C*.py','tests/test_simulation_v3.py']
p['steps'][14]['unresolved']=['pi05 complete simulation_v3 session integration and independent calibration','second real BCI capability and calibration','complete framework 2x2 with held-out combination']
p['steps'][14]['next_action']='按14C完整会话接入π0.5；独立后端烟测不得视作完整迁移'
save('execution/progress.json',p)
notice='2026-10-16 步骤14C更新：非真实完整框架工程闭环完成，398测试PASS；5个功能抓取成功。执行中换目标控制恢复通过但抓取失败，留出0/2。真实BCI/机器人未开展；G2/G3不升级。最新报告docs/14C_full_simulation_report.txt，历史14/15报告仅保留原范围。\n'
for f in ['README.txt','execution/handoff.txt','execution/worklog.txt']:
 old=(ROOT/f).read_text(encoding='utf-8-sig');txt(f,notice+'\n'+old)
# Each historical verifier is excluded by its own frozen source manifest.
for f,var in [('verify_step12_repair.py','progress'),('verify_step13_delivery.py','p'),('verify_step14_delivery.py','p')]:
 q=ROOT/'execution'/f;t=q.read_text(encoding='utf-8');marker="(ROOT/destination).write_text"
 t=t.replace(marker,"if "+var+"['steps'][13].get('nonreal_closed_loop_completed'):\n    destination='execution/"+{'verify_step12_repair.py':'12R','verify_step13_delivery.py':'13','verify_step14_delivery.py':'14'}[f]+"_integrity_recheck_after_step14C.json'\n"+marker)
 q.write_text(t,encoding='utf-8')
q=ROOT/'execution/verify_step15_delivery.py';t=q.read_text(encoding='utf-8')
t=t.replace("p['current_step']=='15' and", "(p['current_step']=='15' or (p['current_step']=='14' and p['steps'][13].get('nonreal_closed_loop_completed'))) and")
t=t.replace("(ROOT/'execution/15_integrity_result.json').write_text", "destination='execution/15_integrity_recheck_after_step14C.json' if p['steps'][13].get('nonreal_closed_loop_completed') else 'execution/15_integrity_result.json'\n(ROOT/destination).write_text")
q.write_text(t,encoding='utf-8')
files=set()
for pattern in ['bci_vla/simulation_v3/*.py','execution/*14C*.py','execution/14C_*.json','execution/14C_*.txt','docs/14C_*.txt','tests/test_simulation_v3.py','runs/pilot_robot_14C_full_pipeline/**/*']:
 files.update(q for q in ROOT.glob(pattern) if q.is_file())
excluded={'verify_step14C_delivery.py','14C_source_manifest.json','14C_integrity_result.json'}
save('execution/14C_source_manifest.json',dict(scope='nonreal_full_framework_v3_and_retained_failed_prototypes',files=[dict(path=str(q.resolve()),sha256=hashlib.sha256(q.read_bytes()).hexdigest()) for q in sorted(files) if q.name not in excluded]))
print(json.dumps(coverage,ensure_ascii=True))
