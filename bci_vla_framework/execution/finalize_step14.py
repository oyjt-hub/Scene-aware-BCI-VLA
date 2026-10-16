"""Finalize the user-directed LIBERO branch, not the real BCI/robot step."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
RUN='runs/pilot_robot_14_libero_simulation_v3/v3/results'
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def write(p,v):(ROOT/p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def txt(p,s):(ROOT/p).write_bytes(b'\xef\xbb\xbf'+s.strip().replace('\r\n','\n').replace('\n','\r\n').encode()+b'\r\n')
s=read(RUN+'/summary.json');tests=read('execution/14_test_result.json')
assert s['all_software_guards_passed'] and s['all_episodes_without_exception'] and tests['result']=='PASS'
assets=read('execution/14_libero_assets.json')
config={'schema_version':'libero-simulation-binding/1.0','execution_scope':'simulation_only','physical_execution_allowed':False,
    'real_BCI_enabled':False,'input_source':'scripted_simulation_fixture','full_framework_integration':False,
    'backend':'StarVLA Qwen2.5-VL-GR00T','checkpoint':s['checkpoint'],'checkpoint_sha256':assets['files'][0]['sha256'],
    'normalization_statistics':assets['files'][2],'normalization_key':'franka','normalization':'masked min/max inverse after clipping [-1,1]',
    'image_observations':{'source_keys':['agentview_image','robot0_eye_in_hand_image'],'source_shape_hwc':[256,256,3],
        'rotation':'180 degrees both spatial axes','dtype':'uint8','policy_resize_hwc':[224,224,3],'policy_key':'examples[0].image'},
    'language_key':'examples[0].lang','state_input':'not_sent_by_checked_StarVLA_LIBERO_evaluation_path',
    'normalized_policy_shape':[1,8,7],'simulation_action_dimension':7,'gripper_conversion':'1 - 2 * (open > 0.5), after repository binary threshold',
    'action_semantics':'LIBERO controller-normalized xyz/rotation_delta/gripper; not physical joint targets',
    'suite':'libero_goal','task_ids':[0,1],'seed':7,'init_state_index':0,'max_policy_steps':300,'settling_steps':10,
    'inference_device':'GPU0','service_scope':'owned process, loopback16814, terminated after run',
    'does_not_verify':['real_Piper_interface','real_EEG','full_joint_belief_planner_integration','formal_success_rate','G2','G3']}
write('configs/execution/libero_simulation_v1.json',config)
events=[json.loads(line) for line in (ROOT/RUN/'events.jsonl').read_text().splitlines()]
current=None;stopped=False;changed=False;counts={};kinds={}
for event in events:
    kind=event['kind'];kinds[kind]=kinds.get(kind,0)+1
    if kind=='episode_start':
        current=event['episode'];stopped=changed=False;counts[current]=0
        assert event['input_source']=='scripted_simulation_fixture'
    elif kind=='stop_latched':
        stopped=True;assert event['snapshot']['queued_actions']==0 and event['snapshot']['stop_latched']
    elif kind=='task_change_no_old_action':
        changed=True;assert event['snapshot']['queued_actions']==0 and event['snapshot']['confirmed'] is None
    elif kind=='probe_pause':assert event['snapshot']['queued_actions']==0 and event['snapshot']['confirmed'] is None
    elif kind=='simulation_step':
        assert not stopped and not changed and event['episode']==current
        assert event['step']==counts[current] and len(event['action'])==7
        counts[current]+=1
assert all(counts[e['episode']]==e['applied_actions'] for e in s['episodes'])
trace={'scope':'independent_recorded_trace_consistency_not_inference_or_full_session_replay','result':'PASS',
    'event_counts':kinds,'simulation_steps':counts,'no_simulation_steps_after_stop_or_task_invalidation':True,
    'full_interaction_session_replay':False}
write('execution/14_trace_audit.json',trace)
txt('docs/14_closed_loop.txt',f'''步骤14：LIBERO平台接入与仿真首轮测试（2026-10-16）
当前范围：用户明确改为“先libero仿真平台测试”；不连接真实脑机，不启动真实机械臂。
结论：已完成实际LIBERO/MuJoCo＋实际StarVLA权重＋新仿真动作门控的接入与冒烟测试。
步骤14整体仍进行中：完整联合信念/诊断策略/契约授权尚未适配LIBERO任务本体；真实BCI—机器人闭环未完成。
不能把本次脚本确认称为真实脑机输入，也不能把独立仿真门控称为已接通原InteractionSession授权。

最终有效结果（v3，seed=7，init_state_index=0，libero_goal）：
1. task0：open the middle drawer of the cabinet；独立env.check_success()为true；117个策略动作，15次模型推理。
2. task1：put the bowl on the stove；独立env.check_success()为true；80个策略动作，10次模型推理。
3. task0的PROBE/STOP注入：5步时切入PROBE并清空旧chunk，脚本重新确认后恢复；12步时STOP锁存并清空，之后没有仿真动作。
第三项是故障测试，终止为injected_stop，task_success=false，不能删掉或算任务成功。
以上是两个固定初态的接入样例，不是正式成功率、泛化或策略优越性研究。

实际执行栈：
StarVLA提交{assets['repositories']['starVLA']['commit']}，LIBERO提交{assets['repositories']['LIBERO']['commit']}。
现有Qwen2.5-VL-GR00T-LIBERO-4in1，30000步权重；SHA256及配置/统计见execution/14_libero_assets.json。
该权重是本轮选用的现有仿真基线，不是已确认对应原论文π0.5的checkpoint；没有训练或下载权重。
模型环境starVLA，仿真环境libero，独立目录/服务端口16814，GPU0；原服务器仓库与旧服务未改动。
为缺失diffusers建立任务目录内的纯Python包链接，复用openpi环境已安装的0.33.1，不修改原环境。
观测agentview与wrist RGB uint8 256×256，两轴翻转180度，resize224×224，发送examples[0].image和lang。
输出1×8×7；按franka min/max及mask反归一化，clip与夹爪二值转换和原仓库核对。
夹爪open=1转换为LIBERO -1；该7维仿真控制量不是双Piper14维关节命令。
模型不接收当前路径未发送的state；不能把图像/动作规格沿用到实体机器人。
实际运行与推理计时是机器时间，真实用户阅读、理解和probe遵从耗时未测。

修复历史与证据边界：
v1：模型启动失败，diffusers缺失；保留原日志和清理记录。
v2：服务与仿真可运行，但我新增适配层的夹爪符号与仓库已有转换相反；task0完成、task1超时。
该v2结果为接口缺陷定位材料，不用于策略能力或成功率结论。
修复后增加夹爪阈值0/0.49/0.5/1、mask、shape、非有限值、min/max与clip测试，v3全部3试次重新运行。
模型日志提示当前Transformers默认fast图像处理器与旧slow实现可能不同；本轮未对其做等价性验收，不声称复现论文性能。
启动时datasets目录警告不影响本次BDDL、init_state和资源渲染；没有使用演示数据集进行拟合或正式评估。

按14.1—14.8对应：
14.1：新隔离入口、配置、代码/权重/统计指纹、旧版本和回退说明完成。
14.2：实际模型与MuJoCo接通，图像/7维动作/夹爪映射核对；未运行实体机器人。
14.3：只有开发接入轨迹，没有独立后果表校准，也没有同场景多目的地/用户改目标对照；这些仍待实施。
14.4：记录所有3episode、失败/中止、动作/推理步数与终点。脚本任务与模型提示绑定；不能估计真实错误目标启动或忠实完成率。
14.5：未确认不可入队、PROBE清队列、任务/场景版本变化丢弃旧动作、STOP锁存通过；跨任务测试显式重置环境，没有模拟物理回滚。
14.6：停止、错误上下文、维度/NaN/越界注入留在软件或仿真，不操作实体设备。
14.7：task_limit/策略输出异常/服务异常/注入停止与真实意图或通道原因分开；没有EEG证据就不推断通道变化。
14.8：修复后完整软件回归{tests['tests_run']}项通过；所有新轨迹标开发接入/pilot仿真，未纳入正式实验。

验证与未完成项：
完整回归{tests['tests_run']}项，0失败/0错误/0跳过。独立事件一致性检查见execution/14_trace_audit.json。
该事件审计核动作计数、顺序及失效后无动作，不等于重新执行模型推理或原完整会话replay。
G2研究门槛仍未通过，G3真实BCI未验证，真实步骤14未完成。BCI、实体机器人命令均0。
后续确定工作：将LIBERO候选任务/场景词表与契约绑定接入联合信念、runtime_v2及原会话授权，加入真实模型端点回传。
随后做模拟输入下的改目标/改通道/冲突/再确认/恢复对照；不可用本轮固定脚本门控冒充完整框架验证。

证据：{RUN}/summary.json、episodes.json、events.jsonl、三个mp4及首末帧。
配置：configs/execution/libero_simulation_v1.json；冻结及验收：execution/14_source_manifest.json、14_integrity_result.json。
''')
txt('docs/14_deployment_and_rollback.txt','''步骤14 LIBERO隔离部署与回退
1. 本地新增代码：bci_vla/pilot/libero_gate.py、libero_actions.py；原01—13代码及证据不改写。
2. 新测试由step14_libero_orchestrator.py启动任务专属模型子进程，再启动有限3episode仿真worker；只连接127.0.0.1:16814。
3. 上限：服务启动240秒、worker660秒、外层1000秒。正常/异常退出均在finally终止所属模型子进程；不杀其他任务进程。
4. 最新目录/home/oyjt/Scene-aware-BCI-VLA-runs/step14_libero_20261016_v3；v1/v2保留失败与旧映射材料。
5. 未替换旧仓库、旧服务或Python环境；回退是继续使用原入口。本次任务服务已退出，不需要修改原部署。
6. 重跑时必须给deploy_step14_libero.py换新job目录及本地manifest名，避免覆盖结果；不要直接复用v3。
7. 先核空闲GPU与新端口，再核checkpoint/config/stats SHA；所有配置仍simulation_only，不能切成物理执行。
8. 保留脚本模拟输入标记、同任务BDDL/初态和环境版本；正式任务或统计研究须另建锁定协议。
9. 不启动旧eval_libero.py网页入口：它包含额外视觉/云模型依赖。本次worker只用本地权重和仿真，不加载那些入口。
10. 下次先完成LIBERO任务本体—联合信念—原授权层整合；不将脚本confirm或独立门控视为已通过G3。
''')
coverage={'scope':'user_directed_LIBERO_backend_smoke_and_simulation_gate','simulation_backend_tests_completed':True,
    'full_framework_simulation_completed':False,'real_step14_completed':False,'software_tests':tests['tests_run'],
    'episodes':3,'nominal_successes':2,'nominal_episodes':2,'fault_stop_episodes':1,'formal_success_rate_claim':False,
    'G2_passed':False,'G3_passed':False,'real_BCI_trials':0,'physical_robot_commands':0,
    'effective_run':RUN,'excluded_development_runs':['v1_missing_dependency','v2_wrong_gripper_sign'],
    'remaining':['LIBERO ontology/core belief/planner/session integration','independent consequence calibration','same-scene intention-change and recovery comparisons','real BCI/robot excluded']}
write('execution/14_coverage.json',coverage)
txt('execution/14_acceptance.txt',f'''步骤14用户指定LIBERO首轮平台测试通过。
真实StarVLA模型＋真实MuJoCo仿真：2个名义任务均完成，1个PROBE/STOP故障试次按设计中止。
修复夹爪符号后v3为有效结果；v1/v2保留但不混入结论。{tests['tests_run']}软件回归通过。
这是仿真后端与动作门控验收，不是完整联合诊断框架、真实BCI闭环或步骤14整体验收。
后续任务与部署/回退详见docs/14_closed_loop.txt和14_deployment_and_rollback.txt。
''')
p=read('execution/progress.json');p.update(updated_at='2026-10-16',authorization='执行步骤14；先libero仿真平台测试；继续排除真实BCI',
    current_step='14',current_step_status='LIBERO平台测试通过，完整框架仿真整合待完成',
    next_step='14：LIBERO任务本体与联合信念/规划/原授权整合',next_step_blocked_by='真实分支仍受G2/G3及用户排除真实BCI约束；仿真软件整合可继续')
step=next(v for v in p['steps'] if v['step_id']=='14')
step.update(status='进行中',scope='LIBERO_simulation_only',simulation_backend_tests_completed=True,
    full_framework_simulation_completed=False,real_closed_loop_completed=False,
    changed_files=['bci_vla/pilot/libero_gate.py','bci_vla/pilot/libero_actions.py','execution/*step14*.py','tests/test_libero_gate.py'],
    checks=[f'{tests["tests_run"]}完整软件回归PASS','实际StarVLA/LIBERO 2名义完成+1PROBE/STOP通过'],
    evidence_paths=['docs/14_closed_loop.txt','execution/14_coverage.json',RUN+'/summary.json','execution/14_libero_assets.json'],
    unresolved=coverage['remaining'],next_action='在仿真范围接入LIBERO任务词表/契约、联合信念与runtime_v2原会话授权；保留真实分支未开始')
write('execution/progress.json',p)
txt('README.txt',f'''BCI—VLA框架V2；2026-10-16。
最新：按用户要求执行步骤14的LIBERO仿真分支，完成实际StarVLA模型接入与动作门控测试。
最终v3：打开中间抽屉117步完成，把碗放炉子80步完成；另1个PROBE/STOP注入试次按设计停止。
{tests['tests_run']}项完整软件回归PASS。视频与日志：{RUN}/；先读docs/14_closed_loop.txt。
本轮发现并修复新增适配层夹爪符号错误，v1/v2仅保留调试材料，不能混为最终性能结果。
当前是模拟交互＋实际VLA＋实际LIBERO仿真；完整联合信念/规划/原会话授权尚未整合到LIBERO。
步骤14整体进行中；真实脑机和机械臂均未接入，G2研究未通过、G3真实未验证。
13非真实软件分支及01—12R冻结证据保留。进度execution/progress.json；交接execution/handoff.txt。
配置configs/execution/libero_simulation_v1.json；部署/回退docs/14_deployment_and_rollback.txt。
验证python -B execution/verify_step14_delivery.py；重跑实验必须新目录，不覆盖已冻结结果。
''')
txt('execution/handoff.txt',f'''2026-10-16交接：步骤14转为用户指定的LIBERO平台测试。
授权：先LIBERO仿真，真实BCI继续排除；不运行真实机械臂。
已完成真实StarVLA Qwen2.5-VL-GR00T权重＋libero_goal/MuJoCo接入；2名义任务完成，1PROBE/STOP中止检查通过。
最终证据{RUN}/；权重/config/stats完整SHA在14_libero_assets.json；{tests['tests_run']}软件测试PASS。
v1缺diffusers启动失败；v2新映射夹爪符号相反导致task1失败；修复并测试后v3三试次全部重跑。不要混用开发结果。
独立任务目录/home/oyjt/Scene-aware-BCI-VLA-runs/step14_libero_20261016_v3；GPU0、127.0.0.1:16814临时模型服务已结束。
复用openpi环境diffusers0.33.1纯Python包到job/deps，原环境和原仓库未修改；原服务器仓库提交仍d2245527a767707c19d7305caacb02fa5de89dc3。
本次输入是scripted_simulation_fixture，新LiberoSimulationGate独立实现上下文/确认/队列失效/STOP，不是原InteractionSession授权的替代证明。
原联合信念、runtime_v2、任务本体/契约尚未整合到真实LIBERO反馈；报告full_framework_simulation_completed=false。
下一条确定工作：建立LIBERO候选任务/场景词表及契约绑定，接原滤波、提议和授权层；再用模拟输入做改目标/改通道/冲突恢复对照。
后果表独立校准、同场景多目标能力和完整诊断收益未验证；真实G3/G2状态不提升，14整体进行中，15—17未开始。
源码/结果执行14_source_manifest和14_integrity_result；旧13及12R复验输出另存after_step14，不覆盖旧结果。
TXT统一UTF-8 BOM+CRLF；原冻结runner禁止覆盖，优先只读验收脚本，新仿真用新目录。
''')
old=(ROOT/'execution/worklog.txt').read_text(encoding='utf-8-sig')
entry='\n2026-10-16 步骤14按用户改为LIBERO：独立StarVLA/MuJoCo仿真v3两名义任务完成、1PROBE/STOP通过，376软件回归通过。v1缺依赖/v2夹爪符号错误保留，v3修复重跑。完整框架本体/滤波/规划/授权整合仍待做，真实BCI与机器人0。\n'
if '步骤14按用户改为LIBERO' not in old:txt('execution/worklog.txt',old+entry)
paths=[]
for pattern in ['bci_vla/pilot/libero*.py','tests/test_libero_gate.py','execution/*step14*.py','execution/14_*.json','execution/14_*.txt',
    'docs/14_*.txt','configs/execution/libero*.json','runs/pilot_robot_14_*/**/*']:
    paths.extend(q for q in ROOT.glob(pattern) if q.is_file())
paths.extend([ROOT/'execution/13_source_manifest.json',ROOT/'execution/12R_source_manifest.json'])
excluded={'execution/14_source_manifest.json','execution/14_integrity_result.json','execution/verify_step14_delivery.py'}
files=[{'path':str(q.resolve()),'sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for q in sorted(set(paths))
    if q.relative_to(ROOT).as_posix() not in excluded and '__pycache__' not in q.parts]
write('execution/14_source_manifest.json',{'scope':coverage['scope'],'full_framework_simulation_completed':False,'files':files})
print(json.dumps({'files_frozen':len(files),'tests':tests['tests_run'],'simulation_backend':'PASS','full_framework_simulation_completed':False}))
