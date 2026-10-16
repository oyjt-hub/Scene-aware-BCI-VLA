"""Deliver feasible transfer tests without upgrading missing real/matrix evidence."""
from pathlib import Path
import hashlib,json,statistics
ROOT=Path(__file__).resolve().parents[1]
RUN='runs/transfer_step15_pi05_final/v4/results'
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def write(p,v):(ROOT/p).parent.mkdir(parents=True,exist_ok=True);(ROOT/p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def txt(p,s):(ROOT/p).write_bytes(b'\xef\xbb\xbf'+s.strip().replace('\r\n','\n').replace('\n','\r\n').encode()+b'\r\n')
s=read(RUN+'/summary.json');tests=read('execution/15_test_result.json');freeze=read('execution/15_pre_run_freeze.json')
assert tests['result']=='PASS' and s['all_episodes_without_exception'] and s['all_software_guards_passed']
assert all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in freeze['source_files'])
events=[json.loads(line) for line in (ROOT/RUN/'events.jsonl').read_text().splitlines()]
chunks=[e for e in events if e['kind']=='model_chunk'];terminal=[e for e in events if e['kind']=='terminal_queue_cleared']
assert len(terminal)==3 and all(e['snapshot']['queued_actions']==0 and e['snapshot']['confirmed'] is None for e in terminal)
clipped=sum(e['controller_conversion']['clipped_values'] for e in chunks)
assets=read('execution/15_pi05_assets.json');old=read('runs/pilot_robot_14_libero_simulation_v3/v3/results/summary.json')
rows=[]
for backend,summary in [('StarVLA',old),('pi05',s)]:
    for e in summary['episodes']:rows.append({'backend':backend,**{k:e[k] for k in ['episode','suite','task_id','seed','init_state_index','task_success','termination','applied_actions','inference_calls']}})
write('execution/15_backend_observations.json',{'scope':'two_backend_development_smoke_not_performance_comparison','rows':rows,
    'same_tasks_initial_states_seed_horizon':True,'matched_training_exposure_or_policy_randomness_verified':False,
    'pi05_controller_clipped_values':clipped,'pi05_model_chunk_calls':len(chunks),'terminal_queues_drained':3,
    'paired_statistical_or_backend_superiority_claim':False})
write('configs/transfer/step15_binding.json',{'scope':'simulation_and_replay_development_only','core_threshold_changes':False,
    'frozen_sources_ref':'execution/15_pre_run_freeze.json','real_BCI_enabled':False,'physical_robot_enabled':False,
    'backend_A_ref':'configs/execution/libero_simulation_v1.json','backend_B':{'name':'existing_pi05_LIBERO_SFT',
        'checkpoint':s['checkpoint'],'weight_sha256':assets['files'][0]['sha256'],'norm_stats_sha256':s['stats_sha256'],
        'state':'eef_position3 + quaternion_to_axisangle3 + gripper_qpos2','output':'already unnormalized LIBERO actions',
        'extra_gripper_flip':False,'extra_unnormalization':False,'replan_steps':8,
        'controller_clipping':'explicit env.env.action_spec [-1,1], counts retained'},
    'second_real_BCI':{'backend_id':None,'modality':None,'supports_known_target_probe':None,'calibration_ref':None,'status':'not_selected_or_tested'},
    'software_transport':{'name':'PacketReplayTransport','provenance':'synthetic_codec_fixture','supports_known_target_probe':True,
        'not_a_second_physiological_BCI':True},'held_out_real_combination_tested':False})
extra=['bci_vla/adapters/bci/packet_replay.py','bci_vla/adapters/vla/libero_pi05.py','bci_vla/pilot/transfer_replay.py',
    'execution/step15_libero_worker.py','execution/step15_pi05_orchestrator.py']
effort={path:sum(bool(line.strip()) for line in (ROOT/path).read_text(encoding='utf-8').splitlines()) for path in extra}
write('execution/15_adaptation_audit.json',{'frozen_sources_count':len(freeze['source_files']),'frozen_sources_changed':0,
    'core_and_threshold_changes':0,'added_nonblank_lines_by_file':effort,'generated_worker_derives_from':'frozen step14 worker',
    'human_engineering_hours_measured':False,'human_engineering_hours':None,'new_real_channel_calibration_records':0,
    'new_independent_backend_consequence_calibration_trajectories':0,'real_second_BCI_tested':False,
    'changes':['alternate replay transport','OpenPI image/state/output mapping','task-local Transformers patch','explicit simulator clipping','terminal queue drain'],
    'does_not_prove_zero_adaptation_or_zero_calibration_transfer':True})
a,b=s['episodes'][0],s['episodes'][2]
txt('docs/15_transfer_matrix.txt',f'''步骤15：冻结核心后迁移BCI与VLA——可执行分支交付（2026-10-16）
当前范围沿用用户要求：排除真实BCI，先在LIBERO平台测试。
结论：第二VLAπ0.5已在相同LIBERO任务/初态接通；两种回放传输经原核心的检查点与日志语义一致。
本轮没有完成第二真实BCI或2×2完整框架迁移；步骤14的LIBERO任务本体/原授权层整合仍未完成，步骤15整体记进行中。

一、冻结与禁止调参
在新后端结果之前保存execution/15_pre_run_freeze.json，共{len(freeze['source_files'])}个核心、模型、授权、规划、阈值及原仿真门控来源。
运行前后SHA256相同，改动数0；授权门槛0.8、确认时效2s及runtime_v2策略未重调。
新的输入/模型转换放在适配层；并非零适配成本，也不能由本轮证明完整框架零改动迁移。
原01—14冻结代码和结果保留；无训练、无真实EEG、无实体机器人命令。

二、实际执行矩阵（各自范围不能混合）
回放输入A（原canonical event） × 原核心＋FakeVLA：TASK→PROBE→CONFIRM→授权→提交通过，独立日志重放通过。
回放输入B（新分包JSON传输） × 原核心＋FakeVLA：同上通过；5个核心检查点和journal事件/状态载荷相同。
两份journal拥有不同随机实例ID/哈希链，比较语义载荷而非声称原始字节相同。
脚本模拟交互 × 实际StarVLA＋LIBERO：引用步骤14有效v3，2个名义任务完成＋1个PROBE/STOP试次通过。
脚本模拟交互 × 实际π0.5＋LIBERO：本轮有效v4，2个名义任务及故障试次见下。
上述不是“两个真实BCI×两个VLA”的已完成2×2矩阵。

真实矩阵：
真实SSVEP × StarVLA：完整框架未验证，真实校准缺失。
真实SSVEP × π0.5：完整框架未验证，真实校准缺失。
第二真实BCI × StarVLA：设备/范式/已知目标probe能力/校准未落实，未验证。
第二真实BCI × π0.5：未执行；可在前置条件成立后另锁为留出组合，当前不能声称留出验证通过。
PacketReplayTransport是软件传输，不是第二生理BCI，不用键盘或改标签替代真实性能。

三、第二VLA实际结果（LIBERO goal，任务0/1，seed7，初态0，最大300策略步）
π0.5打开中间抽屉：成功={a['task_success']}，{a['applied_actions']}动作步，{a['inference_calls']}次推理。
π0.5把碗放炉子：成功={b['task_success']}，{b['applied_actions']}动作步，{b['inference_calls']}次推理。
另1次probe/stop试次在第5步前进入PROBE、重新确认、第12步前STOP；清队列和拒绝旧动作通过；不算任务成功。
最终3个episode退出均清空队列并撤销确认；实际终点使用env.check_success()，不以模型ACK或脚本完成标记代替。
StarVLA旧样例为117/80步，π0.5本轮为{a['applied_actions']}/{b['applied_actions']}步；这些固定初态样例不构成优劣或泛化结论。
训练任务暴露、模型随机性、编译/处理器与完整环境冻结未做等价验证；不做统计显著性或正式成功率比较。

四、接口差异与实际修复
π0.5权重：{s['checkpoint']}；权重/统计/配置来源指纹见execution/15_pi05_assets.json。
这是现有LIBERO SFT仿真基线，不是已经唯一确认对应原论文的π0.5权重。
StarVLA使用两张图像＋语言、franka统计反归一化及夹爪转换；π0.5使用图像＋语言＋8维状态。
π0.5状态为末端位置3、四元数转轴角3、夹爪qpos2；图像使用其官方resize_with_pad和uint8路径。
π0.5服务已完成反归一化，不能再套StarVLA统计或翻转夹爪；每次取前8个动作，独立仿真门控和故障时刻保持冻结。
v1服务因缺Transformers补丁退出；v2在task-local4.53.2补丁后可推理，但输出越出门控范围，3试次全部被阻断、未下发动作。
核对robosuite Controller.scale_action内np.clip后，在新后端适配层按环境action_spec显式裁剪；没有放宽核心门槛。
v3两个任务成功，但末端仍残留未消费队列；v4新增终点清队列并重新跑完3试次。所有开发版本日志保留。
最终模型chunk共{len(chunks)}次，实际送入队列的前8步累计裁剪{clipped}个标量，逐次原始范围/边界/次数保留在events.jsonl。
Task-local补丁目录只复制Transformers及仓库自带替换文件，不修改原环境；torch.compile在本冒烟运行关闭，不能据耗时声称速度收益。
π0.5临时服务127.0.0.1:16815已结束，仅终止本任务所属进程。

五、按15.1—15.7验收
15.1：第二真实BCI未选定/未测；已提供能力显式声明与“不支持已知目标probe就拒绝”的软件检查，不默认MI可用。
15.2：新增回放分包适配和配置模板；真实TASK/PROBE/CONFIRM通道校准、遵从与采集成本均未测。
15.3：第二实际VLA在共同LIBERO条件接通，核心不调参；独立能力范围与后果表未完成校准，不能用于当前策略研究效用结论。
15.4：未形成完整框架2×2，不将开发样例充作留出测试。
15.5：本轮两个后端都在LIBERO；实物π0.5仍单列未验证，不把平台差异归为模型替换收益。
15.6：冻结来源变更0；适配代码及依赖变更逐项记录，旧证据不覆盖。
15.7：新增代码行数、依赖补丁、校准数与结果在15_adaptation_audit.json；人工工时未实测，填null。
核心诊断行为仅在回放层得到等价证据；真实通道/后端后果变化下的诊断保留效果尚未验证。

六、下一条确定工作
先完成步骤14尚缺的LIBERO任务本体、联合信念、runtime_v2、原会话授权与实际后端终点回传整合。
随后做模拟通道变化/意图变化与恢复的两后端公平检查，独立划分后果校准和验证轨迹。
若继续排除真人采集，只能完成模拟迁移研究；第二真实BCI和真实2×2应保留未验证。
G2研究门槛未通过，G3未验证，不能据本轮接入通过进入带性能结论的正式实验。

证据：{RUN}/，runs/transfer_step15_replay/，execution/15_pre_run_freeze.json、15_backend_observations.json、15_adaptation_audit.json。
软件回归：{tests['tests_run']}项PASS；交付验证：execution/15_integrity_result.json；所有旧测试结果保留。
''')
txt('docs/15_execution_notes.txt','''步骤15复跑、回退与证据规则
1. 阅读15_transfer_matrix.txt与configs/transfer/step15_binding.json。确认只有回放编码等价及真实模型仿真接入，不是已完成真实2×2。
2. 优先只读运行python -B execution/verify_step15_delivery.py。不要运行旧finalize或runner覆盖冻结来源/结果。
3. 新模型运行需新job目录与本地manifest名；deploy_step15_libero.py当前v4路径不可直接复用。
4. 临时目录/home/oyjt/Scene-aware-BCI-VLA-runs/step15_libero_20261016_v4；模型服务只绑定127.0.0.1:16815，GPU0。
5. 原环境不修补；Transformers4.53.2和OpenPI替换文件复制到task-local deps。运行关闭torch.compile，仅用于有限冒烟试次。
6. 服务启动240秒、worker660秒、外层1000秒；finally仅结束所属模型服务。旧服务/旧仓库不替换，继续原入口即回退。
7. π0.5输出已经反归一化，禁止重复处理；环境action_spec必须匹配[-1,1]七维，否则拒绝，不能猜物理单位。
8. 非有限数、shape错误直接拒绝；有限数按已核LIBERO控制器规则裁剪并记录数量；成功/失败终点均清队列与确认。
9. 回放PacketReplayTransport仅允许synthetic_codec_fixture和offline_replay，禁止把真实数据改标签伪装成回放。
10. 第二真实BCI仍未知；实现真实适配/校准后再检查probe能力和规划器动作集合，不能仅接传输就宣称可部署。
11. 旧13/14与G2/G3状态保留；新测试结果为开发/pilot，不进入正式独立测试或留出组合。
''')
coverage={'scope':'step15_feasible_replay_and_second_VLA_simulation_only','replay_encoding_equivalence':'PASS',
    'second_VLA_simulation_backend':'PASS','software_tests':tests['tests_run'],'frozen_sources_changed':0,
    'real_second_BCI_tested':False,'real_2x2_completed':False,'full_framework_migration_completed':False,
    'independent_consequence_calibration_completed':False,'formal_held_out_transfer_tested':False,
    'nominal_pi05_successes':s['nominal_task_successes'],'nominal_pi05_episodes':2,'stop_injection_episodes':1,
    'controller_clipped_values':clipped,'real_BCI_trials':0,'physical_robot_commands':0,'G2_passed':False,'G3_passed':False,
    'effective_pi05_run':RUN,'remaining':['step14 full framework LIBERO integration','independent consequence calibration',
        'second real BCI capability and calibration','complete framework 2x2 with held-out combination']}
write('execution/15_coverage.json',coverage)
txt('execution/15_acceptance.txt',f'''步骤15可执行分支：回放编码等价PASS；第二VLA的LIBERO接入与仿真门控PASS；{tests['tests_run']}软件回归PASS。
冻结核心/参数来源变更0；两个π0.5任务成功，另1个故障试次按设计停止，终点队列已清空。
真实第二BCI、完整框架迁移、2×2留出与独立后果校准未完成，步骤15整体保留进行中。
主报告docs/15_transfer_matrix.txt；冻结来源execution/15_source_manifest.json；不自动通过G2/G3。
''')
p=read('execution/progress.json');p.update(updated_at='2026-10-16',authorization='执行步骤15；沿用无真实BCI、LIBERO仿真范围',
    current_step='15',current_step_status='回放等价与第二VLA仿真接入通过，完整迁移未完成',
    next_step='补齐14完整框架LIBERO整合，再做15两后端迁移',next_step_blocked_by='真实分支仍排除；第二真实BCI/校准及完整框架集成缺失')
step=next(v for v in p['steps'] if v['step_id']=='15')
step.update(status='进行中',scope=coverage['scope'],changed_files=extra+['tests/test_transfer_replay.py','tests/test_pi05_libero_adapter.py'],
    checks=[f'{tests["tests_run"]}软件回归PASS','冻结核心变更0','2回放传输核心等价PASS','第二VLA实际LIBERO 2名义完成+1故障通过'],
    evidence_paths=['docs/15_transfer_matrix.txt','execution/15_coverage.json','execution/15_adaptation_audit.json',RUN+'/summary.json'],
    unresolved=coverage['remaining'],next_action=p['next_step'],replay_transport_equivalence_completed=True,
    second_VLA_backend_smoke_completed=True,full_framework_migration_completed=False,real_2x2_completed=False)
write('execution/progress.json',p)
txt('README.txt',f'''BCI—VLA框架V2；更新2026-10-16。
最新步骤15：冻结核心后，完成新回放传输的核心等价检查，以及第二VLA π0.5的LIBERO仿真接入。
π0.5最终两个相同任务完成：抽屉{a['applied_actions']}步、碗放炉子{b['applied_actions']}步；PROBE/STOP故障试次通过，终点清队列。
{tests['tests_run']}项完整软件回归PASS；冻结25份来源变更0。先读docs/15_transfer_matrix.txt。
新增传输仅软件回放，不是第二真实BCI；完整联合诊断/原授权与实际LIBERO仍未整合，15整体进行中。
真实BCI、真实机器人、完整2×2与独立后果校准均未做；G2未通过、G3未验证。
最终π0.5证据{RUN}/；原StarVLA证据保持步骤14有效v3，开发/故障版本均保留。
配置configs/transfer/step15_binding.json；冻结execution/15_source_manifest.json；验收verify_step15_delivery.py。
旧01—14结果不覆盖；交接execution/handoff.txt；进度execution/progress.json。下一步补齐14真实仿真后端与原核心整合。
''')
txt('execution/handoff.txt',f'''交接2026-10-16：用户执行步骤15，保留无真实BCI与LIBERO范围。
可执行部分完成：两回放编码TASK/PROBE/CONFIRM/授权/提交检查点与日志语义一致，各自独立重放；核心25份来源SHA未变。
第二VLA选现有RLinf-Pi05-LIBERO-SFT，实际OpenPI服务＋LIBERO完成2任务，1PROBE/STOP故障检查通过；{tests['tests_run']}软件回归PASS。
最终{RUN}/；主报告docs/15_transfer_matrix.txt；配置configs/transfer/step15_binding.json。
v1缺Transformers补丁退出；v2输出越界被门控阻断；v3加控制器裁剪两个任务成功但终点残留队列；v4清空终点队列并完整重跑。
π0.5已反归一化，不再用StarVLA统计/夹爪翻转；按实际env.action_spec七维[-1,1]裁剪，次数保留；NaN拒绝。
隔离目录/home/oyjt/Scene-aware-BCI-VLA-runs/step15_libero_20261016_v4；GPU0、127.0.0.1:16815所属服务已结束，原环境不修改。
task-local Transformers4.53.2补仓库transformers_replace，torch.compile关闭；不做速度或正式成功率对照。
新增PacketReplayTransport为synthetic_codec_fixture，不是第二真实BCI；未知真实BCI/probe/通道校准保持未测。
14的任务本体—联合信念—runtime_v2—原会话授权—LIBERO反馈尚未完整整合。15真实2×2/留出/后果校准未做。
下一条确定工作：先补14核心与实际LIBERO后端整合，再做模拟通道/目标变化的两后端公平测试。
G2研究未通过、G3未验证；14/15整体进行中；16/17未开始。不把当前接口证据包装成完整迁移或创新研究结论。
验收verify_step15_delivery.py；旧12R/13/14重验另存after_step15，不覆盖旧结果。TXT为UTF-8 BOM+CRLF。
''')
oldlog=(ROOT/'execution/worklog.txt').read_text(encoding='utf-8-sig')
entry=f'\n2026-10-16 步骤15可执行分支：冻结25份核心/阈值未改，2回放传输核心等价；π0.5 LIBERO v4两任务完成与PROBE/STOP通过，{tests["tests_run"]}回归PASS。真实第二BCI/2×2/独立后果表与完整框架集成未完成，进度保留进行中。\n'
if '步骤15可执行分支' not in oldlog:txt('execution/worklog.txt',oldlog+entry)
paths=[]
for pattern in ['bci_vla/adapters/bci/packet_replay.py','bci_vla/adapters/vla/libero_pi05.py','bci_vla/pilot/transfer_replay.py',
    'tests/test_transfer_replay.py','tests/test_pi05_libero_adapter.py','execution/*step15*.py','execution/15_*.json','execution/15_*.txt',
    'docs/15_*.txt','configs/transfer/*.json','runs/transfer_step15*/**/*']:
    paths.extend(q for q in ROOT.glob(pattern) if q.is_file())
paths.append(ROOT/'execution/14_source_manifest.json')
exclude={'execution/15_source_manifest.json','execution/15_integrity_result.json','execution/verify_step15_delivery.py'}
files=[{'path':str(q.resolve()),'sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for q in sorted(set(paths))
    if q.relative_to(ROOT).as_posix() not in exclude and '__pycache__' not in q.parts]
write('execution/15_source_manifest.json',{'scope':coverage['scope'],'full_migration_completed':False,'files':files})
print(json.dumps({'files_frozen':len(files),'software_tests':tests['tests_run'],'pi05_task_steps':[a['applied_actions'],b['applied_actions']],
    'clipped_values':clipped,'full_migration_completed':False}))
