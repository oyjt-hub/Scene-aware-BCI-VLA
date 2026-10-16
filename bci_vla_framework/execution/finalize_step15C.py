"""One-shot documentation and scope freeze after actual simulator migration."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[1]
assert not (R/'execution/15C_source_manifest.json').exists(),'Already frozen; run the verifier'
def read(p):return json.loads((R/p).read_text(encoding='utf-8-sig'))
def save(p,d):(R/p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def txt(p,s):(R/p).write_bytes(b'\xef\xbb\xbf'+s.replace('\r\n','\n').replace('\n','\r\n').encode())
jobs=dict(pi05='runs/transfer_step15C_full/step15C_full_20261016_v1',star='runs/transfer_step15C_full/step15C_star_20261016_v1')
summaries={k:read(v+'/results/summary.json') for k,v in jobs.items()}
assert all(s['all_checks_passed'] and all(e['reason'] is None and e['checks'] for e in s['episodes']) for s in summaries.values())
assert read('execution/15C_test_result.json')['result']=='PASS'
matrix=[]
old=read('runs/pilot_robot_14C_full_pipeline/v3/results/summary.json')
for model,s in [('StarVLA-14C',old),*summaries.items()]:
 for transport in sorted({e.get('transport','direct') for e in s['episodes']}):
  rows=[e for e in s['episodes'] if e.get('transport','direct')==transport]
  independent=read(('runs/pilot_robot_14C_full_pipeline/v3' if model=='StarVLA-14C' else jobs[model])+'/results/independent_trajectories.json')
  held=[r for r in independent if r['split']=='calibration_holdout']
  matrix.append(dict(model=model,transport=transport,episodes=len(rows),engineering_checks_passed=all(all(e['checks'].values()) for e in rows),
   task_successes=sum(e['faithful_task_success'] for e in rows),actual_sim_steps=sum(e['actual_sim_steps'] for e in rows),
   heldout_successes=sum(any(r['held'].values()) for r in held),heldout_count=len(held),
   scenarios=[dict(id=e['id'],state=e['final_state'],steps=e['actual_sim_steps'],success=e['faithful_task_success']) for e in rows]))
c=dict(scope='two_synthetic_encodings_by_two_actual_VLA_full_LIBERO_sessions',nonreal_migration_completed=True,successful_jobs=jobs,
 matrix=matrix,new_functional_episodes=30,new_independent_pi05_calibration_records=4,new_independent_pi05_holdout_records=2,
 star_calibration_reused_from14C=True,real_2x2_completed=False,second_real_BCI_tested=False,formal_heldout_transfer_claim=False,
 formal_strategy_superiority_claim=False,real_BCI_trials=0,physical_robot_commands=0,software_tests=401,
 known_target_probe_capability='explicit software codec support; not physiological evidence',
 remaining_limitations=['same two pick intents and privileged simulator grounding','fixed four-probe repair outside planner',
 'tiny calibration and heldout sample','training exposure equivalence unknown','logical clock, no human cost measurements'])
save('execution/15C_coverage.json',c)
cost=dict(frozen_core_changes=0,threshold_changes=0,human_engineering_hours=None,real_user_calibration_minutes=None,
 independent_pi05_trajectories=6,star_calibration_trajectories_reused=6,
 new_source_lines={str(p.relative_to(R)):len(p.read_text(encoding='utf-8').splitlines()) for p in list((R/'bci_vla/simulation_transfer_v4').glob('*.py'))+[R/'execution/step15C_pi05_engine.py']},
 dependencies='task-local Transformers4.53.2 copy plus existing repository patches; torch.compile disabled; original environments unchanged',
 backend_adaptation='pi05 eef3+axisangle3+gripper2 state; resize_with_pad; first8 actions; explicit controller clipping; no second normalization/gripper flip')
save('execution/15C_adaptation_audit.json',cost)
lines=['步骤15C：完整框架的非真实迁移验收（2026-10-16）','',
 '结论：两种软件输入方式×两个实际VLA的LIBERO功能矩阵完成。步骤15非真实分支工程完成；第二生理BCI和真实2×2未测试。',
 '本报告取代原15报告中“步骤14完整整合未完成”的当前状态描述；旧报告及失败数据保持历史冻结。', '',
 '一、实际链路与冻结边界',
 '沿用14C的原联合信念、runtime_v2、确认/版本授权、实际VLA仿真适配器和独立终点验证；新增simulation_transfer_v4配置与传输适配，未修改14C冻结namespace。',
 'π0.5由同一会话链路驱动，后果表与终点证据的executor_id/checkpoint均按新后端绑定；不再仅是原15的独立门控烟测。',
 'core、contracts、runtime_v2、阈值0.8、确认有效期2秒与原通道/转移数值配置保持冻结，不根据新后端成绩调参。',
 '输入A是canonical事件直接输入；输入B经PacketReplayTransport分片JSON编码/解码，每17字节一片，TASK、PROBE、CONFIRM请求和响应均走实际codec。',
 '相同固定时钟与模拟器反馈下，两路径的全部会话检查点与日志载荷一致；实际模型运行的计算耗时与随机采样不同，不声称原始日志或模型轨迹一致。', '',
 '二、完成的四个工程单元（每单元10场景）']
for m in matrix:
 lines.append(f"{m['model']} × {m['transport']}：工程检查通过={m['engineering_checks_passed']}，完成抓取{m['task_successes']}/{m['episodes']}条混合功能轨迹，实际动作{m['actual_sim_steps']}步；对应独立留出抓取{m['heldout_successes']}/{m['heldout_count']}。")
 for row in m['scenarios']:lines.append(f"  {row['id']}：{row['state']}，{row['steps']}步，任务成功={row['success']}")
lines+=['以上10条包含确认过期、STOP、推理故障、1步强制执行失败，不能把混合功能轨迹完成数解释为抓取成功率。',
 'StarVLA direct复用14C冻结结果，其余3个单元本轮实际运行。StarVLA packet与π0.5 direct/packet分别使用相同init3、seed7、两种pick契约和故障流程。',
 '固定4次公开已知目标probe修复沿用14C，保留完整目标先验后重新确认；执行中换目标在同一环境先清队列再授权新指令。', '',
 '三、独立后果校准与留出',
 'π0.5新采init0/1×2目标共4条校准，init2×2目标共2条留出；功能矩阵只用init3。StarVLA复用14C原始6条记录，校验哈希一致，不择优重采样。',
 '每个后端独立生成后果表，只有calibration记录用于Beta(1,1)后验均值；留出不参与拟合。编码方式不改变后果表，也没有在packet×π0.5单元调参。',
 'packet×π0.5在固定适配与参数下执行，属于保留组合的工程检查；样本、设备与训练暴露不满足正式跨脑机/跨模型统计留出结论。',
 'recovery_s是仿真reset墙钟，user_time_s=2是声明的合成确认预算，非真人操作成本；其他代价权重和噪声仍为示例配置。', '',
 '四、模型输入输出及适配成本',
 'π0.5实际checkpoint/归一化统计重新SHA256核验，见15C_current_assets.json；服务使用既有pi05_libero配置。',
 '图像为双视角RGB、180度旋转、官方resize_with_pad到224；状态为末端位置3、轴角3、夹爪位置2。',
 '服务返回已反归一化动作，每次取前8步，显式按LIBERO action_spec裁剪并记录次数；不再套StarVLA归一化或夹爪翻转。',
 '适配代码行数与依赖清单见15C_adaptation_audit.json；人工工时未实测为null，不声称零成本迁移。',
 '两个临时服务均只绑定服务器本机，任务结束后停止；原仓库和全局环境没有修改。', '',
 '五、验收与限制',
 '401项软件测试通过。实际模型轨迹、指令、动作、接触/抬升谓词、每次诊断/确认/授权及全部会话journal留存。',
 '所有日志禁用执行器后独立重建。跨进程浮点求和只在审计工具容许1e-12绝对差异，标识/版本/摘要/状态严格相等，运行时不放宽。',
 '源代码包、部署文件、结果和旧冻结清单校验；完整性结果见execution/15C_integrity_result.json。',
 '仅限LIBERO goal6场景中的奶油奶酪/酒瓶pick契约；物体身份和几何来自特权模拟器状态，不是通用视觉检测评估或官方goal6任务得分。',
 '抓取失败和所有留出负结果保留，不用工程检查通过代替机器人能力通过。没有真实EEG、第二生理BCI、实体机器人或真人实时预算证据。',
 'G2效用仍未通过，G3未验证，训练暴露/任务分布/随机性未做等价证明，不作策略优越性和普适框架研究结论。', '',
 '六、执行与回退',
 '隔离目录、源码ZIP和冻结计划在runs/transfer_step15C_full；重新核验用python -B execution/verify_step15C_delivery.py。',
 '部署器/终结器有防覆盖约束，复现实验应新建独立版本目录；不重跑旧步骤采集器覆盖证据。',
 '回退时停止本任务所属临时模型服务，恢复使用14C或更早冻结入口；新增namespace与后端不影响旧入口。',
 '下一步按用户指令推进步骤16；若需论文迁移性能结论，另设计多任务、多种子、独立真实通道校准与公平比较。']
txt('docs/15C_full_transfer_report.txt','\n'.join(lines)+'\n')
txt('execution/15C_acceptance.txt','步骤15非真实迁移分支工程完成：四个软件输入/VLA单元，每单元10场景；401测试PASS。\n实际失败见报告，真实2×2未测试，G2/G3不升级。\n')
p=read('execution/progress.json');p.update(current_step='15',current_step_status='步骤15非真实完整框架迁移工程验收完成；真实迁移未开展',updated_at='2026-10-16',authorization='做步骤15；沿用无真实BCI、LIBERO仿真范围')
st=p['steps'][14];st.update(nonreal_migration_completed=True,nonreal_scope_status='已完成',full_framework_simulation_migration_completed=True,real_2x2_completed=False,
 checks=['401项软件测试PASS','两个软件编码×两个实际VLA完整会话工程检查通过','独立π0.5后果校准4条、留出2条；失败保留'],
 unresolved=['second real BCI excluded and uncalibrated','real 2x2 and statistically adequate migration evaluation not performed'],next_action='非真实分支完成；依用户要求进入步骤16，研究门槛保持未通过/未验证')
st['evidence_paths']+=['docs/15C_full_transfer_report.txt','execution/15C_coverage.json','execution/15C_integrity_result.json'];save('execution/progress.json',p)
st['changed_files']+=['bci_vla/simulation_transfer_v4/*.py','execution/*15C*.py','tests/test_simulation_transfer_v4.py'];save('execution/progress.json',p)
notice='2026-10-16 步骤15C：非真实完整框架迁移工程分支完成，401测试PASS。四个软件编码/VLA单元各10场景；真实2×2未测试，失败结果及G2/G3限制见docs/15C_full_transfer_report.txt。\n'
for f in ['README.txt','execution/handoff.txt','execution/worklog.txt']:txt(f,notice+'\n'+(R/f).read_text(encoding='utf-8-sig'))
# Own delivery verifiers are deliberately outside their source manifests.
for name,var,tag in [('verify_step12_repair.py','progress','12R'),('verify_step13_delivery.py','p','13'),('verify_step14_delivery.py','p','14'),('verify_step15_delivery.py','p','15')]:
 q=R/'execution'/name;t=q.read_text(encoding='utf-8');marker='(ROOT/destination).write_text';t=t.replace(marker,f"if {var}['steps'][14].get('nonreal_migration_completed'):\n    destination='execution/{tag}_integrity_recheck_after_step15C.json'\n"+marker);q.write_text(t,encoding='utf-8')
q=R/'execution/verify_step14C_delivery.py';t=q.read_text(encoding='utf-8').replace("p['current_step']=='14'", "p['current_step'] in {'14','15'}")
t=t.replace("(ROOT/'execution/14C_integrity_result.json').write_text", "destination='execution/14C_integrity_recheck_after_step15C.json' if p['steps'][14].get('nonreal_migration_completed') else 'execution/14C_integrity_result.json'\n(ROOT/destination).write_text");q.write_text(t,encoding='utf-8')
files=set()
for pat in ['bci_vla/simulation_transfer_v4/*.py','execution/*15C*.py','execution/15C_*.json','execution/15C_*.txt','docs/15C_*.txt','tests/test_simulation_transfer_v4.py','runs/transfer_step15C_full/**/*']:
 files.update(q for q in R.glob(pat) if q.is_file())
excluded={'verify_step15C_delivery.py','15C_source_manifest.json','15C_integrity_result.json'}
save('execution/15C_source_manifest.json',dict(files=[dict(path=str(q.resolve()),sha256=hashlib.sha256(q.read_bytes()).hexdigest()) for q in sorted(files) if q.name not in excluded]))
print(json.dumps({'matrix':matrix},ensure_ascii=True))
