"""Report all locked outcomes and retain the failed/unverified research gates."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[1]
assert not (R/'execution/16_source_manifest.json').exists(),'Delivery already frozen'
def read(p):return json.loads((R/p).read_text(encoding='utf-8-sig'))
def save(p,d):(R/p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def txt(p,t):(R/p).write_bytes(b'\xef\xbb\xbf'+t.replace('\r\n','\n').replace('\n','\r\n').encode())
base='runs/registered_step16_synthetic_v1';s=read(base+'/summary.json');stats=read(base+'/paired_statistics.json');cfg=read('configs/study16.json')
rows=[json.loads(x) for x in (R/base/'evaluation_joined.jsonl').read_text(encoding='utf-8').splitlines()]
assert s['episodes']==cfg['episodes'] and not s['runtime_errors']
assert read('execution/16_test_result.json')['result']=='PASS'
coverage=dict(scope=cfg['scope'],nonreal_study_completed=True,local_protocol_locked_before_run=True,externally_registered=False,
 planned_episodes=cfg['episodes'],executed_episodes=s['episodes'],seed_clusters=32,cases=14,methods=14,
 runtime_errors=0,software_tests=409,whole_step16_completed=False,G4_passed=False,
 complete_time_overruns=sum(r['elapsed_s']>cfg['episode_budget_s'] for r in rows),missing_evaluation_rows=cfg['episodes']-len(rows),
 synthetic_primary_acceptance_passed=s['judgement']['synthetic_primary_acceptance_passed'],
 real_BCI_trials=0,physical_robot_commands=0,new_LIBERO_rollouts=0,subjective_results=None,
 limitations=['no human power or calibration','same illustrative generator family previously inspected','32 seeds are clusters not subjects',
 'six planning ablations do not remove actual joint-filter implementation','fixed execution order; CPU timing not causal speed evidence','no actual robot recovery or complete real migration test'])
save('execution/16_coverage.json',coverage)
items={
 '16.1':('部分完成','本地协议先锁；32块为可执行性预算，真人pilot/功效/正式样本量缺失'),
 '16.2':('非真实完成','2×2基础因素、可观测顺序、严重度/频率、稳定对照、无信息probe/执行失败'),
 '16.3':('非真实完成','八种冻结主方法+六个规划消融，B9同probe强对照；原LLM/NOIR外部trace不可用'),
 '16.4':('非真实完成','全部episode完成/错误/未完成/全时间与未知字段；无参数更新率记null'),
 '16.5':('部分完成','派发时压制/误启与错误通道视图记录；无实际机器人恢复性能结论'),
 '16.6':('非真实完成','新seed与旧开发种子分离，公共/私有/join分开；无真人跨用户划分'),
 '16.7':('部分完成','六项规划消融及源/授权/STOP安全负例；未实现的持续校准等仍缺'),
 '16.8':('完成','合成、既有LIBERO、真实脑机、实体机器人与自然变化证据分层'),
 '16.9':('非真实完成','按seed整块配对、固定情境、区间和保守边界；不做受试者推断'),
 '16.10':('完成','固定样本、同probe主比较、接受条件在运行前冻结；失败全留'),
 '16.11':('仅模板','主观量表与直接控制对照方案；无受试者答案'),
 '16.12':('完成','按实际结果收缩策略贡献；G2/G3/G4不以工程通过升级')}
save('execution/16_requirement_coverage.json',dict(items={k:dict(status=v[0],evidence=v[1]) for k,v in items.items()}))
lines=['步骤16 G4报告：本地预先冻结的合成复验','日期：2026-10-16','',
 '结论：非真实协议、实验运行、区间分析和负结果判定已执行。步骤16整体仍进行中，正式人体/真实迁移G4未通过。',
 '本轮是L1合成复验，不是新增LIBERO抓取实验，不是公开注册平台登记。此前已知12R的M/B9相同及M2风险，不能声称未经探索的算法盲测。',
 '协议源码Git提交：'+read('execution/16_registration_git_receipt.json')['protocol_commit'],
 '锁定时间：'+read('execution/16_preregistration.json')['created_utc']+'；运行开始：'+read(base+'/start_receipt.json')['started_utc'],
 '旧15C备份分支与提交保持不动；本轮工作在work/step16-synthetic-20261016。', '',
 '一、执行规模',f"全部{s['episodes']}个episode保留：14情境×32新seed块×14方法；无运行异常，409项软件测试通过。",
 '主方法B4/B5/B7/B8/B9/B10/M/M2；六个规划消融A_no_probe/A_periodic/A_fixed_duration/A_no_task_consequence/A_ignore_probe_cost/A_factorized。',
 '同一seed覆盖全部情境和方法；配对随机数按mode和该mode次数/时长构造，适应性采样不同不冒充同一原始数据。',
 '新seed10000—10031；开发种子0—23不计入。32块只按预算固定，无真人样本量依据；真实参数和主观体验未采集。', '',
 '二、主要比较 M−B9（正完成差有利、正错误/耗时差不利）']
for row in stats['primary']:
 lines.append(f"{row['metric']}：配对均差{row['mean_difference']:.8f}；seed块bootstrap区间{row['cluster_percentile_interval']}，alpha={row['alpha']:.8f}。")
 if 'bounded_difference_Hoeffding_interval' in row:lines.append('  二元差保守Hoeffding区间：'+str(row['bounded_difference_Hoeffding_interval']))
lines += ['主要判定：'+json.dumps(s['judgement'],ensure_ascii=False),
 f"M与B9全部动作序列相同={s['M_B9_action_sequences_equal']}；完成/错误/未启动结局相同={s['M_B9_outcomes_equal']}。",
 '若全零差导致bootstrap零宽区间，也不能证明总体等效或非劣；二元差同时给保守区间。预声明精度对应的保守块数为'+str(cfg['conservative_blocks_for_005_hoeffding_halfwidth'])+'，本轮32块不足。',
 ('未满足预声明的完成增益/错误启动上界/耗时联合条件，不能声称新策略优于同probe标准VoI。若B9相同，应将贡献收缩至协议机制和工程实现。' if not s['judgement']['synthetic_primary_acceptance_passed'] else '仅在锁定合成范围满足主条件；不升级真实G4或一般策略结论。'),
 'Hoeffding界依赖独立seed块及差值范围[-1,1]假设；固定hash种子是生成器设计，不是生理独立性证据。', '',
 '三、完整方法结果（全部episode，包括暂停/失败，不只成功时间）']
for row in s['overall']:
 lines.append(f"{row['method']}：n={row['episodes']}，虚拟完成={row['faithful_virtual_completion']:.6f}，错误启动={row['wrong_start']:.6f}，未解决={row['unresolved']:.6f}，未启动={row['unstarted']:.6f}，完整逻辑耗时={row['elapsed_s']:.6f}s，probe={row['probe_s']:.6f}s。")
lookup={r['method']:r for r in s['overall']}
lines += [f"M2−B9错误启动差={lookup['M2']['wrong_start']-lookup['B9']['wrong_start']:.8f}；不能只展示更高完成或选取成功子集。",
 '全部次要基线/消融区间见paired_statistics.json；均为描述性分析，不挑最有利比较作显著性结论。', '',
 '四、机制与边界',
 '两种变化顺序的新条件在采样窗口之间发生，第一窗口真值不同；与12R中都在首窗前完成的不可观测顺序明确区分。',
 '窗内变化由注入检测器抑制为NO_RESPONSE，必须解释为理想化安全检测假设，不能据此声称真实EEG非平稳检测有效。',
 '实际信念滤波、确认和授权规则保持冻结。factorized/代价消融仅改变规划视图或目标，不是完整删除联合滤波器的消融。',
 '源混合、无效确认、停止等使用安全负例；禁止通过绕过授权夸大任务完成。公共轨迹不含私有时间表，评价真值在轨迹生成后join。',
 '主要真值定义在派发时（未派发取结束时）；不能解释为对执行期间后续意图变化的持续机器人恢复证明。',
 '执行器失败由独立随机键生成；执行反馈不作为BCI通道参数/信念监督。没有实际在线参数更新，更新错误率为null。',
 '时间由声明的setup/采集/返回/虚拟执行加实际CPU计算组成，不是人体全任务实测时间。',
 '方法按锁定列表固定顺序运行；CPU耗时受缓存和负载影响，时间条件只描述本机条件，不能单独支持策略速度的因果优势。',
 '完整时间超过20秒的episode数='+str(coverage['complete_time_overruns'])+'；缺失评价行='+str(coverage['missing_evaluation_rows'])+'。完成/错误/未解决均用全部episode作分母。', '',
 '五、前置证据和未完成项',
 '14C/15C实际VLA-LIBERO模型、独立状态谓词和失败轨迹仍作为工程证据单列，未混入本轮虚拟完成或用来制造模型成功率。',
 '真实G3未验证；真实脑机迁移、真实2×2、用户负担/控制感及公平原LLM/NOIR实测trace仍缺。',
 '正式人体样本量、设备/时序误差、训练暴露等价与足够的跨用户/会话留出都未落实，不能宣称正式G4通过。', '',
 '六、16.1—16.12逐项交付']
for k,(status,description) in items.items():lines.append(k+' '+status+'：'+description)
lines += ['', '七、复核与接续',
 '核验：python -B execution/verify_step16_delivery.py。锁定来源在16_preregistration.json；全量结果与来源分层在runs/registered_step16_synthetic_v1。',
 '所有旧冻结清单保留、逐项复核；旧verifier仅适配当前进度并另存after_step16结果，不覆写历史结果。',
 '不重跑同一输出目录、不根据test调参数。新假说需新协议/新版本/新数据，不能在本轮结果上补调后沿用预注册标签。',
 '后续步骤17应据此重构贡献与证据链，明确策略收益未证明、真实验证缺失及工程贡献边界。']
txt('docs/16_G4_report.txt','\n'.join(lines)+'\n')
txt('execution/16_acceptance.txt','步骤16非真实分支：本地协议冻结、6272个episode、统计与逐项证据完成。\n409软件测试PASS；完整性另见16_integrity_result.json。\nG2未通过、G3未验证、G4正式研究未通过；不得将本轮称作人体/实际模型实验。\n')
p=read('execution/progress.json');p.update(current_step='16',current_step_status='步骤16非真实复验完成；正式G4证据不足',updated_at='2026-10-16',authorization='做步骤16；沿用排除真实BCI/机器人范围')
st=p['steps'][15];st.update(status='进行中',nonreal_study_completed=True,formal_G4_passed=False,
 changed_files=['bci_vla/study_v1/*.py','configs/study16.json','execution/*step16*.py','tests/test_study_v1.py','docs/16_*.txt'],
 checks=['409项软件测试PASS','本地协议在6272个新seed episode前锁定','全结果、错误与配对区间保留'],
 evidence_paths=['docs/16_registered_protocol.txt','docs/16_G4_report.txt','execution/16_preregistration.json','execution/16_coverage.json','execution/16_integrity_result.json',base+'/summary.json'],
 unresolved=['正式人体pilot/样本量/主观量表未完成','真实BCI迁移与真实G4前置未满足','策略收益未建立'],next_action='用户要求步骤17时按负结果与真实证据缺项重构论文贡献，不自动推进人体实验')
p['research_gates']['G4'].update(status='未通过',scope='synthetic_replication_complete_formal_G4_evidence_insufficient',research_acceptance_passed=False,
 formal_real_study_evaluated=False,nonreal_study_completed=True,synthetic_verdict=s['judgement']['verdict'],evidence_paths=st['evidence_paths'])
save('execution/progress.json',p)
notice='2026-10-16 步骤16：本地协议冻结后完成6272个非真实合成episode及409项测试；结果/区间/消融见docs/16_G4_report.txt。正式G4未通过，G2/G3不升级。\n'
for f in ['README.txt','execution/handoff.txt','execution/worklog.txt']:txt(f,notice+'\n'+(R/f).read_text(encoding='utf-8-sig'))
for name,var,tag in [('verify_step12_repair.py','progress','12R'),('verify_step13_delivery.py','p','13'),('verify_step14_delivery.py','p','14'),('verify_step15_delivery.py','p','15'),('verify_step14C_delivery.py','p','14C')]:
 q=R/'execution'/name;t=q.read_text(encoding='utf-8')
 t=t.replace("{'13','14','15'}","{'13','14','15','16'}").replace("{'14','15'}","{'14','15','16'}").replace("p['current_step']=='15'","p['current_step'] in {'15','16'}")
 t=t.replace('(ROOT/destination).write_text',f"if {var}['current_step']=='16':\n    destination='execution/{tag}_integrity_recheck_after_step16.json'\n(ROOT/destination).write_text")
 q.write_text(t,encoding='utf-8')
q=R/'execution/verify_step15C_delivery.py';t=q.read_text(encoding='utf-8').replace("p['current_step']=='15'","p['current_step'] in {'15','16'}")
t=t.replace("(R/'execution/15C_integrity_result.json').write_text","destination='execution/15C_integrity_recheck_after_step16.json' if p['current_step']=='16' else 'execution/15C_integrity_result.json'\n(R/destination).write_text");q.write_text(t,encoding='utf-8')
files=set()
for pat in ['bci_vla/study_v1/*.py','execution/*step16*.py','execution/16*.json','execution/16*.txt','docs/16*.txt','configs/study16.json','tests/test_study_v1.py','runs/registered_step16_synthetic_v1/**/*']:
 files.update(q for q in R.glob(pat) if q.is_file())
excluded={'verify_step16_delivery.py','16_source_manifest.json','16_integrity_result.json'}
save('execution/16_source_manifest.json',dict(files=[dict(path=str(q.resolve()),sha256=hashlib.sha256(q.read_bytes()).hexdigest()) for q in sorted(files) if q.name not in excluded]))
print(json.dumps(coverage,ensure_ascii=True))
