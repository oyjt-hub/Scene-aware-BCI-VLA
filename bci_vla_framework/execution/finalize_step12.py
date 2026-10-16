"""Freeze a completed negative G2 audit without presenting it as a research pass."""
from pathlib import Path
import csv
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/offline_step12_acceptance'
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def write(p,v):(ROOT/p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def txt(p,s):(ROOT/p).write_bytes(b'\xef\xbb\xbf'+s.strip().replace('\r\n','\n').replace('\n','\r\n').encode('utf-8')+b'\r\n')


def main():
    result=read('runs/offline_step12_acceptance/summary.json');tests=read('execution/12_test_result.json')
    oracle=read('runs/offline_step12_acceptance/oracle_audit.json');gap=read('execution/12_runtime_alignment.json')
    assert result['episodes']==720 and tests['result']=='PASS' and tests['tests_run']==311
    assert result['G2_verdict']=='NOT_PASSED_PENDING_MODEL_ALIGNMENT_AND_CLAIM_NARROWING'
    assert result['M_vs_B9_actions_all_equal'] and result['M_vs_B9_outcomes_all_equal']
    table='方法 | 流程数 | 忠实虚拟完成 | 错误启动 | 未解决 | 全流程秒(均值) | probe秒(均值)\n'
    for r in result['overall']:
        table+=f"{r['method']:4s} | {r['episodes']:3d} | {r['faithful_virtual_completion']:.4f} | {r['wrong_start']:.4f} | {r['unresolved']:.4f} | {r['mean_elapsed_s']:.6f} | {r['mean_probe_s']:.4f}\n"
    count=tests['tests_run'];blocked=result['authorization_wrapper_blocked_episodes']
    txt('docs/12_G2_report.txt',f'''
步骤12：离线机制、oracle及故障验收报告
V2；日期2026-10-16；用户授权“做步骤12”。

一、结论与进度
步骤12的开发、合成流程、机制/故障检查及报告已经执行完成。
G2结论：未通过，需模型对齐和贡献收窄后复验。软件检查通过不等于研究门槛通过。
当前停在步骤12修复/复验阶段，步骤13未开始且由G2阻断；未进行真实BCI、服务器、云调用或机器人操作。
不能把本报告改写为“新框架有效”“优于标准VoI”或“达到真实脑控完成率”。

二、实际执行资产与复现
新增bci_vla/benchmark/offline.py（公开控制端）、synthetic_world.py（评价私有生成器）、runner.py（流程调度/后置评价），
audits.py（机制、有限oracle、故障），runtime_gate.py（规划与授权不一致的确定性反例）。
公开控制器仅取得会话/公开配置/公开历史，不接收case、seed、真实goal/channel或未来变化计划。
私有世界只向控制器返回经同一模型生成的含噪类别响应；独立评价在控制器完成后连接目标/通道真值。
每个流程使用随机不编码条件的UUID；没有把评价端理解确认或真值当免费在线确认。
这是逻辑/API隔离，不是抵御恶意文件读取的OS权限沙箱。测试检查在线模块不导入私有世界、禁用truth访问仍可决策。

项目根目录D:\\Scene_Aware\\bci_vla_framework，实际命令：
  python -B execution/run_step12_offline.py
  python -B execution/run_step12_tests.py
  python -B execution/verify_step12_delivery.py
离线默认输出runs/offline_step12_acceptance；已完成目录拒绝覆盖。重跑必须指定新的--output目录。
统一软件回归{count}项PASS，无失败/错误/跳过，含原291项；不要运行旧runner覆盖冻结结果。

三、12.1：完整流程接入与记录
每条流程从共同TASK回放开始，接05的绑定/去重校验器、07固定场景、08联合滤波、09/10策略和11的契约/授权及FakeVLAAdapter。
选择只提出动作；真实绑定的含噪CONFIRM响应后才可authorize，之后才dispatch。所有确认受同一0.8门槛/2秒规则。
一次流程最多一次最终确认；拒绝/无响应/证据不足均保留为未解决，不重新问到同意。
模拟提交ACK不算执行，需显式STARTED、COMPLETED及本次绑定EndpointEvidence；日志逐记录回放不调用后端。
端点生成器按实际下发目的地生成虚拟物体位置，与用户任务卡真值独立。契约SUCCEEDED仍可能是错误用户目标。
例如同时改目标/翻转通道时，旧目标指令的虚拟端点可以满足旧契约，但独立评价仍记wrong_start且不记忠实完成。
720条流程全部保留，全部会话日志逐检查点回放一致；JSONL压缩包保留所有公开事件及末尾hash锚。

四、12.2：条件与暴露规则
10种开发条件×12个公开端不可见的配对种子×6种方法，共720条流程，每方法120条。
四格：C00双稳定、C10仅改目标、C01仅翻转通道、C11_simultaneous同时改变。
另有C11_goal_first、C11_channel_first、stable_sporadic_error、wrong_strong_prior、probe_noncompliance、uninformative_probe。
基础模型保留随机误码和NO_RESPONSE；稳定偶发错误条件只强制首个TASK发生一次误码，不标持久通道变化。
错强先验条件公开先验为(.97,.01,.01,.01)；其它默认(.81,.09,.09,.01)。不依据结果重新调先验/代价。
probe_noncompliance：控制端假设imperfect，但世界使用uninformative遵从，作为明确模型失配压力。
uninformative_probe：控制端和世界均知道遵从无信息，作为负对照，不把无信息观测更新成确定性。

所有受控变化在共同首个TASK采集窗(1—2秒)之前完成，采集窗内隐状态固定；方法不能早停逃避变化暴露。
默认变点0.5秒；先后改变条件为0.25/0.75秒，均在该采集窗之前。
因此三个C11条件在已观察窗中的有效世界相同，本轮没有时间分辨能力，不能声称验证了变化先后顺序的识别或变点延迟。
未扫描渐变、burst、窗内变化或更多目标/通道。12种压力的全部笛卡尔积不在本轮完成范围。
条件是预定合成压力，非按CTMC自然抽样的人体数据；规划CTMC仍为原示例参数，不能反推真实变化率。

五、12.3—12.4：机制边界
8项解析机制检查通过。
二元码本原映射与0/1交换下，“右目标+normal”与“左目标+flip”对每个输出的TASK分布相等；重编码本身不能破别名。
未独立校准的新码本在动作目录中仍被拒绝，不假定换频率即可修复通道。
已知提示的PROBE似然对给定z不依赖g；不把探测标签直接替换成用户目标。
在相关先验(左/flip=.5，右/normal=.5)下，观测probe输出0后，因联合相关性目标边缘可改变。
使用imperfect遵从模型时该例最大目标后验为0.7675/0.855≈0.8976608，不是完美遵从下的数值。
独立均匀先验下同一probe不改变目标边缘；无信息probe保持全部联合质量且I(Z;Y)=0。
“有信息”是条件模型性质；并非每次随机观测都必须降低熵，也未验证真实受试者理解或task/probe迁移。

六、12.5：同probe强基线与全流程结果
共同方法B4（联合无probe）、B5（固定周期）、B7（冲突/可用质量阈值）、B9（标准一步任务VoI）、M（原一步候选）、M2（有限两步）。
B5/B7/B9/M/M2都允许同一套probe；共同两个目标、校准、动作时长/菜单、确认、STOP、预算与执行器。
B7未提供实际可测quality，保持None，仅用公开last_confirmed_goal和冲突阈值；没有伪造坏质量来帮助触发。
B5 period等沿用10的development_only配置，本轮不调参；不能称已充分验证的最强周期法。
每流程20秒预算、共同首窗后最多2次诊断、最终确认一次、单次计算预算5秒；所有结束状态进入分母。
所有用户时间含阅读/切换setup和采集/确认；全时间还包括本机实际计算、0.01秒示例返回/授权和虚拟执行/端点阶段。
区间并集避免重复计费。失败/弃权时间照录，不只统计成功流程；本机compute_ns另存，均非真实交互或机器人时延。
计算预算超限检测是在返回后丢弃结果，不是能够抢占卡死进程的硬实时watchdog。

以下为10条件等权混合的开发描述，仅便于检查，主要应读summary.json逐条件结果：
{table}
三个C11条件可观察世界相同，混合表因此重复加权该压力；不能将120条/方法当独立真实试次或总体性能估计。
M与B9在全部120个配对条件/种子上的动作序列及完成/错误启动/未启动结果均相同。
M在本矩阵没有主动选probe，不能以模块名称声称已产生主动诊断收益。
M2在部分仅目标变化条件增加忠实完成，也在其它条件增加错误启动和耗时；本轮未满足“不牺牲错误启动”的研究主张。
周期法在稳定期会多付probe时间；但存在条件差异不等于M拥有新的VoI算法贡献。没有删B9、提高惩罚或筛条件来制造胜出。
12个合成种子不足以作正式统计、非劣或临床结论；没有用本矩阵选择新阈值或删除不利条件。

七、12.6：有限oracle核对及其缺口
8种联合先验×两套时长菜单×horizon0/1/2，记录80项M/B9/M2与相同展望O1的动作Q核对。
内部规划模型最大同展望决策regret={oracle['max_same_horizon_regret']:.12g}，最大planning value差={oracle['max_planning_value_gap']:.12g}。
记录actual perf_counter_ns、候选数、horizon、4个隐状态及5秒开发计算预算；最大O1计算{oracle['max_oracle_compute_ns']/1e6:.4f}毫秒。
仅改变树展望、菜单和先验；2目标×2通道固定，未做更大状态空间扩展。单次本机计时不是可部署延迟benchmark。
不把horizon1和2的价值差冒称同展望近似误差，也不宣称胜过完整最优POMDP。

关键缺口：09/10的有限终端模型在observed accept分支估执行后果，未包含11的严格授权后验门槛。
720条流程中实际有{blocked}条在该门槛被阻断，未删除、未改为成功。内部O1等价只能证明原内部模型一致，不能证明完整运行最优。
确定性反例：目标先验0.4，normal通道CONFIRM实际输出accept0；Bayes后验0.32/(0.32+0.09)={gap['posterior_after_noisy_confirm']:.10f}。
它低于0.8，11正确拒绝authorize，真实fake提交为0；原规划accept分支却仍按执行估值。
独立反例execution/12_runtime_alignment.json与单测保留；没有为通过G2降低授权门槛或把确认改成无噪。
因此G2必须先对齐“规划模型”和“实际授权/暂停后果”，再用相同规则重核O1、所有强基线和完整流程。

八、12.7：故障验收
10种独立注入均通过且各自日志可回放：
无响应、旧菜单、过期确认、重复确认/重复提交、目标消失、策略超计算预算、非法非有限动作、取消后晚到响应、执行中STOP、未知提交禁止重发。
全部仅software fault，非真实EEG噪声替身或硬件安全验证。STOP请求后仍REQUESTED，显式模拟反馈后才SIMULATED_STOPPED；物理停止始终false。
非法action只验证fake格式/数值边界，没有驱动器、碰撞、关节限位、RTC、动作平滑或物理停止实验。
既有05—11的窗口去重、跨会话端点、时效等回归也纳入{count}项统一测试。

九、12.8：来源、完整性与研究门槛
全部结果标L1/software/synthetic/replay。既有SSVEP回放不等于新probe遵从效果；本轮根本没有采集/使用新真人EEG。
runs/offline_step12_acceptance/evaluation_private/保存case、seed、变点和噪声；online/仅保存公开输出、决策与日志。
run_plan在运行前固定源hash、条件/种子、预算；artifacts_manifest保存所有正式开发输出hash。不是正式预注册或独立验证集。
首次开发试跑runs/offline_step12_development保留作调试记录；验收使用后续随机UUID版acceptance目录，未按结果选模型/阈值。
初次单测误把imperfect probe后验要求成>0.9，已改成含遵从混合的解析值0.7675/0.855；未调整模型来满足断言。
早期01—11源/测试结果冻结；只使11 verifier允许后续阶段并将回查另存，不覆盖原11完整性结果。

G2逐项判定：
  核心不可辨识/probe边界：在声明模型内通过。
  状态、来源、授权和故障行为：软件检查通过。
  强同probe基线与负对照：保留并执行；无信息probe不制造信息。
  完整规划与执行契约模型对齐：未通过，已有确定性反例。
  候选复杂策略有独立额外价值：未证明；M=B9，M2无一致优势。
总体G2未通过，不直接上大规模真人/真机研究。真实校准、π0.5接口和I01—I14仍待外部条件。
修复顺序详docs/12_repair_plan.txt；修复须新版本/新运行目录保留本次负结果，不以改指标或反复确认绕过失败。
''')
    txt('docs/12_repair_plan.txt','''
步骤12负结果后的修复与复验清单；当前未执行这些修复，不进入13。
1. 先读12_G2_report.txt、12_runtime_alignment.json及acceptance/summary.json，保留原结果和01—11冻结源。
2. 从确定性反例写新的版本化规划/执行契约桥，明确每个CONFIRM观测分支的授权概率门槛、确认年龄、版本/场景失效、返回与计算延迟及暂停后果。
3. 保留0.8/2秒原示例规则用于回归，不降低门槛、不借评价端真值或无噪确认；模型变化另开配置版本并说明未校准。
4. 用独立实现的O1同步覆盖同一终端约束，不能只对主方法补规则；周期、阈值、B9、M/M2共用实际确认/授权/STOP与时间。
5. 核对先验0.4、accept后0.7804878这一分支：规划价值必须包含授权失败及未完成，而不是继续计执行收益。对低/高后验、延迟降阈、缺失证据、过期、STOP均写分支测试。
6. 核对双变化但TASK输出不变的C11：这可能没有冲突触发，不能声称总可识别。需要的额外证据、探测时机及其成本须与同probe强基线公平比较，不能把隐藏变点当触发器。
7. 重新冻结独立的开发配置及新运行目录，完整保留四格、稳定随机错误、错先验、不遵从、无信息probe、所有弃权/失败时间和错误启动。
8. 若研究先后变化，在固定观察阶段中加入能分辨顺序的公开采样窗及明确暴露规则；窗内变点不受当前稳定窗模型支持，先实现条件核或保持窗口外改变。
9. 验证集调参必须用独立分组与冻结规则；本次12个种子和反例已经用于开发，不当正式test，也不根据表现删B9或选择有利子集。
10. 报告同展望O1差距、4状态适用界线及实际计算预算；若扩展目标/通道数，必须新增模型与状态规模验收，不能沿用二元结果宣称通用。
11. 收窄贡献：Bayes/CTMC/VoI和更多规划深度均不是新算法贡献；M与B9等效时按采用标准VoI说明。只保留有独立证据支持的双来源交互协议/辨识边界与实际效用主张。
12. G2复验须明确：协议机制正确、完整运行模型一致、强基线公平且有值得继续验证的差异；若仍无额外收益，继续收窄而非自动标通过。13还需真实BCI/采样/遵从与当前采集安排落实。
''')
    coverage={f'12.{i}':{'status':'executed','evidence':[]} for i in range(1,9)}
    refs={1:['runs/offline_step12_acceptance/online/journals.jsonl.gz'],2:['runs/offline_step12_acceptance/evaluation_private/run_plan.json'],
        3:['runs/offline_step12_acceptance/mechanism_checks.json'],4:['tests/test_offline_benchmark.py'],
        5:['runs/offline_step12_acceptance/summary.json','runs/offline_step12_acceptance/evaluation_joined.jsonl'],
        6:['runs/offline_step12_acceptance/oracle_audit.json','execution/12_runtime_alignment.json'],
        7:['runs/offline_step12_acceptance/fault_acceptance.json','execution/12_test_result.json'],
        8:['docs/12_G2_report.txt','runs/offline_step12_acceptance/artifacts_manifest.json']}
    for i,paths in refs.items():coverage[f'12.{i}']['evidence']=paths
    write('execution/12_coverage.json',{'scope':'offline_execution_coverage_not_research_pass','substeps':coverage,
        'G2_passed':False,'verdict':result['G2_verdict']})
    txt('execution/12_acceptance.txt',f'''
步骤12执行与验收记录；日期2026-10-16；用户授权“做步骤12”。
12.1 完整TASK回放→联合滤波→策略→含噪确认→11严格授权→fake执行→独立虚拟端点，720条日志逐检查点重放；不构造/调用后端回放。
12.2 四格/双稳定偶发错误/同时及前后变化/错强先验/不遵从/无信息probe，10条件×12配对种子×6方法。先后变化均在首窗前，顺序不具可观察性，不主张变点定位。
12.3 二元码本不能破别名；有效probe在声明假设下提供证据；无信息probe不变联合belief且MI=0。
12.4 probe仅依赖通道/已知提示，可经联合相关性影响目标；独立先验目标不被覆盖。控制端无truth/seed/case输入，评价后置；逻辑隔离不宣称恶意调用安全。
12.5 同probe B5/B7/B9/M/M2及无probe B4完整比较；保留全部失败/弃权和全时间。M与B9全部配对动作/结果等效，M2未满足错误启动不增加。
12.6 80项相同展望内部O1核对最大regret0，记录实际计算预算/菜单/深度，4状态固定；发现原模型未包含11授权门槛，故不能称完整运行oracle，G2需修复。
12.7 无响应/旧菜单/过期与重复确认/目标消失/策略超时/非法动作/取消/停止/未知提交共10种故障均通过且日志可回放。
12.8 全标software/synthetic/replay，公开/私有/后置评价及源hash和输出hash分别记录；未训练、EEG/云/服务器/硬件调用均0。
{count}项软件回归PASS、0失败/错误/跳过；仅证明实现/机制检查。总体G2未通过：规划/实际授权模型不一致，独立复杂策略价值未证明。
步骤12执行完成但研究验收未过，当前停12修复/复验，不开始13；修复计划docs/12_repair_plan.txt。真实校准/π0.5/I01—I14保留。
''')
    with (ROOT/'execution/12_episode_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        cols=[k for k in result['per_case_method'][0] if k!='termination_counts'];writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore')
        writer.writeheader();writer.writerows(result['per_case_method'])
    txt('README.txt',f'''
BCI—VLA框架V2；更新2026-10-16。
01—11既有离线实现完成；12的完整流程/机制/oracle/故障验收已经执行，G2未通过。当前停在12修复/复验，13未开始。
先读docs/12_G2_report.txt及12_repair_plan.txt；证据runs/offline_step12_acceptance/。
720条L1合成流程（10条件×12配对种子×6方法）全部保留/日志重放一致；不是人体或机器人结果。
{count}项软件测试PASS（含原291项），8机制检查和10故障检查通过；80项内部有限O1相同展望核对regret0。
M与同probe标准一步VoI B9的120组配对动作/结果完全相同；M2增加部分完成同时增加错误启动，不能声称额外策略优势。
已证实原09/10终端代价未包含11严格授权门槛，内部O1不等于完整运行oracle；不要降低授权门槛来通过验收。
执行入口execution/run_step12_offline.py；统一回归仅execution/run_step12_tests.py；重跑需新的--output，不能覆盖现有完成目录。
Fake/DryRun-only；不训练、不接EEG/π0.5/机器人，无云/服务器调用；真实校准/接口和I01—I14仍待条件。
主计划D:\\Scene_Aware\\BCI_VLA通用交互框架_逐步执行手册.txt；进度execution/progress.json；交接execution/handoff.txt。
''')
    txt('execution/handoff.txt',f'''
交接：步骤12执行完成；G2未通过，先修复/复验，不能进入13
日期2026-10-16；授权“做步骤12”。先读docs/12_G2_report.txt、12_repair_plan.txt和execution/12_runtime_alignment.json。
新benchmark包接冻结05—11实际校验/filter/策略/授权/fake/端点；在线与评价私有生成器分开，随机opaque UUID，所有720流程及其失败保留。
正式本轮开发证据runs/offline_step12_acceptance：10条件×12种子×B4/B5/B7/B9/M/M2；online包含输出/压缩完整日志，evaluation_private包含预定计划和噪声，evaluation_joined只用于事后评分。
首次runs/offline_step12_development是早期调试试跑，保留而不作为最终验收；没有依其结果调参或删方法。
{count}软件测试PASS；8解析机制/10故障/80内部O1核对完成；最大同展望regret0不是完整系统最优证明。
M在本矩阵不主动probe，且与B9全部配对动作/结果等效；M2在部分条件多完成同时多错误启动，不能写成更强或创新已成立。
确定性反例：prior目标0.4，CONFIRM输出accept0后目标后验0.7804878<0.8，11阻止授权；09/10仍给接受分支计执行后果，规划/实际代价未对齐。
先实现新的版本化授权感知终端核，并对独立O1和所有基线同等处理；保留现有门槛/负结果，不改旧冻结源或借无噪确认通过。
四格变化全部在共同首窗前，C11先后顺序未被采样区分；当前稳定窗模型不支持窗内变化，不能冒称变点检测。
全时间由模拟setup/acquisition/return/execution与实际本机compute组成，含失败/弃权，非真实交互时间；quality保持None，先验/阈值未拟合。
当前进度步骤12待修复与复验，G2未通过，G3/G4和13—17未开始；不启动采集/服务/硬件。
可复现：python -B execution/run_step12_tests.py；新开发运行python -B execution/run_step12_offline.py --output runs/offline_step12_new_run。
只调整11 verifier进度条件并另存历史回查，11及以前源/测试证据不变。真实校准、精确checkpoint/接口、持久提交/身份认证、I01—I14继续保留。
''')
    log=(ROOT/'execution/worklog.txt').read_text(encoding='utf-8-sig');marker='工作记录：2026-10-16，步骤12'
    if marker not in log:txt('execution/worklog.txt',log+f'''

{marker}
授权“做步骤12”；单智能体执行，未启动子代理。读12.1—12.8和11交接，新增公开控制器/私有合成世界/后置评价/审计及测试。
10条件×12种子×6方法=720流程，全部保留含弃权/失败，全日志逐检查点重放且回放不调用后端。
初次开发试跑后将公开流程ID改为不编码条件的随机UUID，并在新的acceptance目录重跑完整720条；旧开发结果保留，未调模型参数。
机制8项、故障10种通过，内部O1扩展80项相同展望regret0；软件{count}项PASS，无跳过。
单测最初错误要求含imperfect遵从的probe后验>0.9，按解析混合值0.7675/0.855修正，未更改模型。
研究负结果保留：M=B9全部配对动作/结果；M2部分改善同时增加错误启动；完整规划终端代价未包含11授权门槛，反例后验0.7804878被正确阻断。
G2标未通过，12执行完成但停修复/复验；不进入13。01—11冻结源与测试结果不改，11 verifier只放宽后续进度/另存回查。
无训练、真人EEG、服务器、云调用或机器人操作；真实外部缺项不冒充已解决。
''')
    outputs=[str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'bci_vla/benchmark').glob('*.py')]+[
        'tests/test_offline_benchmark.py','execution/run_step12_offline.py','execution/run_step12_tests.py','execution/finalize_step12.py',
        'execution/12_test_output.txt','execution/12_test_result.json','execution/12_runtime_alignment.json',
        'execution/12_episode_summary.csv','execution/12_coverage.json','execution/12_acceptance.txt',
        'docs/12_G2_report.txt','docs/12_repair_plan.txt']
    p=read('execution/progress.json');p.update(updated_at='2026-10-16',authorization='做步骤12',
        current_step='12',current_step_status='待修复与复验',last_completed_step='12',
        next_step='13',next_step_blocked_by='G2未通过；须先按12_repair_plan修复并复验')
    p['steps'][11].update(status='已完成',execution_scope='L1_synthetic_development_acceptance',
        acceptance_result='G2未通过',changed_files=outputs+['README.txt','execution/handoff.txt','execution/worklog.txt',
            'execution/progress.json','execution/12_source_manifest.json','execution/verify_step11_delivery.py','execution/verify_step12_delivery.py'],
        checks=[f'{count} software tests PASS; 0 failures/errors/skips',
            '720 synthetic episodes kept and journals replay each checkpoint','8 mechanism / 10 injected fault checks passed',
            '80 internal same-horizon oracle checks; max regret0; not complete runtime oracle',
            'negative G2: authorization gate mismatch and no distinct M over B9 advantage'],
        evidence_paths=['docs/12_G2_report.txt','docs/12_repair_plan.txt','execution/12_coverage.json',
            'execution/12_runtime_alignment.json','execution/12_test_result.json','execution/12_source_manifest.json',
            'execution/12_integrity_result.json','execution/11_integrity_recheck_after_step12.json',
            'runs/offline_step12_acceptance/summary.json','runs/offline_step12_acceptance/artifacts_manifest.json'],
        unresolved=['09/10终端代价未包含11严格授权，需新版本规划/O1/基线共同对齐后复验',
            'M与B9等效，M2无一致额外效用；收窄标准VoI算法创新主张',
            'C11顺序未被窗口区分、无窗内变化核、固定4状态、不构成规模/真实性能验证',
            '真实校准/遵从/后果/耗时/π0.5接口与I01—I14仍待条件'],
        next_action='先修复12_repair_plan并复验G2；不得自动进入13',
        real_validation={'status':'未开展','deployment_allowed':False,'hardware_calls':0,'cloud_calls':0})
    p['research_gates']['G2'].update(status='未通过',verdict='需模型对齐和贡献收窄后复验',
        scope='L1_synthetic_development_only',evidence_paths=p['steps'][11]['evidence_paths'],
        software_checks_passed=True,research_acceptance_passed=False,
        blockers=p['steps'][11]['unresolved'][:2])
    p['steps'][12].update(blocked_by=['G2'],next_action='G2复验通过且真实采集条件/当前授权落实后才执行；本轮未开始')
    write('execution/progress.json',p)
    sources=[ROOT.parent/'BCI_VLA通用交互框架_逐步执行手册.txt',ROOT/'execution/11_source_manifest.json',
        ROOT/'execution/verify_step11_delivery.py']+[ROOT/q for q in outputs]
    sources+=list(q for q in RUN.rglob('*') if q.is_file())
    write('execution/12_source_manifest.json',{'date':'2026-10-16','step_id':'12','scope':'L1_synthetic_development_acceptance',
        'G2_passed':False,'files':[{'path':str(q),'sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for q in sorted(set(sources))]})
    print('Step12 execution and negative G2 report frozen; step13 blocked.')


if __name__=='__main__':main()
