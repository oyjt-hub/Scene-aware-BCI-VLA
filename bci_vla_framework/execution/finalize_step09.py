"""Write local TXT delivery records and freeze step09 implementation evidence."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
def read(p): return json.loads((ROOT / p).read_text(encoding='utf-8-sig'))
def write_json(p, v): (ROOT / p).write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
def txt(p, value):
    (ROOT / p).write_bytes(b'\xef\xbb\xbf' + value.strip().replace('\r\n', '\n').replace('\n', '\r\n').encode('utf-8') + b'\r\n')

POLICY = '''
步骤09：主动诊断动作与有限预算策略（离线有限模型首版）
版本V2；日期2026-10-16；用户授权“步骤9执行”。

一、实现结果与适用边界
入口bci_vla/core/diagnostic_actions.py、diagnostic_policy.py；配置configs/diagnosis.json。
承接08的相关联合信念与07终端后果，不训练DiT/LLM，不连接服务器、BCI或机器人。
只有两种规范任务、normal/flip通道和模拟概率表。126项软件检查通过，34项为本步新增。
所有时间、代价和概率为illustrative_not_fitted；不是已测用户时间、真实校准或实验结果。
本步“完成”专指合法动作生成、一步VoI与有限候选搜索的离线实现；不代表真实系统已可部署。

二、动作与生成条件（09.1、09.2、09.6）
TASK_QUERY：按规范goal_symbols询问当前目标；不能使用“old/new”字面别名推断目标。
CHANNEL_PROBE：分别枚举所有标记与允许时长，使用已知提示和含遵从混合的probe似然。
每个probe记录输出分布跨通道模式的总变差TV；TV=0不作为诊断候选。
最终选择按任务预期损失，不按TV或熵直接排序。对称标记可同价；非对称测试确认能选择另一个有区分度的标记。
RETRY：需要retry_allowed，含义是新鲜、不重叠的一次采集，不能重复累计旧窗。
EXTEND：需要extension_allowed，含义是预先选定较长最终窗；已消费的短前缀不能再视为独立证据。
实际窗/事件去重仍由05/06/08办理，布尔许可不是数据已去重的证明。
RECODE：已定义但当前没有另一套独立校准映射，候选日志注明不可用；不拿旧C矩阵假充新映射。
RECOMMEND_RECALIBRATION：暂停可附带建议。没有校准耗时/成功率模型，因此不将它计为可预测修复动作，不自动重训。
PAUSE/EXIT：不发任务；以当前任务未完成损失1计，不靠无限惩罚驱动探测。相同代价优先暂停。
CONFIRM：只对有正确场景/任务版本、目标匹配、仍有效的proposed契约生成。
EXECUTE_CONTRACT：枚举并注明需独立有效确认和授权；本规划器始终execution_allowed=false。
没有PROPOSE自动改目标：调用方必须先显式创建契约，再请求确认。
单一模拟registry只有binary-v1/stimulus-v1、250Hz和1/2秒窗；拒绝插值、超时长和新编码。
pending采集时整次规划暂停；不会在刺激期间换问题或菜单。实际显示与采集需在11接入。

三、规划状态与价值（09.3、09.4、09.9、09.10）
PlanningState显式保存联合belief、scene_id/revision、task_id/revision、所有proposed契约及其期限，
参考时钟now_s、已用used_s、剩余remaining_s、场景有效期、停止/采集/证据许可和inference_ready。
数值时钟为参考UTC Unix秒；不能混入设备时间或把本机当前时间代替样例时间。
from_filter_snapshot要求08输出时刻与now_s一致；paused/unknown不允许静默恢复。
当前不支持对过时snapshot自动预测；调用方须先按明确空闲时间模型更新后再规划。
depth=0：在暂停、退出和必须含噪确认的终端候选之间比较。
depth=1：对每个诊断动作完全枚举0/1/NO_RESPONSE，计算后验，再接depth=0的合法终端候选。这就是标准一步任务VoI。
depth=2：允许最多两个诊断采集，再接同样的确认/暂停终端。配置上限2；接口硬上限3，避免开放世界无界搜索。
时域计数的是诊断动作数，最终确认另保留一个位置，但确认消耗同一剩余秒预算。
Q(a,S)=elapsed(a)*time_weight + sum_y P(y|b,a)*V(next_state,depth-1)。
V包含场景、契约期限和预算的合法性；不是只依赖belief的无时钟表。
对每个正支持纯状态先计算depth=0的下一选择。如果所有解释都只能暂停/退出，就直接暂停，不为原因分类追问。
每一层均重新检查预算、场景和契约期限；不足以完成后续确认的诊断通常无收益。
根日志记录每个候选的合法性、拒绝原因、立即时间费、总损失、预测观察、概率、相关后验、
各观察分支的后续选择/损失/预算/时刻，以及选中动作和未解决完整belief。state_order给出后验向量索引。
分支后续值通过同一递归求解；根日志没有展开所有深层候选树，但不隐藏根候选或概率。
规划不修改输入belief或契约，也不产生真实观察/确认事件。

四、时间、转移与不重复收费
setup_s在当前示例中代表阅读/显示准备1秒，刺激窗1或2秒；返回/解码/计算额外时间暂声明0，必须经实测替换。
实际总时长用于截止条件和remaining_s扣减，时间权重只计一次；used_s只作已付历史，不再次收费。
继承08的条件事件核：窗前准备使用T(setup_s)，窗内声明状态固定，发射结束时得到后验。
因此本版K_a是“窗前按实际秒数转移＋固定状态采集”，不是让所有动作套同一个无时间意义的一步T。
改变setup会改变T；改变刺激时长会改变允许的发射矩阵、预算、超时和后续契约有效性。
较长窗内仍固定状态是模型条件，不代表现实更长采集没有更多变化机会。
当前没有窗内变点、随机延迟或整窗联合事件核；存在此类变化时必须暂停/模型失配，不能宣称已处理。
所有候选显式声明此条件；不能把T(total_elapsed)再乘一次，否则与08证据定义不一致。
07表中终端user_time_s=2且included_time_phases=['confirmation']。
本步只允许这类独占确认费用行：剥离该历史确认时间，保留wrong_start/task_failure/recovery_s，
再按本候选实际setup+确认窗计费。若行还含其他阶段或分量缺失则拒绝。
示例正确执行非时间损失0.2；错误目标非时间损失1+1+20*0.02=2.4。
暂停/退出损失1来自一个未完成任务，不是把其余未知恢复时间当0。
未拿07的CHECK_CHANNEL/WAIT continuation空结果当成零失败终端。
实际probe、显示、解码、切换、返回任务与确认时间须后续逐阶段标定，禁止用测试数据拟合。

五、含噪确认与停止（09.7、09.8）
本模拟配置显式指定accept_symbol='0'，reject为另一标记；不能假定所有既有请求都沿用此语义。
确认请求构造器在11必须按该映射、具体contract_id/revision和有效期生成，并经05验证；现在只返回候选计划。
CONFIRM使用06独立矩阵，正确0.8、误码0.15、遗漏0.05；不是完美目标标签。
接受分支按后验对正确/错误执行后果求期望，因此错误确认造成的错误启动被计入。
拒绝/遗漏分支明确暂停；不能反复问到“同意”。本首版不在同一规划内确认拒绝后重开任务。
模拟接受分支名为SIMULATED_ACCEPT_REQUIRES_EXTERNAL_AUTHORIZATION；既非真确认，也不能下发。
STOP在评分前拦截；不依赖VoI阈值，软件暂停通知不声称机器人已经物理停止。
unknown/none/unknown_channel有正质量、filter不ready、预算不足、场景失效、采集中都暂停并保留解释。
未知遵从只给区间的probe不可点估值；无区分度的probe剪除。实际超时、无效信号或确认未接受不自动沿旧目标执行。
若上游发现探测不遵从/时钟问题，应由08暂停，使inference_ready=false；本规划器不能从模拟概率自动识别人类真实不遵从。

六、跨任务信息（09.5）
carry_channel_to_new_task要求显式新任务先验，保留旧联合分布的channel边缘，重新建立new_goal_prior*channel_posterior。
这样保留已获得的通道信息，又不把旧任务后验或“惯性”当成新任务目标。
新任务可以100%改为另一目标；测试与09_policy_trace演示这一路径。
调用方须给新任务上下文、提议契约和正确时刻，在同一模型适用范围内另建配置/PlanningState再调用plan。
本版跨任务信息的作用在下一任务规划中兑现；没有假设未知未来用户任务或给本轮动作额外虚构长期奖励。
尚未实现多任务联合预见值或通道误更新的跨任务性能评价；该主张保留待10/12/16，不称已证明长期收益。

七、与基线和算法主张的区别
一步VoI和两步候选搜索使用完全相同belief、probe、通道/后果、预算、确认和暂停规则；只改变诊断展望深度。
周期probe按固定日程触发；本策略仅在相对合法终端选择有预期收益时探测。周期/阈值/信息增益等完整方法在10实现。
它们仍应享有相同信息、探测、确认与预算；不可只比较“无probe”弱对手。
Bayes、VoI、Bellman和有限树枚举均为标准方法，不作为原创，也不声称优于相同完整模型的最优POMDP。
固定样例depth1与depth2都选择probe:0:1.0，期望损失均0.8713177498731328；不诊断为1。
这些是手写模型的解析值，不是真实成功率，也不是实验优势。当前保留一步VoI作为默认，无两步算法增益结论。
不得为制造差异调大错误/误更新惩罚。若10/12仍无增益，收窄贡献为交互协议与机制证据。

八、复现与下一步
在D:\\Scene_Aware\\bci_vla_framework运行：
  python -B execution/run_step09_tests.py
  python -B execution/replay_step09_policy.py
前者输出126项测试；后者只保存确定性候选树与暂停/停止/跨任务先验演示，不采数据。
最终证据execution/09_test_result.json、09_test_output.txt、09_policy_trace.json、09_source_manifest.json、09_integrity_result.json。
已有05—08输出保留冻结；只用最新runner跑统一回归。不要用旧runner覆盖旧测试数和原始哈希。
下一步10需实现强基线/消融/同信息oracle，比较近似差距与计算耗时；11才办理请求发出和契约授权。
06/07真实模型与用户时间校准仍待外部条件；I01—I14保留；G1条件性通过，G2—G4未开始。
'''

ACCEPTANCE = '''
步骤09离线首版验收；2026-10-16；授权“步骤9执行”。
09.1 动作catalog包含任务问题、probe、retry/extend、recode、暂停/退出、重校准建议、含噪确认及执行占位；无模型/授权动作明确拒绝。
09.2 全标记/时长枚举、TV筛选与任务风险选择；非对称标记测试通过，不固定探测无区分度标记。
09.3 纯支持解释终端比较；都暂停时直接暂停，不为熵继续追问。
09.4 depth1标准一步VoI和depth2有限候选枚举共享模型/预算/确认；实际秒时钟与模拟耗时明确分开。
09.5 显式新目标先验×保留通道边缘；允许彻底改目标。下一任务可利用保留信息；多任务联合前瞻值和长期性能尚未实现/验证。
09.6 仅模拟registry的映射/刺激/时长；不插值、不在pending采集改菜单。真实已校准集合仍缺失，不能部署。
09.7 每个方法同样含噪确认及收费，错误接受计损失，始终不授权；STOP在评分前处理。
09.8 未知、不可辨识、无预算、期限失效、非ready、未知遵从探测等拒绝/暂停，无默认旧目标。
09.9 完整根候选及各预测观察后验、代价、分支后续动作/时间、未解决解释可JSON回放；不宣称是真实日志。
09.10 状态含场景、契约版本、时刻、已用/剩余预算、时域；K_a使用窗前T与固定窗内假设，时长扣预算/检期限；剥离07确认费后只收一次。
09.10限制：没有窗内变点/随机时长事件核；沿用08稳定窗范围，模型不满足时暂停，不能据此声称完整非平稳规划已完成。
实际命令run_step09_tests.py：126项通过（新增34，旧92），0失败/错误/跳过。
replay_step09_policy.py：一步与两步同选probe:0:1.0，同损失0.8713177498731328；是解析示例而非实验。
保留一步VoI为默认，不增加求解器原创/优越性主张。周期probe/完整强基线/oracle留10。
本步为offline_illustrative_diagnostic_policy完成；真实校准/多任务长期价值/闭环未验收，不调用硬件/云，不训练。
旧08源哈希与结果保持，阶段断言只允许进度继续，回归单独保存08_integrity_recheck_after_step09。
进度推进10未开始；G1条件性状态不变，G2—G4未开始，I01—I14保留。
'''

if __name__ == '__main__':
    tests = read('execution/09_test_result.json')
    assert tests['tests_run'] == 126 and tests['result'] == 'PASS'
    txt('docs/09_policy.txt', POLICY)
    txt('execution/09_acceptance.txt', ACCEPTANCE)
    txt('examples/diagnosis/README.txt', '''
步骤09固定模型规划示例；simulation_only。
state.json来自08规范目标与相关别名先验，新增两个proposed契约和20秒假设预算。
时间2026-10-15 UTC是固定fixture，不能替换成本轮日期后当真实记录。
prepare_step09_assets.py只显式将07 old/new映射为left/right规范ID，保留全部原始损失，不拟合参数。
python -B execution/replay_step09_policy.py保存09_policy_trace.json；没有随机性能实验/真实事件/机器人动作。
详情docs/09_policy.txt；执行前需11真实契约与授权链，不能将模拟accept当用户确认。
''')
    txt('README.txt', '''
BCI—VLA框架V2；更新2026-10-16。
01—05已完成；06离线通道、07模拟后果、08联合信念、09离线主动诊断首版完成；10未开始。
最新入口bci_vla/core/diagnostic_actions.py和diagnostic_policy.py；说明docs/09_policy.txt；配置configs/diagnosis.json。
一步任务VoI为默认；有限两步候选搜索共享预算、probe、含噪确认和后果。未知/无预算/过期明确暂停。
126项软件测试通过：python -B execution/run_step09_tests.py；解析回放replay_step09_policy.py不代表性能实验。
最新证据execution/09_acceptance.txt、09_policy_trace.json、09_source_manifest.json、progress.json和handoff.txt。
仅模拟有限模型，不训练、不采EEG、不调用云/VLA/机器人。真实通道、时间和后果待校准；窗内变点不支持。
跨任务显式保留通道边缘并重设目标先验；多任务联合前瞻值尚未实现。两步未显示额外作用，不作算法优越性主张。
完整强基线/oracle、会话授权和闭环分别留10/11；G1只条件性通过，G2—G4未开始。
主手册D:\\Scene_Aware\\BCI_VLA通用交互框架_逐步执行手册.txt。历史05—08测试/验收冻结；统一回归只用最新runner。
''')
    txt('execution/handoff.txt', '''
交接：步骤09离线主动诊断首版完成；步骤10未开始
日期2026-10-16；V2；先读docs/09_policy.txt和execution/09_acceptance.txt，再读progress.json。
核心diagnostic_actions/policy与configs/diagnosis.json，OutcomeTable规范ID配置为显式别名重映射；原07/08文件未改。
126项软件检查通过（34新+92旧），0跳过；09_policy_trace保存每候选代价/预测/后验/后续选择。
depth1标准一步VoI为默认，depth2最多两个诊断再含噪确认；同预算/动作/后果/暂停，样例选择和损失完全一致。
不声称两步增益或Bayes/Bellman原创；不要调大惩罚制造区别。
先做legality与模型状态检查；unknown正质量、pending、stop、无预算/过期会暂停。STOP不经诊断评分。
确认accept也可能错误，确认费从07行剥离后按动作实际示例时长收费；所有方法共用；最终execution_allowed恒false。
CONFIRM用本配置accept_symbol=0，不应与既有例中accept_symbol=1混用；11发请求时必须显式绑定映射、契约和版本。
重编码无单独映射、重校准无修复模型，因此只能拒绝/建议，不能假定成功。RETRY/EXTEND还须05/06/08实际去重。
K_a沿用08：窗前T(setup)，窗内固定；不支持窗内变点/随机延迟，不可宣称完整非平稳策略。
cross-task handoff需显式新goal prior，保留channel边缘，不能沿用旧目标/确认；多任务联合前瞻值未实现。
下一步10按主手册10.1—10.8建立B0—B10/M/O1/O2，至少同信息周期/阈值/联合滤波/信息增益/一步VoI；补同模型小oracle与计算时间。
不要提前运行正式性能实验，不拿手写fixture宣称成功率/优势；确认与停止所有真实方法保留，危险消融仅虚拟执行。
11再做session/契约执行授权和fake后端。未获用户要求不自动执行10或11。
06/07真实校准、实际显示/切换/返回/确认成本和I01—I14仍待条件；G1条件性通过，G2—G4未开始。
统一测试只跑run_step09_tests.py；旧05—08输出和manifest冻结，旧runner不要重跑覆盖历史计数。
''')
    oldlog = (ROOT / 'execution/worklog.txt').read_text(encoding='utf-8-sig')
    txt('execution/worklog.txt', oldlog + '''

工作记录：2026-10-16，步骤09
授权“步骤9执行”。读取主手册09、核心V2第六/七节、08交接、06/07模型及现有进度；未发现工作区AGENTS。
实现合法动作catalog、PlanningState、标准一步VoI、两步候选搜索、含噪确认、时间/期限/预算、候选审计和跨任务channel handoff。
未变更旧协议/模型；新增canonical outcome配置只显式重映射old/new；确认费从终端旧行显式剥离，避免双收。
初次手工试运行发现冻结scope为MappingProxy但旧fields只收dict，已在调用边界thaw，未放宽旧模型校验。
新增34项代数/故障路径检查，最终126项全部通过，无失败/错误/跳过。实际运行run_step09_tests.py及replay_step09_policy.py。
固定解析例一步/两步同选probe:0:1.0，损失均0.8713177498731328；保留一步默认，无两步增益主张。
本版稳定窗内状态假设、模拟耗时和单任务前瞻边界明确；跨任务保留信息但未做多任务联合前瞻/长期性能。
未采真实数据/训练/联网/连接服务器/调用机器人。I01—I14和真实校准待条件，G1条件性不变，G2—G4未开始。
旧08 verifier仅修改阶段断言并另存回归，不覆盖原08输出；09完成后10未开始。
''')
    p = read('execution/progress.json')
    p.update(updated_at='2026-10-16', authorization='步骤9执行', current_step='10',
             current_step_status='未开始', last_completed_step='09')
    outputs = ['bci_vla/core/diagnostic_actions.py', 'bci_vla/core/diagnostic_policy.py',
        'configs/diagnosis.json', 'configs/consequences/simulation_canonical_outcomes.json',
        'docs/09_policy.txt', 'examples/diagnosis/state.json', 'examples/diagnosis/README.txt',
        'tests/test_diagnostic_policy.py', 'execution/run_step09_tests.py',
        'execution/replay_step09_policy.py', 'execution/prepare_step09_assets.py',
        'execution/finalize_step09.py', 'execution/09_acceptance.txt',
        'execution/09_test_result.json', 'execution/09_test_output.txt', 'execution/09_policy_trace.json']
    p['steps'][8].update(status='已完成', execution_scope='offline_illustrative_diagnostic_policy',
        changed_files=outputs + ['README.txt', 'execution/progress.json', 'execution/handoff.txt', 'execution/worklog.txt',
            'execution/verify_step08_delivery.py', 'execution/verify_step09_delivery.py'],
        checks=['126 software tests PASS; 34 new and 92 previous',
                'analytic one-step/two-step trace and explicit pause/stop/handoff',
                'step08 frozen source consistency separately rechecked'],
        evidence_paths=['execution/09_acceptance.txt', 'execution/09_test_result.json',
            'execution/09_test_output.txt', 'execution/09_policy_trace.json', 'execution/09_source_manifest.json',
            'execution/09_integrity_result.json', 'execution/08_integrity_recheck_after_step09.json'],
        unresolved=['真实通道/后果/时间校准仍待外部条件；I01—I14保留',
            '窗内变点/随机延迟核未支持；仅稳定窗条件模型',
            '多任务联合前瞻值及长期误更新效应未实现/验证',
            '重编码/重校准修复模型缺失；完整授权与闭环留11',
            '未证明两步额外价值，保留一步VoI默认；强基线/oracle留10'],
        real_validation={'status': '待外部条件', 'deployment_allowed': False},
        next_action='用户要求时执行10：强基线、消融、同信息小oracle；不自动推进')
    write_json('execution/progress.json', p)
    inputs = [ROOT.parent / 'BCI_VLA通用交互框架_逐步执行手册.txt',
        ROOT.parent / 'BCI_VLA框架_主动诊断核心方案_V2.txt'] + [ROOT / s for s in [
        'bci_vla/protocols.py', 'bci_vla/core/joint_belief.py', 'bci_vla/models/channel.py',
        'bci_vla/models/consequence.py', 'configs/channels/simulation_binary_v1.json',
        'configs/belief/simulation_binary_filter.json', 'configs/belief/simulation_canonical_transition.json',
        'configs/consequences/simulation_outcome_table.json', 'docs/07_outcome_model.txt', 'docs/08_update_rules.txt']]
    files = [{'path': str(path), 'kind': kind, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
             for kind, paths in [('input', inputs), ('software_or_analytic_fixture', [ROOT / s for s in outputs])]
             for path in paths]
    write_json('execution/09_source_manifest.json', {'date': '2026-10-16', 'plan_version': 'V2',
        'step_id': '09', 'scope': 'offline_illustrative_diagnostic_policy', 'files': files})
    print('Step09 TXT, progress and source manifest written; delivery verification remains.')
