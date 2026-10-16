"""Local TXT delivery and source evidence for step10's offline branch."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def write(p,v):(ROOT/p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def txt(p,s):(ROOT/p).write_bytes(b'\xef\xbb\xbf'+s.strip().replace('\r\n','\n').replace('\n','\r\n').encode('utf-8')+b'\r\n')

COMPARISON='''
步骤10：强基线、消融与小规模oracle比较矩阵
V2；日期2026-10-16；授权“步骤10实行”。范围：离线有限模型实现及软件验收。

一、完成情况与不能扩大解释的边界
新增bci_vla/baselines/finite_model.py、methods.py、oracle.py、selection.py和configs/comparisons/。
B0—B10、M、O1/O2均列清单，另保留B1P预测纠正、M2两步候选两个变体。
其中B0明确缺外部原LLM调用记录和完整实测时间，返回unavailable而非填假结果；其余为当前二元模拟适配。
43项新增软件检查，加旧126项共169项通过，0失败/错误/跳过。
4个手写小例、12项同信息同展望的oracle核对，决策后悔值均0，并记录实际本机计算耗时。
不是独立验证集、真实BCI/机器人数据或正式实验；不把169项测试写成成功率，不把4例当G2通过。
真实“最强可部署基线”暂为空；需独立先导校准、成套验证集调参和部署审计。10离线首版完成不代表这项实证已完成。

二、输入与共同约束
共同输入为PlanningState的公开联合信念、场景/契约版本、参考时刻、剩余秒预算，以及公共历史/质量/预测记忆。
每种可比方法可用同一目标词表、校准数据、probe标记、问题、合法时长、执行后果模型。
各方法对信息/动作的刻意限制逐项如下，不能偷偷不给强基线probe。
所有方法共享最终CONFIRM误码/耗时、STOP与期限检查；execution_allowed始终false。
没有真实执行后端比较：executor同为07已标未验证的模拟接口，未核准多目的地真机能力。
质量未知保持None，不填好质量；predicted_goal须来自独立公共预测器，样例只是显式手写预测，不能来自真g/z。
API拒绝true_goal、true_channel等私有记忆字段；O2不在choose在线方法列表。
observe只处理有限模拟离散观察，不处理真实事件/原始分数，不宣称具备05/08实际事件去重能力。
后续接真实数据时仍必须用05合法输入、06最终前缀与08来源/时钟/去重规则；不将此模拟桥替换生产协议。

三、逐方法矩阵：ID | 行为/信息差异 | 当前范围
B0 | 原固定补选LLM链，源码身份见02冻结清单 | 外部API/完整查询与推理耗时、真实proposal记录缺失；不实际调用，不填0耗时或预测。
B1 | NOIR-inspired固定选择：置信不足任务查询，足够后共同最终确认 | 用当前两个完整目标替代原对象/技能层级，非完整NOIR复现。
B1P | 外部预测与公开belief一致且足够明确时直接最终确认；不一致/不明确时查询核验，回答支持另一目标则纠正 | 不把预测或核验当授权，不反复确认到同意；预测拒绝后的信息来自含噪任务证据。
B2 | 固定normal通道假设，每次推断投影固定通道；目标仍随时间/证据可变 | 实际评分世界仍允许通道变化，不拿错误模型当真实世界。
B3 | 独立通道边缘监测，目标gap保持、证据可改目标，g/z独立重组，无主动probe | 目标惯性/被动监测的明确简化，不是最强已发表漂移估计器复现。
B4 | 相关联合belief，允许任务澄清，不主动已知目标探测 | 主动probe被明确删除，其余共同终端规则不变。
B5 | 固定诊断次数周期，首次处探测相位；到期选同模型风险最低probe，否则任务阶段 | period可调1/2/3/4，未声称已充分调参；固定周期不按隐藏变化时刻对齐。
B6 | 公共历史目标冲突超过阈值时重复任务问题或合法新窗重试/延长 | 无新鲜窗/最终前缀许可则不能重用；与无噪确认不同。
B7 | 历史目标冲突或实际可得信号质量阈值触发probe | 质量缺失不假造，阈值待独立验证集选。
B8 | 在通道不确定时选每秒I(Z;Y)最大的合法probe，随后单独语义澄清 | 参数信息收益不同于任务后果，不能用真实标签替在线熵。
B9 | 标准一步任务VoI，完整枚举同一问题、probe与后续含噪终端 | 独立枚举实现，可和M的09代码互核；不是弱化版VoI。
B10 | RBC-inspired递归Bayes证据＋MI问题选择＋最终窗置信停时＋语义VoI/probe | 仅单一已验证二元码本，不能代表完整MMI重编码或窗内动态停时；未训练actor-critic。
M | 步骤09原默认一步任务VoI/主动诊断 | 与B9同模型同信息，当前没有独立算法优势。
M2 | 步骤09有限两步候选 | 对照horizon2的O1，不能用horizon1差距冒称近似误差。
O1 | 同belief、同动作/转移/后果/期限/预算/含噪确认，独立精确有限树 | 不调用09的plan/_evaluate/_branches；不读取真实隐状态。
O2 | 每个决策揭示g/z的条件策略期望 | 非现实信息参考，仍付同样确认费用并承受误码；不是O1，也不能部署。
完整机器可读清单configs/comparisons/method_registry.json；不改02历史冻结的baseline_manifest。

四、公平评分与“相同信息”的确切含义
FiniteModel独立实现T(setup)、条件固定窗发射、相关Bayes、终端后果和只计一次的时间。
共享ActionCatalog保证时长/菜单/契约限制一致；保留原08/09源文件哈希，不为基线放宽实际授权。
同信息O1的未知状态是联合belief，不是某个点真值。O2单独记录状态条件动作，不给在线方法单个“完美”答案。
所有方法的remaining_s随动作实际时长扣减。公共剩余诊断horizon耗尽后转合法终端/暂停，最终确认仍付同一秒预算。
一次终端确认拒绝或NO_RESPONSE按共同规则暂停；没有免费无误确认或重复问到同意。
与09相同：窗前T按秒转移、窗内状态固定；无窗内变点和随机延迟核。本步不扩大模型有效性。
固定策略evaluate将“真实评分belief”和“方法内部belief”分开，按原始世界发射和原始后果/时间评分。
因此B2的错误通道假设、消融的来源混用/因子化/去后果/免probe计划费用，都不能篡改评分世界。
各分支记忆深拷贝，伪标签计数不串分支，也不修改06冻结校准模型。
evaluate是小树解析一致性审计，不是收集整集指标的正式实验引擎；完整机制/fault sweep留12。

五、消融清单与执行限制
A_no_probe：删probe，保留任务澄清/共同确认。
A_factorized：预测与每次观察后强行b(g)b(z)，不仅在根打印一个因子化结果。
A_periodic：固定周期替主动选择，周期需充分独立验证调参。
A_no_task_consequence：规划用对称目标0/1损失，实际评分仍是07原后果表。
A_mix_sources：故意把PROBE响应当TASK_QUERY答案，世界仍按真正probe发射。
A_pseudo_labels：拿自己先验预测的任务/通道作标签，给TASK emission累积伪计数；CONFIRM不当完美真值更新。
A_ignore_probe_cost：规划probe费用置0，但实际时间、预算和评分费用仍完整收取。
A_fixed_duration：限制1秒最终窗；不复制短前缀为独立长窗。
所有消融标virtual_only、deployment_allowed=false，保留共同STOP/含噪确认/版本与期限；拒绝execution_mode=live。
错误场景/通道先验、遵从下降等额外环境敏感性列在配置，完整扫描留12，不能写成已跑完。

六、NOIR/RBC/停时/语义VoI来源与适配差异
NOIR作者论文： https://proceedings.mlr.press/v229/zhang23f.html
作者摘要已有用户适应和意图预测。本步B1/B1P为同执行接口的二元交互适配，不能称原NOIR完整复现，也不宣称原文没有上下文。
NOIR2作者预印本元数据核对： https://arxiv.org/abs/2511.20848
RBC作者摘要： https://pubmed.ncbi.nlm.nih.gov/27416602/ （DOI 10.1109/TNSRE.2016.2590959）
其递归Bayes、混淆矩阵与MMI编码是已有基础。单一码本下B10的编码搜索退化；不将该受限实现包装为完整强RBC。
动态停时/刺激选择原文： https://www.imstat.org/publications/aoas/aoas_20_1/aoas_20_1.pdf
作者代码： https://github.com/BangyaoZhao/P300RL
原方法包含早停与动态刺激选择，本文未复现其actor-critic/GP。本步只实现允许最终离散窗之后的置信停止；真实同窗增量停时需条件似然/前缀模型后再扩展。
语义VoI作者预印本： https://arxiv.org/abs/2601.06407
采用既有任务效用与询问成本决策原则，不把Bayes/Bellman/VoI公式说成原创。
核对日期、访问受限处与来源范围见execution/10_references.json。

七、独立验证集调参与冻结（10.8）
当前parameters.json仅开发默认，不是经过验证的最优参数；不可跳过B5/7/8/10调参后挑弱对手。
tuning_and_selection.json声明period/confidence/conflict/quality/channel_entropy搜索集合；真实选优记录和chosen_baseline均为空。
必须先按用户/会话/物理场景分组划development、calibration、validation、test，整episode及其前缀同组。
在独立calibration先导数据与任务要求上确定错误启动率和全episode时间上限，写证据source_ref；未填上限时selector拒绝运行。
然后冻结候选配置清单/哈希，在同一配对validation episode上充分搜索；安全/时限不合格不得入选。
预先锁定排序：全episode意图忠实完成率降序、错误启动率升序、全episode平均时间升序、弃权率升序、candidate_id确定性打破平局。
缺完整候选、跨组泄漏、仅成功episode、formal test split、未冻结先导规则或缺配置哈希均拒绝。
真实入口拒绝simulation_fixture；软件单测可显式simulation_only=True，仅返回simulation_selection_check_only，始终不能部署。
输出记录规则/记录/分区/获选配置哈希；真实返回也只是validation_choice_pending_deployment_audit，不冒称真机通过。
若无合格者不能在看正式结果后放宽门槛；回先导/任务范围重设计，再冻结下一版。
本轮未用这4个解析样例选最强基线，也未执行真实网格调参。

八、实际复现命令与结果
目录D:\\Scene_Aware\\bci_vla_framework：
  python -B execution/run_step10_tests.py
  python -B execution/replay_step10_comparisons.py
169项软件检查PASS；4个fixture、22个方法/消融决策记录（B0显式不可用），12项M/M2/B9同展望oracle核对最大后悔值0。
10_oracle_audit.json保存每候选精确值、O2条件参考、同展望regret和实际perf_counter_ns；单次本机耗时不是延迟benchmark。
10_decision_trace.json保留所有方法的选择/不可用原因，未采集任何人体/机器人样本。
这些结果仅支持实现一致性：目前不能回答“比同样有probe的简单方法更好”成立，更不能得出策略优越或创新充分。
依主计划保留B9和强简单方法，M默认一步；若后续独立实验仍等效则回01/09收窄贡献，不删除强基线或调大惩罚制造差异。
下一步11办理会话、契约授权、fake VLA适配与监督；G2要等12，G3/G4与真实校准仍待外部条件。
'''

ORACLE='''
步骤10 oracle边界与可复现检查
O1：独立ExactOracle树枚举，同信息belief、同合法动作、K_a、后果、时间与含噪确认；horizon最多3，当前M2对照horizon2。
不调用DiagnosticPolicy.plan/_evaluate/_branches；单测mock使它们抛异常仍能求解，避免拿自己验证自己。
共享channel/transition/outcome/catalog是有意保持同模型；不是独立重新估计校准数据或另加oracle信息。
O1不按TV或纯解释结果剪枝，枚举所有可评估合法动作；无模型的重编码/重校准不自动假定成功。
time accounting和固定窗条件与09一致；拒绝/NR暂停，不是完整无限时域POMDP的oracle。
同horizon决策regret=O1对被选动作的精确Q - O1最小Q；不能用M自己的近似估值代替。
同时报告planning_value_difference与实际本机perf_counter_ns。horizon差异单独标注，不当作近似误差。
O2在每个后续决策揭示真实g/z，返回状态条件动作期望，仍保留共同含噪最终确认；非现实信息参考、execution_allowed=false。
4个手写fixture：alias、known_normal、mixed_prior、no_budget；不是统计样本、验证集或真实任务。
12项同展望核对最大决策regret为0；只证明这些小例的一致性，不证明任意状态最优、实时可用或两步优越。
原始结果execution/10_oracle_audit.json；代码bci_vla/baselines/oracle.py。
'''

ACCEPTANCE='''
步骤10离线首版验收；日期2026-10-16；授权“步骤10实行”。
10.1 注册B0—B10、M、O1/O2，另B1P/M2；输入/适配/缺外部条件逐项写矩阵，不覆盖02冻结记录。
10.2 实现联合无probe、固定周期、冲突重复、质量/冲突阈值、参数信息增益、标准一步VoI；真实LLM B0缺记录明确unavailable。
10.3 同模型/词表/动作registry/校准/预算/确认/STOP；刻意无probe或错误模型限制显式记录，真实评分世界独立。
10.4 B1P提供公共预测核验/纠正且仍需含噪确认；B10为RBC/离散最终窗停止/语义VoI受限组合，注明非完整原文复现和码本限制。
10.5 独立O1有限树、O2揭示g/z非现实参考；12项同展望检查regret0，实际CPU计时可对照，不作实时benchmark。
10.6 八种机制消融，其中七种主要求均有：去probe、因子化、周期、去后果、来源混用、伪标签适配、忽略probe费；另固定时长。
10.7 所有消融offline/virtual-only，不放宽真实确认/停止/预算/版本；live调用拒绝。
10.8 验证选优函数与待填先导界限/网格/分组冻结规则；禁止formal test、泄漏组、缺候选或非配对episode；真实最强基线未选，需真实先导/验证集。
169项软件测试PASS（新增43+旧126），0失败/错误/跳过；4个手写fixture输出22方法决策，其中B0保持不可用。
结果只支持软件/算术一致性，未证明优于同样有probe的一步VoI；默认一步，不夸大标准算法创新，不删除强基线。
实际命令run_step10_tests.py和replay_step10_comparisons.py；09旧文件/结果哈希保留，09阶段断言只允许进度继续并另存回归。
本轮未训练、调用云/LLM、连接服务器、采EEG、操作VLA/机器人或跑正式实验。
进度10标offline_comparison_implementations完成，11未开始；真实校准/网格验证/完整NOIR-RBC系统重现仍待条件。
G1条件性状态不变，G2—G4未开始，I01—I14保留。
'''

if __name__=='__main__':
    tests=read('execution/10_test_result.json'); audit=read('execution/10_oracle_audit.json')
    assert tests['result']=='PASS' and tests['tests_run']==169 and audit['max_decision_regret']<=1e-10
    txt('docs/10_comparison_matrix.txt',COMPARISON)
    txt('docs/10_oracle.txt',ORACLE)
    txt('execution/10_acceptance.txt',ACCEPTANCE)
    txt('README.txt','''
BCI—VLA框架V2；更新2026-10-16。
01—05已完成；06通道模拟、07后果、08联合belief、09诊断、10比较基线离线首版完成；11未开始。
最新说明docs/10_comparison_matrix.txt、docs/10_oracle.txt；实现bci_vla/baselines/；配置configs/comparisons/。
B0—B10/M/O1/O2逐项登记，B1P预测纠正、B10受限RBC/停时/语义VoI、8种消融；B0原LLM链缺真实记录明确不可用。
169项软件测试通过：python -B execution/run_step10_tests.py；小例核对replay_step10_comparisons.py。
4个解析fixture、12项同信息同展望oracle检查regret0；不是性能实验，不证明比B9强，M保留默认一步。
真实强基线选优尚需独立先导与验证集；不得用formal test选优；配置中的chosen_baseline保持空。
仅simulation_only，不训练、不采EEG、不调用云/VLA/机器人；完整NOIR/RBC复现、真实模型/耗时/能力均未验证。
09及以前输出保持冻结；统一回归仅用最新10 runner。11再接会话授权/fake适配，G1条件性通过，G2—G4未开始。
主计划D:\\Scene_Aware\\BCI_VLA通用交互框架_逐步执行手册.txt；进度execution/progress.json，交接execution/handoff.txt。
''')
    txt('execution/handoff.txt','''
交接：步骤10离线强基线/消融/oracle首版完成；步骤11未开始
日期2026-10-16；先读docs/10_comparison_matrix.txt、10_oracle.txt、execution/10_acceptance.txt及progress.json。
核心bci_vla/baselines/finite_model/methods/oracle/selection；configs/comparisons包含完整方法、八种消融、开发参数、待验证网格。
169项软件测试PASS（43新+126旧）；4fixture、22方法决策记录，12项同展望oracle核对regret0。
O1独立枚举不调用09 solver；共享channel/transition/outcome/catalog，仍是同模型条件固定窗和共同含噪确认。O2非现实状态条件参考。
M/B9当前同模型一步等效，无算法优势；M2未显示额外作用，不调高惩罚/删除强基线来制造差异。
B0原LLM缺外部调用/耗时记录，明确unavailable；不要填假预测。NOIR适配不冒充全系统复现。
B10仅单一验证二元码本/最终窗置信停止，没有完整MMI新编码、窗内动态停时或actor-critic；扩展前不能把它称原文完整强基线。
B2/B3/消融有独立内部belief更新；评分始终原世界/后果/时间，伪标签分支计数不污染06模型。
ComparisonSuite.observe是合成离散窗口桥，不处理真实原始事件、不替代05/08去重/来源/时钟验证。
选择器要求完整配对validation候选网格、分组无泄漏、校准先导界限/配置哈希，拒formal test；真实chosen_baseline为空，未做验证集选优。
下一步11按11.1—11.12做contracts/session、fake/dry-run VLA适配、执行监督、确认消费一次及STOP反馈；默认不导入设备SDK、不下发机器人。
11需核后端checkpoint/动作维度单位坐标/归一化等，不把HTTP返回当任务成功，未知提交不可自动重发。
确认accept_symbol在09配置为0，必须绑定实际问题/contract_id/revision及有效期；不拿模拟接受分支当用户确认或授权。
通道/后果/时间真实校准与I01—I14仍待条件；G1条件性，G2—G4未开始。完整故障/episode验收在12，本轮未代办。
用户未要求前不自动推进11。统一检查仅run_step10_tests.py；旧05—09输出/manifest冻结。
''')
    log=(ROOT/'execution/worklog.txt').read_text(encoding='utf-8-sig')
    txt('execution/worklog.txt',log+'''

工作记录：2026-10-16，步骤10
授权“步骤10实行”；读取V2步骤10、核心方案方法矩阵、09交接/模型/配置，未发现工作区AGENTS。
实现比较有限核、B1—B10及B1P/M/M2、八种消融、独立同信息O1和全知O2参考、验证集选优与冻结检查。
B0原LLM只保留缺外部记录/实测耗时状态，未调用服务或伪造输出。RBC/NOIR/VoI与停时来源经作者/出版页面核对，适配限制写明。
调试中发现receding一步选择需收到剩余全局horizon，否则会在预算末尾多问；已统一终端位置并核对固定策略真实评分等于一步枚举。
首轮167测试一项失败：固定规则相同损失按action_id选择EXIT，现与公共规则统一同价PAUSE优先。
另修正belief视图advance(0)误清retry/extension许可，增加相同合法动作回归；B1P也检查一致预测可省纠正、但不省最终确认。
最终169项测试全过（43新+126旧），无失败/错误/跳过；运行replay_step10_comparisons得到4fixture、12同展望oracle核对regret0与本机实际CPU时间。
固定小例仍不支持两步优于一步；保留简单VoI默认与算法主张收窄边界。
真实校准先导界限、充分验证调参、最强可部署baseline和完整NOIR/RBC复现均未具备，配置选择保持空；G2—G4未开始。
原05—09源结果不改，09 verifier只改进度断言并另存历史回归。无训练/EEG/服务器/云/机器人调用，无正式实验。
10仅离线比较实现完成，11未开始；I01—I14保留。
''')
    outputs=['bci_vla/baselines/__init__.py','bci_vla/baselines/finite_model.py','bci_vla/baselines/methods.py',
        'bci_vla/baselines/oracle.py','bci_vla/baselines/selection.py','tests/test_comparison_baselines.py',
        'configs/comparisons/method_registry.json','configs/comparisons/parameters.json',
        'configs/comparisons/tuning_and_selection.json','configs/comparisons/ablations.json',
        'docs/10_comparison_matrix.txt','docs/10_oracle.txt','execution/10_acceptance.txt','execution/10_references.json',
        'execution/10_test_output.txt','execution/10_test_result.json','execution/10_decision_trace.json',
        'execution/10_oracle_audit.json','execution/run_step10_tests.py','execution/replay_step10_comparisons.py',
        'execution/prepare_step10_assets.py','execution/finalize_step10.py']
    progress=read('execution/progress.json')
    progress.update(updated_at='2026-10-16',authorization='步骤10实行',current_step='11',current_step_status='未开始',last_completed_step='10')
    progress['steps'][9].update(status='已完成',execution_scope='offline_comparison_implementations',
        changed_files=outputs+['README.txt','execution/handoff.txt','execution/worklog.txt','execution/progress.json',
            'execution/verify_step09_delivery.py','execution/verify_step10_delivery.py'],
        checks=['169 software tests PASS (43 new + 126 previous)',
            '4 handwritten fixtures / 12 same-horizon oracle checks, max regret 0',
            'real validation selection deliberately unfilled'],
        evidence_paths=['execution/10_acceptance.txt','execution/10_test_result.json','execution/10_test_output.txt',
            'execution/10_oracle_audit.json','execution/10_decision_trace.json','execution/10_source_manifest.json',
            'execution/10_integrity_result.json','execution/09_integrity_recheck_after_step10.json'],
        unresolved=['真实通道/后果/耗时/执行能力校准与I01—I14仍待条件',
            'B0缺外部LLM调用及完整实测时间，不填假结果',
            'B1/B1P/B10仅明确适配，完整NOIR/RBC/动态停时未复现',
            '先导上限、充分独立验证调参和最强可部署基线仍待数据',
            '4解析例非性能证据；同样有probe的VoI目前等效，不扩大算法主张'],
        real_validation={'status':'待外部条件','chosen_baseline':None,'deployment_allowed':False},
        next_action='用户要求时执行11：会话/契约/fake VLA适配/监督；不自动接真机')
    write('execution/progress.json',progress)
    inputs=[ROOT.parent/'BCI_VLA通用交互框架_逐步执行手册.txt',ROOT.parent/'BCI_VLA框架_主动诊断核心方案_V2.txt']+[
        ROOT/p for p in ['bci_vla/core/diagnostic_policy.py','bci_vla/core/diagnostic_actions.py',
        'bci_vla/core/joint_belief.py','bci_vla/models/channel.py','bci_vla/models/consequence.py','bci_vla/protocols.py',
        'configs/diagnosis.json','configs/channels/simulation_binary_v1.json','configs/belief/simulation_canonical_transition.json',
        'configs/consequences/simulation_canonical_outcomes.json','configs/baseline_manifest.json',
        'examples/diagnosis/state.json','docs/09_policy.txt','execution/09_source_manifest.json']]
    entries=[{'path':str(path),'kind':kind,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        for kind,paths in [('input',inputs),('software_or_analytic_fixture',[ROOT/s for s in outputs])] for path in paths]
    write('execution/10_source_manifest.json',{'date':'2026-10-16','plan_version':'V2','step_id':'10',
        'scope':'offline_comparison_implementations','files':entries})
    print('Step10 local TXT, progress and source snapshot written; delivery verification remains.')
