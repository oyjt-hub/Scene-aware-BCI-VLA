"""Record actual repair/revalidation and narrowed claims; keep G2 honest."""
from pathlib import Path
from statistics import mean
import csv
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/offline_step12_repair_v2'
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def write(p,v):(ROOT/p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def txt(p,s):(ROOT/p).write_bytes(b'\xef\xbb\xbf'+s.strip().replace('\r\n','\n').replace('\n','\r\n').encode('utf-8')+b'\r\n')
def lines(path):return [json.loads(r) for r in path.read_text(encoding='utf-8').splitlines()]


def main():
    summary=read('runs/offline_step12_repair_v2/summary.json');audit=read('runs/offline_step12_repair_v2/oracle_audit.json')
    tests=read('execution/12R_test_result.json');plan=read('runs/offline_step12_repair_v2/evaluation_private/run_plan.json')
    assert summary['episodes']==1440 and summary['authorization_wrapper_blocked_episodes']==0
    assert tests['result']=='PASS' and tests['tests_run']==330 and audit['counterexample_fixed']
    rows=lines(RUN/'evaluation_joined.jsonl');old=lines(ROOT/'runs/offline_step12_acceptance/evaluation_joined.jsonl')
    comparisons=[]
    for method in summary['methods']:
        groups=[('original_0_11',[r for r in old if r['method']==method]),
            ('repaired_0_11',[r for r in rows if r['method']==method and r['seed']<12]),
            ('new_development_12_23',[r for r in rows if r['method']==method and r['seed']>=12])]
        for name,group in groups:
            comparisons.append({'method':method,'group':name,'episodes':len(group),
                **{k:sum(bool(r[k]) for r in group) for k in ['faithful_virtual_completion','wrong_start','unstarted','unresolved']},
                'mean_elapsed_s':mean(r['elapsed_s'] for r in group)})
    write('execution/12R_before_after.json',{'scope':'development_regression_and_new_development_seeds',
        'groups':comparisons,'old_wrapper_blocked':read('runs/offline_step12_acceptance/summary.json')['authorization_wrapper_blocked_episodes'],
        'repaired_wrapper_blocked':summary['authorization_wrapper_blocked_episodes'],'models_or_thresholds_retuned':False})
    with (ROOT/'execution/12R_comparison.csv').open('w',encoding='utf-8-sig',newline='') as output:
        writer=csv.DictWriter(output,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    table='方法 | 原0—11种子 完成/错启/未启动 | 修复后同种子 完成/错启/未启动 | 新12—23种子 完成/错启/未启动\n'
    for m in summary['methods']:
        rs=[r for r in comparisons if r['method']==m]
        table+=m+' | '+' | '.join(f"{r['faithful_virtual_completion']}/{r['wrong_start']}/{r['unstarted']}" for r in rs)+'\n'
    low=next(r for r in audit['gate_cases'] if r['prior']==[.4,0,.6,0] and r['return_s']==.01 and r['fixed_transition'])
    txt('docs/12R_revalidation_report.txt',f'''
步骤12修复及复验报告；日期2026-10-16。
授权：“修复并复验”，随后“继续”。

一、结果
授权终端代价不一致已在新版本authorization-aware-finite/2.0中修复，工程复验通过。
G2总体仍未通过：M与B9仍等效，M2没有在错误启动不增加的前提下证明额外效用；C11的别名负结果仍存在。
本轮没有把工程修复当作新算法创新或G2研究门槛通过，步骤13未开始。
原01—12代码、模型、参数和负结果保持冻结；修复在bci_vla/runtime_v2/，原报告可对照复现。

二、具体修复
1. CONFIRM接受分支先按原CTMC传播到实际授权/下发时刻，再检查同一目标后验>=0.8。
2. 不足门槛时计pause_loss=1和任务未完成，不继续按执行收益估值；实际授权仍由11处理。
3. 确认后的0.01秒返回/授权延迟计入动作总时长、预算、场景/契约有效期和时耗成本，只收一次。
4. 同时检查原2秒确认年龄；超龄、无契约、旧版本、过期、STOP、pending和推断不可用不绕过实际约束。
5. 真实决策计算耗时之后，再用当前公开信念重查动作合法性和接受分支能否通过授权；无新BCI证据或伪造确认。
6. B4/B5/B7/B9/M/M2均使用相同版本和授权规格；没有只修主方法或降低基线的探测能力。

AuthorizationAwarePolicy从原09有限Bellman代码复用并修改接受分支；AuthorizationAwareFiniteModel用独立的predict/marginals/terminal_loss代数供原独立O1枚举。
O1不调用候选confirmation_value或_evaluate；测试将两者设为抛错仍可求解，避免拿同一候选函数验证自身。
这属于标准VoI的工程对齐，不主张新的Bayes/Bellman/CTMC公式。

新控制器是11会话外部的任务提议者。会话保留原授权器和其policy/config指纹；新外部规划器的version/gate/planner_ref单独写入每个公开流程/决策与本轮源hash。
不把11授权记录中的旧policy_sha256宣称为新外部规划器代码指纹，也不改变已冻结的11事件格式或回放规则。
任务、来源、实际含噪q/r、契约/场景版本、一次消费和STOP规则继续由原会话执行。

三、反例与分支核对
原反例仍保留execution/12_runtime_alignment.json。
先验0.4、normal CONFIRM观测accept0后的目标后验0.32/0.41≈0.7804878049，低于0.8。
修复后的候选分支continuation_loss={low['candidate_continuation_loss']:.1f}，独立O1={low['independent_O1_continuation_loss']:.1f}，实际授权拒绝，实际提交仍为0。
该完整确认动作包含2.01秒及0.0402时间代价，接受/拒绝/无响应均不能偷偷变成任务已完成。
29组低/高概率、固定/动态转移、0/0.01/2/2.01秒延迟与实际11授权结果全部一致。
80项8种先验×两套菜单×0/1/2展望的内部O1核对，最大同展望regret={audit['max_same_horizon_regret']:.12g}。
330项软件测试全部通过，无失败/错误/跳过；含原311项及19项本轮修复检查。

四、完整流程复验
新运行目录runs/offline_step12_repair_v2，不覆盖原acceptance/或development/。
10条件×24种子×6方法=1440条完整合成流程，全部保留失败/弃权并逐检查点回放，回放不构造/调用后端。
种子0—11为原开发回归；12—23为新加入开发种子，运行前写入计划，未用于挑阈值、参数或方法。
新增种子仍是development，不称独立正式test、真实受试者或真实probe校准。
依旧包含四格、双稳定偶发误码、错强先验、probe不遵从、无信息probe；同一时间预算和共同确认/停止规则。
原8项机制、10种故障再次执行通过。无信息probe不制造确定性，二元重编码仍不能破别名。
修复后出现“规划后又被实际授权门槛阻断”的流程为0；原回归中为1条。被预先判为不宜授权的流程暂停，并未被删出分母。

每组/方法各120条，以下均为合成描述性计数：
{table}
同一组的成功、错误启动与未启动三项合计120；未解决率另含所有未忠实完成的已启动错误。
原种子下各方法的忠实完成/错误启动计数没有因修复改变；工程修复主要消除了错误的规划估值及无效确认，并非性能提升。
所有24种子的M/B9共有240个配对条件，动作序列和完成/错误启动/未启动结果全部相同。
新24种子混合中M忠实完成50/240、错误启动66/240；M2忠实完成55/240、错误启动73/240。增加完成同时增加错误启动，不能判定M2更好。
上述混合重复加权三个可观察世界相同的C11子条件，逐条件结果与全部分母保存在summary/evaluation_joined；不能据混合率推断真实总体性能。
CPU实际计算另记，模拟setup/acquisition/return/execution和失败/弃权时间完整记录；不把本机小幅耗时差当真实交互效率优势。

五、贡献与G2状态
工程一致性：PASS。原确定性门槛缺口已在新版本消除，独立O1和实际执行规则得到交叉核对。
研究额外效用：未证明。修复没有令M区别于标准一步VoI；不通过改名、删B9、调大惩罚或隐藏C11错误来制造创新。
贡献范围已另写docs/12R_claim_scope.txt：标准VoI作为采用工具，C1限于当前来源分离协议及声明模型的边界，C2额外收益和C3迁移均不写成已成立。
G2总体保持未通过，阻断项更新为研究效用/诊断覆盖与证据，而不是已经修好的授权代价不一致。

六、仍然不支持的主张
有限4状态/窗口内固定假设不变；C11先后变化仍在共同采集窗前，不验证先后顺序或精确变点。
未来任意场景变化、真实网络/调度/硬件异常未被有限oracle穷尽，不称完整实际系统或无限POMDP最优。
现有协议无法保证在无冲突、同时改目标/通道且TASK输出不变时总能发现变化；本轮保留这种失败，不引入私有变化触发器。
真实probe理解、遵从、task迁移、采样、π0.5 checkpoint/接口/RTC/停止仍待外部条件；I01—I14和真实校准不被本轮软件结果解决。
本轮无训练、云/服务器、真实EEG或机器人调用，未进入13。

七、复现与证据
项目D:\\Scene_Aware\\bci_vla_framework：
  python -B execution/run_step12_repair_tests.py
  python -B execution/run_step12_repair.py --output runs/offline_step12_repair_new
  python -B execution/verify_step12_repair.py
公开输入/完整会话日志、评价私有生成器记录、后置评分、29门槛分支/80项O1、原8机制/10故障、运行前源hash与输出hash均保存在repair_v2目录。
源清单execution/12R_source_manifest.json；测试12R_test_result.json；原/新分组计数12R_before_after.json与12R_comparison.csv。
原12证据另存历史完整性回查，不覆盖原报告和结果。当前进度为“待研究效用复验”，不会把G2改成自动通过。
''')
    txt('docs/12R_claim_scope.txt','''
步骤12修复后的贡献范围；2026-10-16；与原负结果和修复报告一起保存。
1. 当前系统：有限二元目标/通道、来源分离的TASK/PROBE/CONFIRM协议，联合信念、采用标准任务VoI、版本化任务契约及fake执行监督。
2. Bayes、CTMC、Bellman、VoI、更多规划深度、belief和wait/act均不单独作为新算法贡献。
3. M与同probe B9等效。方法描述明确为标准一步任务VoI实现；不能只改模块名称或将M2作为默认升级来宣称新策略优势。
4. C1只保留当前条件模型下可复现的任务/通道混淆边界、有效与无信息probe负对照，以及显式输入来源和授权流程。是否具足够论文新颖性仍需近邻及实证支撑，不宣布首个通用框架。
5. C2从“已经取得额外诊断策略收益”收窄为“在交互协议中采用任务后果和时间的标准决策工具”。目前没有超越同信息B9的算法结果，M2的完成与错误启动取舍照实报告。
6. C3跨BCI/VLA真实迁移及用户收益没有新实证，本地多接口/fake格式通过不能替代迁移和真人实验。
7. G2技术子项复验通过，研究价值门槛仍未通过；不能因收窄文案就改G2验收标准或自动进入13。
8. 后续若研究额外协议价值，应先明确要增加的独立证据、可观察时机及真实成本，保留同probe的周期/阈值/标准VoI，按独立开发/验证规则冻结。没有差异就继续收窄，不换指标直到胜出。
9. C11无冲突别名失败、错先验、probe不遵从/无信息、全部弃权和时间保留；不能删除这些条件作为“修复成功”。
''')
    txt('README.txt','''
BCI—VLA框架V2；更新2026-10-16。
最新：步骤12授权模型不一致已在runtime_v2修复并完成复验，工程一致性PASS；G2研究效用仍未通过，13未开始。
先读docs/12R_revalidation_report.txt、12R_claim_scope.txt。原01—12代码/模型/负结果均冻结保留。
新1440条合成流程（原12种子回归+12个新开发种子，10条件×6方法）全部保留/重放；29门槛实测分支、80内部O1核对通过。
330项软件测试PASS：python -B execution/run_step12_repair_tests.py。不要运行旧runner覆盖冻结输出。
新版本bci_vla/runtime_v2/为外部提议规划器，所有方法同一授权门槛0.8/时效2s/返回0.01s；原会话仍掌管真实绑定含噪确认、授权和STOP。
原0.7804878接受分支现在计未完成暂停，不计执行收益；规划后被授权层阻断从旧回归1条变为本轮0条。
M仍与B9全部240组配对动作/结果等效；M2增加完成同时增加错误启动，未证明额外价值，不能把标准VoI当新算法。
证据runs/offline_step12_repair_v2/；分组对照execution/12R_before_after.json和12R_comparison.csv；源execution/12R_source_manifest.json。
重跑须新目录：python -B execution/run_step12_repair.py --output runs/offline_step12_repair_new。
仅software/synthetic/replay；不训练、不接真实EEG/π0.5/机器人，无云或服务器调用；真实校准与I01—I14仍待条件。
进度execution/progress.json；交接execution/handoff.txt。当前停在12研究效用复验，不启动13。
''')
    txt('execution/handoff.txt','''
交接：步骤12授权一致性已修复复验；G2研究效用仍未通过，13未开始
2026-10-16；用户授权“修复并复验”“继续”。先读docs/12R_revalidation_report.txt、12R_claim_scope.txt及progress.json。
新代码bci_vla/runtime_v2/{planning,controller,runner,audit}.py；01—12冻结源不改，原负结果/720流程保留。
工程修复：接受分支预测到下发，检查0.8与2秒；不足按未完成暂停；0.01返回计时/预算/期限；计算后再用最新公开信念校验。所有6方法共享规格。
新模块是外部任务提议者，planner_ref与本轮source hash另存；11仍使用其原授权器/模型指纹，不把旧授权policy_sha冒充新外部规划代码指纹。
候选Bellman分支与独立O1的终端代数各自实现；禁用候选solver/gate函数时O1仍可运行。29门槛分支与实际授权一致，80同展望regret0。
330软件测试PASS；1440完整合成流程（种子0—11回归、12—23新增开发）全部保留、逐检查点回放；授权后置阻断0条，原回归1条。
工程修复未改变原种子的各方法完成/错误启动计数；M仍与B9全部240组配对动作/结果等效。
24种子混合M完成50/240、错启66/240；M2完成55/240、错启73/240，不能将多完成当统一优越。C11别名失败照录。
G2总体未通过，授权缺口已从未解决项移至已解决项；现阻断为额外协议/策略效用与诊断覆盖缺证据。不能靠弱化门槛、删B9或重复确认改通过。
下一研究行动先明确真正要获取的额外证据、其可观察时机和开销，保持强同probe对照；标准VoI不当原创，真实迁移未做。
1440、24种子均为L1开发，不是真人/物理结果、正式测试或已充分调参；先后C11窗前顺序未识别，4状态/稳定窗边界不变。
最新输出runs/offline_step12_repair_v2，原输出acceptance/development保留。统一测试run_step12_repair_tests.py，重跑修复runner必须新--output。
真实采样/理解遵从/校准、checkpoint/接口/RTC/停止及I01—I14继续待条件。本轮无训练/云/服务器/设备操作，不自动进入13。
''')
    progress=read('execution/progress.json');progress.update(authorization='修复并复验；继续',current_step='12',
        current_step_status='待研究效用复验',next_step_blocked_by='G2研究效用未通过；授权模型一致性已修复')
    step=progress['steps'][11];step['engineering_revalidation']='PASS'
    step['active_planner_version']='authorization-aware-finite/2.0'
    step['next_action']='保留G2未通过；按收窄贡献研究额外协议证据与效用，不自动进入13'
    step['unresolved']=['M与B9等效，M2无不增加错误启动的额外效用；研究贡献仍需证据',
        '无冲突C11别名可导致错误启动，未来额外证据/探测时机须公平验证',
        '4状态/稳定窗限制；真实校准/接口/遵从/迁移及I01—I14仍待条件']
    step['checks'] += [r for r in ['330 software checks PASS','29 actual authorization cases and80 corrected O1 checks PASS',
        '1440 synthetic episodes replayed; unexpected authorization blocks after planning0; no threshold tuning'] if r not in step['checks']]
    refs=['docs/12R_revalidation_report.txt','docs/12R_claim_scope.txt','execution/12R_before_after.json',
        'execution/12R_test_result.json','execution/12R_source_manifest.json','execution/12R_integrity_result.json',
        'runs/offline_step12_repair_v2/summary.json','runs/offline_step12_repair_v2/oracle_audit.json']
    step['evidence_paths']+= [q for q in refs if q not in step['evidence_paths']]
    gate=progress['research_gates']['G2'];gate.update(verdict='授权一致性复验通过；额外协议/策略效用未证明',
        engineering_revalidation='PASS',authorization_alignment='resolved_in_external_proposer_v2',
        blockers=step['unresolved'][:2],evidence_paths=step['evidence_paths'],research_acceptance_passed=False)
    gate['resolved_blockers']=['原09/10接受分支与11授权门槛不一致：新版本已修复，原冻结版本保留作负回归']
    gate['status']='未通过'
    progress['research_gates']['G1']['limitations']=[('G2研究门槛未通过；G3—G4未开始' if s=='G2—G4未开始' else s) for s in progress['research_gates']['G1']['limitations']]
    write('execution/progress.json',progress)
    logfile=ROOT/'execution/worklog.txt';log=logfile.read_text(encoding='utf-8-sig');marker='工作记录：2026-10-16，步骤12修复复验'
    if marker not in log:txt('execution/worklog.txt',log+f'''

{marker}
授权修复并复验/继续；未启动子代理。保留01—12冻结代码与全部原负结果，新runtime_v2实现授权感知终端代价、独立O1代数与统一baseline桥。
不调阈值/模型参数，原0.7804878例改计pause1；0.01返回计预算，计算后重查当前公开信念；会话仍掌管实际q/r和授权。
29实际门槛案例一致、80同展望O1差0；19新增+311旧共330软件检查PASS。
新1440流程全保存/重放，原0—11回归+新12—23开发；实际授权后置阻断0，旧回归1。原种子各方法完成/错启计数均不变，未伪造性能改善。
M=B9全部240配对，M2多完成同时多错启；收窄标准VoI算法贡献。G2工程子项PASS，研究效用仍未通过，13未开始。
新12R证据与源清单冻结，原12 verifier仅改进度状态兼容与另存回查；不覆盖原12完整性结果。本轮无训练、云、服务器或设备调用。
''')
    outputs=[str(q.relative_to(ROOT)).replace('\\','/') for q in (ROOT/'bci_vla/runtime_v2').glob('*.py')]+[
        'tests/test_runtime_v2.py','execution/run_step12_repair.py','execution/run_step12_repair_tests.py',
        'execution/finalize_step12_repair.py','execution/12R_test_output.txt','execution/12R_test_result.json',
        'execution/12R_before_after.json','execution/12R_comparison.csv','docs/12R_revalidation_report.txt','docs/12R_claim_scope.txt']
    inputs=[ROOT/'execution/12_source_manifest.json',ROOT/'execution/verify_step12_delivery.py']
    files=inputs+[ROOT/q for q in outputs]+[q for q in RUN.rglob('*') if q.is_file()]
    write('execution/12R_source_manifest.json',{'date':'2026-10-16','scope':'authorization-aware-finite/2.0',
        'engineering_revalidation':'PASS','G2_passed':False,
        'files':[{'path':str(q),'sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for q in sorted(set(files))]})
    print('Repair evidence frozen: engineering PASS, G2 research benefit still unproven,13 unstarted.')


if __name__=='__main__':main()
