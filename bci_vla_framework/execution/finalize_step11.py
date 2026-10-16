"""Write offline step11 handoff and freeze its new source/evidence files."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def write(p,v):(ROOT/p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def txt(p,s):(ROOT/p).write_bytes(b'\xef\xbb\xbf'+s.strip().replace('\r\n','\n').replace('\n','\r\n').encode('utf-8')+b'\r\n')


def main():
    t=read('execution/11_test_result.json');replay=read('execution/11_replay_result.json');trace=read('execution/11_lifecycle_trace.json')
    assert t['result']=='PASS' and t['failures']==t['errors']==t['skipped']==0 and t['tests_run']>=291
    assert replay['result']=='PASS' and all(c['exact_state_match'] for c in replay['cases'])
    assert trace['legal']['final_state']=='SUCCEEDED' and trace['unknown_stop']['new_id_resubmit_rejected']
    for doc in ['docs/11_execution_contract.txt','docs/11_pi05_interface_checklist.txt']:assert (ROOT/doc).is_file()
    count=t['tests_run']
    txt('execution/11_acceptance.txt',f'''
步骤11验收；V2；2026-10-16；用户授权“完成步骤11”。
范围：offline_fake_lifecycle；真实BCI/π0.5/物理控制能力仍待外部条件。
11.1 ExecutionContract绑定完整目标、实例/场景/任务版本、预置条件、授权、独立完成谓词、中止与期限；TASK澄清不能授权。
11.2 InteractionSession覆盖收集、澄清、探测、待确认、待授权、已授权、已提交/结果未知、执行、验证、暂停、成功、失败、取消、过期。
11.3 PROBE有独立来源/显示回执/遵从模型；pending时冻结下发，无contract，探测标签不渲染成机器人指令。
11.4 版本/场景/模型变更撤销授权，执行中先请求停止；同版本确认只尝试一次；确认与授权只消费一次；同提交相同payload缓存，改ID/body拒绝重发。
11.5 已核本地02旧源码证据，逐列真实checkpoint、图像、state、normalization、动作维度/单位/坐标和RTC未核项；不改VLA主干。详11_pi05_interface_checklist.txt。
11.6 Fake/DryRun先验图像/state/action结构与版本、NaN/越界/单位/坐标；无网络、设备SDK、硬件连接。fixture不是实际π0.5规格。
11.7 当前只有离线监督器的版本/时效/一次消费/锁存规则实际执行；自然语言约束不提供物理硬保证；无实际控制器限位/碰撞/平滑/RTC实现。
11.8 日志区分BCI收证、策略返回、授权、提交意图、接收ACK、模拟STARTED/COMPLETED和终态；ACCEPTED仅到SUBMITTED，COMPLETED仅到VERIFYING。
11.9 EndpointEvidence独立判定，绑定session/submission/contract/revision/scene及完成反馈后且当前时刻前的时钟区间；谓词缺失unknown，旧轨迹/跨绑定拒绝；VLA失败不更新意图真值。
11.10 STOP锁存，请求后仍REQUESTED，收到显式模拟STOPPED才SIMULATED_STOPPED；物理停止始终未验证；提交/连接未知不自动重发。
11.11 EventJournal保存严格顺序/哈希链；回放重建BCI校验/联合Bayes/策略和生命周期，逐记录比快照，禁止构造/调用后端；拒绝原始输入未留日志，仅核无状态变化。
11.12 AuthorizationRecord保存实际含噪q/r、完整版本/模型hash、有效期、确认belief与授权预测belief；下发时再次预测延迟且不修改原filter；禁止反复问到同意。错误确认可在独立评价中计wrong_start。
实际：{count}项软件测试PASS，无失败/错误/跳过，含历史169项；两个手写离线生命周期日志逐步重放一致，回放后端调用0。
演示用显式已知left/normal先验、固定转移及手写端点，仅验协议。未做真实成功率、性能/创新优越性或步骤12机制实验。
0.8概率与2秒时效为未调优开发示例。真实校准/I01—I14/精确checkpoint等仍待条件；G1条件性不变，G2—G4未开始。
当前内存去重/日志是审计功能，无跨进程持久exactly-once提交保障；崩溃后不可自动恢复下发。host白名单/哈希不等于发送者认证或物理证明。
执行命令：python -B execution/run_step11_tests.py；python -B execution/replay_step11_session.py；python -B execution/verify_step11_delivery.py。
10及更早冻结文件/测试结果不重跑覆盖；10 verifier只调整进度条件与另存历史重查。
步骤11离线实现完成，步骤12未开始；本轮没有训练、云调用、EEG采集或机器人操作。
''')
    txt('README.txt',f'''
BCI—VLA通用交互框架V2；更新2026-10-16。
01—10既有离线阶段完成；11契约/会话状态机/fake执行/监督和事件回放已完成；12未开始。
最新说明docs/11_execution_contract.txt；真实接口缺项docs/11_pi05_interface_checklist.txt。
会话入口bci_vla/contracts/session.py；不可变契约/授权/端点records.py；纯事件回放replay.py。
VLA仅FakeVLAAdapter/DryRunVLAAdapter；监控日志monitoring/journal.py；配置configs/execution/。
{count}项软件测试PASS：python -B execution/run_step11_tests.py。不要运行旧runner覆盖冻结结果。
本地演示python -B execution/replay_step11_session.py：含TASK/PROBE/含噪确认→授权→fake提交→显式模拟执行→独立端点，以及未知提交/停止分支。
输出execution/11_lifecycle_trace.json、11_session_events.jsonl、11_unknown_stop_events.jsonl、11_replay_result.json。
ACK不算执行，COMPLETED不算目标成功，STOP请求不算物理停止；旧确认/跨会话端点/未知提交重发均拒绝。
只读预测计算收证到授权/下发的延迟；不会重复推进原始证据filter。含噪确认不是正确意图真值。
这是一套有限二元目标/通道的离线框架实现，非真实可部署脑控机器人；示例先验/阈值/时效/端点不作实证。
π0.5真实checkpoint、图像/state/action/归一化/RTC与硬件停止仍未验证；真实校准、I01—I14未解决。
G1保持条件性通过，G2—G4未开始；本轮未训练、连接服务器、使用云或操作机器人。崩溃恢复只审计，不自动重发动作。
主计划D:\\Scene_Aware\\BCI_VLA通用交互框架_逐步执行手册.txt；进度execution/progress.json；交接execution/handoff.txt。
''')
    txt('execution/handoff.txt',f'''
交接：步骤11离线契约/状态机/执行适配及监督完成；步骤12未开始
日期2026-10-16；先读docs/11_execution_contract.txt、11_pi05_interface_checklist.txt、11_acceptance.txt与progress.json。
当前{count}项软件检查通过（旧169项保留）；测试/演示命令见README。真实π0.5及BCI仍不接入。
会话只允许Fake/DryRun精确类型，physical_execution_allowed、execution_allowed和physical_stop_verified均false。
TaskContract仍是05冻结协议；新增ExecutionContract包裹场景指纹、backend规格hash、完成/中止。原05—10核心模型/结果未改。
CONFIRM来自实际绑定离线观测且accept_symbol由请求语义明确为0；不能借模拟接受分支、score-only、默认接受或评价端真值授权。
同完整context只尝试一次最终确认，拒绝/超时后不得换contract_id重复问至同意。任务/场景/码本/模型换代须显式重建，并有真实新任务依据，不能借新Session规避规则。
授权冻结实际q/r、确认belief与模型hash；authorize/dispatch均用只读T(收证至当前)重算概率；原filter不推进避免下一证据双计。
单Session最多一次提交；同id/body重复返回缓存，换id/body拒绝；UNKNOWN/异常永不自动重发。当前内存ledger不是跨进程事务保障；崩溃日志只审计。
PROBE没有可执行contract，pending期间冻结下发，不把probe提示目标写进语言任务。
接受ACK→SUBMITTED；显式fake STARTED→EXECUTING；COMPLETED→VERIFYING；EndpointEvidence独立判定才可SUCCEEDED。
EndpointEvidence精确绑定session/submission/contract版本/scene与完成反馈之后、当前之前的保守时间区间；原校准trajectory不能原样当新任务终态。
STOP与取消锁存先PAUSED/REQUESTED，模拟STOPPED才SIMULATED_STOPPED；不能声称物理停止。后端失败不反向训练意图解码器。
replay.py从公开事件重建校验器/联合filter/策略和生命周期，逐快照比较，不导入/构造/调用后端。hash链不提供身份认证，外部执行反馈依然只是声明。
日志不保存拒绝原始内容，回放只能核其公开状态未改变；完整性还需外部保留终端锚，合法前缀不能证明一整次运行完整。
演示两个手写fixture：已知left/normal先验+固定转移，用于格式与生命周期验收，不是系统性能实验。开发阈值0.8/2s未校准。
现有真实接口证据只是02旧源码，真实checkpoint/归一化/图像/state/action/RTC/停止材料未齐，不得把fixture改开关冒充部署。
步骤12再按主计划做完整episode/故障/机制验收与G2，先保留强B9等同信息基线；当前不宣称任务VoI算法优越。
真实通道/遵从/后果/耗时与I01—I14仍待条件；G2—G4未开始；本轮无训练、服务器、云或硬件调用。
''')
    log=(ROOT/'execution/worklog.txt').read_text(encoding='utf-8-sig')
    marker='工作记录：2026-10-16，步骤11'
    if marker not in log:
        txt('execution/worklog.txt',log+f'''

{marker}
授权“完成步骤11”；依11.1—11.12实现离线ExecutionContract/AuthorizationRecord/EndpointEvidence、InteractionSession、Fake/DryRun适配和事件日志回放。
并行独立审查发现旧终态可能混用、授权记录验证不足、延迟概率未推进和仅恢复快照回放的缺口；已加精确提交/版本/场景/时间绑定、严格实际q/r与model hash检查、只读CTMC授权/下发预测及真正事件reducer。
{count}项软件测试PASS、无跳过；两条手写离线生命周期重放逐步一致，禁止backend构造/提交/query/stop的回放检查通过。
演示成功链区分收证、授权、ACK、模拟执行和独立端点；未知提交不会重发，STOP请求与显式模拟STOPPED分开。
本地核02历史资料并写真实π0.5待核清单；没有选/加载真实checkpoint、修改主干、联网或连接设备。
真实校准和接口、跨进程持久提交/身份认证/物理执行或停止不在当前完成证据中，G2—G4仍未开始。
只更新10 verifier进度断言并另存10历史回归，不改早期source/test结果；11离线首版完成，12未开始。
''')
    outputs=[str(p.relative_to(ROOT)).replace('\\','/') for folder in ['bci_vla/contracts','bci_vla/adapters/vla','bci_vla/monitoring']
        for p in (ROOT/folder).glob('*.py')]+[
        'configs/execution/session.json','configs/execution/fake_vla_interface.json',
        'tests/test_contract_records.py','tests/test_execution_session.py','tests/test_execution_binding_regressions.py',
        'tests/test_execution_journal.py','tests/test_session_replay_events.py','tests/test_vla_adapter.py',
        'docs/11_execution_contract.txt','docs/11_pi05_interface_checklist.txt','execution/11_acceptance.txt',
        'execution/run_step11_tests.py','execution/replay_step11_session.py','execution/finalize_step11.py',
        'execution/11_test_output.txt','execution/11_test_result.json','execution/11_lifecycle_trace.json',
        'execution/11_session_events.jsonl','execution/11_unknown_stop_events.jsonl','execution/11_replay_result.json']
    p=read('execution/progress.json');p.update(updated_at='2026-10-16',authorization='完成步骤11',
        current_step='12',current_step_status='未开始',last_completed_step='11')
    p['steps'][10].update(status='已完成',execution_scope='offline_fake_lifecycle',
        changed_files=outputs+['README.txt','execution/handoff.txt','execution/worklog.txt','execution/progress.json',
            'execution/11_source_manifest.json','execution/verify_step10_delivery.py','execution/verify_step11_delivery.py'],
        checks=[f'{count} software tests PASS; 0 failures/errors/skips; includes 169 prior checks',
            'two handwritten lifecycle journals rebuild each checkpoint without backend calls',
            '11.1-11.12 mapped; real pi0.5 interface remains unverified'],
        evidence_paths=['docs/11_execution_contract.txt','docs/11_pi05_interface_checklist.txt',
            'execution/11_acceptance.txt','execution/11_test_result.json','execution/11_test_output.txt',
            'execution/11_lifecycle_trace.json','execution/11_replay_result.json','execution/11_source_manifest.json',
            'execution/11_integrity_result.json','execution/10_integrity_recheck_after_step11.json'],
        unresolved=['真实通道/遵从/后果/耗时校准与I01—I14仍待条件',
            '真实π0.5 checkpoint/观测/归一化/action/RTC与物理控制及停止未核',
            '内存去重及日志审计不具备跨进程exactly-once提交或发送者认证',
            '开发概率/时效阈值未校准；仅有限二元模拟与稳定窗模型',
            '完整episode/机制和故障实验及G2留12，G3/G4未开始'],
        real_validation={'status':'待外部条件','deployment_allowed':False,'hardware_calls':0,'cloud_calls':0},
        next_action='用户要求后执行12：离线完整episode/机制及故障验收；不自动接真机或宣称G2通过')
    write('execution/progress.json',p)
    inputs=[ROOT.parent/'BCI_VLA通用交互框架_逐步执行手册.txt',ROOT.parent/'BCI_VLA框架_主动诊断核心方案_V2.txt',
        ROOT/'execution/10_source_manifest.json',ROOT/'execution/verify_step10_delivery.py']
    entries=[{'path':str(path),'kind':kind,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        for kind,paths in [('input',inputs),('offline_software_or_fixture',[ROOT/s for s in sorted(outputs)])] for path in paths]
    write('execution/11_source_manifest.json',{'date':'2026-10-16','plan_version':'V2','step_id':'11',
        'scope':'offline_fake_lifecycle','files':entries})
    print(f'Step11 offline delivery frozen: {count} software tests; next step12 unstarted.')


if __name__=='__main__':main()
