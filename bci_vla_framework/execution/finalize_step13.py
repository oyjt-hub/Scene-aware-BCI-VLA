"""Deliver the non-real-BCI branch, retaining the unvalidated real G3 scope."""
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parents[1]
RUN='runs/pilot_virtual_13_nonreal_acceptance_v3'
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def write(p,v):(ROOT/p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def txt(p,s):(ROOT/p).write_bytes(b'\xef\xbb\xbf'+s.strip().replace('\r\n','\n').replace('\n','\r\n').encode('utf-8')+b'\r\n')
tests=read('execution/13_test_result.json');assert tests['result']=='PASS'
assert read(RUN+'/summary.json')['result']=='PASS'
txt('docs/13_pilot_protocol.txt','''步骤13：非真实BCI测试交付协议（2026-10-16）
当前授权：用户明确要求“完成除真实脑机以外的其他测试”。本阶段不再连接8712、不启动旧刺激程序、不使用机器人或云端。
主手册13的真实在线试运行另列待测；本次完成其可独立执行的软件分支。G2研究效用仍未通过，G3未验证。

13.1 三模式呈现：已实现离线呈现模型，使用真实协议Event校验TASK_QUERY、CHANNEL_PROBE、CONFIRM请求。
请求与session/task/scene/menu/channel版本绑定，过期、未来和旧菜单请求拒绝；probe明确显示不下发任务。
静态PNG只用于审查文案与流程，没有实际闪烁显示；并非已部署的SSVEP界面。

13.2 软件采样与时序：接收器支持任意字节分包、残包保留、非有限值计数、接收期限及连接失败。
模拟服务器仅使用127.0.0.1系统分配端口，明确禁止将8712当模拟端口。
最终测试1000帧×12通道逐字节一致；空流超时及拒绝连接都保留报告，不构造选择。
信号处理按原始采样率切窗、滤波、只重采样一次；4000@1000Hz→1000@250Hz，已250Hz输入不再次4倍抽取。
合成12Hz保频和180Hz→70Hz混叠分量抑制通过测试，仅适用于该软件fixture，不是完整滤波或解码质量验证。
软件帧序列与时间戳检查能检出注入的长帧间隔；60Hz及10/15Hz均为测试常量，不是实测刷新率或现场刺激参数。
实际显示时序、触发对齐、硬件时钟和通道身份仍未验证；I01不标为完全解决。

13.3 校准与迁移：完成记录格式、独立标签、TASK/PROBE/CONFIRM分组和适用条件检查。
按用户/会话/菜单/解码版本/采样率/通道/时长/校准条件保留分组，拒绝重叠样本及原始记录组跨split泄漏。
未知遵从不伪造意图标签，probe提示不当作独立理解/遵从真值；有效probe计数不能单独分解C_probe与遵从。
5条显式synthetic_fixture记录完成CLI检查，包括真无响应、超时和未知标签；真实校准、迁移误差、遵从均未采集或拟合。

13.4 全程时间：9类阶段的合成记录可完整计账；缺阶段时最终用户耗时为null。
重叠阶段按区间并集计算，不双倍收费。真实阅读、切换、刺激、解码、确认和重试耗时未测。
本次没有用玩具时间比较策略优劣，真实固定probe/主动策略对照仍待真实pilot。

13.5—13.7 条件与扰动：延用步骤12R冻结的10条件、1440合成流程、机制和故障证据，不改写为真实跨会话结果。
旧证据覆盖稳定/改目标/改通道/两者改变/别名条件；所有合成变化都标为合成，不推断生理原因。
本次新增确定性三模式集成流程用于生命周期验收，固定高置信先验，不估计诊断准确率或过度诊断率。
TASK→PROBE→规划提议→CONFIRM→原会话授权→Fake VLA→STARTED/COMPLETED→独立虚拟终点验证，最终SUCCEEDED。
完整journal在禁用后端构造、提交、查询、停止的条件下重放并逐检查点匹配。
策略仍未证明优于同信息B9，真实新目标误拒绝、低变化频率开销及现实效用未测。

13.8 数据边界：所有新生成记录明确synthetic_fixture，真实pilot为0，不混入正式数据。
runs/pilot_virtual_13_preparation内保留未采集manifest，真实准确率/遵从/迁移/用户时间都为null。
前期用户允许时曾做一次接收检查；最新指令后的非真实测试硬件调用为0。前期原始流不是带标签脑机试次。

运行与审查：
1. 读docs/13_G3_report.txt、execution/13_coverage.json和13_test_result.json。
2. 只读查看runs/pilot_virtual_13_nonreal_acceptance_v3/的summary、virtual_flow、session_journal和ui_preview.png。
3. 核execution/13_source_manifest.json及13_integrity_result.json。
4. 单元回归入口python -B execution/run_step13_tests.py；会覆盖当前13测试结果，交付冻结后优先只跑验证脚本。
5. 重新跑非真实验收必须新目录：python -B execution/run_step13_nonreal.py --output runs/pilot_virtual_13_nonreal_NEW。
6. 不运行capture_step13_stream.py默认真实端口、不启动D:\\脑机目录原程序；本轮范围已排除真实BCI。
''')
txt('docs/13_G3_report.txt',f'''步骤13非真实脑机部分验收报告（2026-10-16）
结论：用户指定的非真实BCI软件分支已完成；真实在线BCI试运行未开展，G3未验证，步骤13整体不标完成。

实际通过：
1. {tests['tests_run']}项完整软件回归，0失败、0错误、0跳过（包含冻结旧测试）。
2. 三项本机模拟TCP集成：分包1000帧×12通道与源字节完全一致；空流超时；连接失败。
3. TASK/PROBE/CONFIRM呈现、上下文与时效校验；合成帧序列、长间隔检测、静态中文版界面预览。
4. 源时基切窗和单次重采样，合成信号保频及抗混叠检查。
5. 三模式→原会话授权→Fake VLA→独立虚拟终点，最终SUCCEEDED，日志逐检查点独立重放通过。
6. 5条显式合成记录经过CLI，保留未知遵从/无响应/超时；校准与holdout泄漏、重叠窗口和独立标签规则测试通过。
7. 9类时间分解的完整与缺失合成记录检查，缺失项不写0。

发现及处理：
接收空流时，单次recv等待可跨设定截止时间；现按剩余时间设置timeout，复测通过。
帧亮度周期边界存在浮点舍入1级差异，现规范相位及量化，周期一致性测试通过。
高频软件滤波测试观察到慢端点瞬态；验收直接检查70Hz混叠分量，不把全段标准差当混叠。
该零相位离线滤波路径不等于在线低延迟实现，不能据此声称滤波质量/脑电解码已验证。
首次nonreal_acceptance目录保留超时测试失败现场，v2保留中间结果，v3为最终验收，不将中间结果用于最终结论。

前期设备工程事实（在用户排除真实BCI之前）：
根据本地interface.py/dataServer.py来源声明12通道、1000Hz；设备型号/实际通道/时钟并未独立确认。
曾对本机8712做1次接收检查，收到3080完整帧，无应用数据发送、无刺激、无解码、无机器人动作。
对该文件离线处理得到770@250Hz（3.08s）；按旧代码重复降采样路径反事实得到162样本（若当250Hz为0.648s）。
这是工程原始流及代码问题证据，不是4秒刺激试次或真实有标签pilot。原采集项目未修改。
最新用户指令后：真实设备连接0，刺激呈现0，机器人命令0，云调用0。

仍缺真实证据：probe可理解性/实际遵从、真实解码准确率、C_probe→C_task迁移、跨会话稳定性、实际屏幕时序与完整用户耗时。
上述指标全部null；真实pilot数量0。SSVEPAdapterShell尚无实时呈现/采集实现，当前信念和授权只接受offline_replay。
因此软件PASS不能提升为真实部署就绪。G2仍未通过（M=B9，M2错误增加），G3/G4未通过/未验证。
下一步可继续非真人的软件开发与G2研究方案修订；若恢复真实pilot，需要另行明确现场条件、采集范围和真实校准实现。

主要证据：{RUN}/summary.json、virtual_flow.json、session_journal.jsonl、synthetic_record_summary.json、synthetic_timing.json、ui_preview.png。
来源及结果冻结清单：execution/13_source_manifest.json；交付验证：execution/13_integrity_result.json。
''')
txt('docs/13_data_collection_checklist.txt','''未来恢复真实pilot时的执行清单（本轮不执行）
1. 先读13_G3_report.txt，确认G2研究门槛及真实实现缺口；不凭软件通过直接开展大规模试验。
2. 明确设备型号、通道身份/单位、硬件采样率、显示器实测刷新率、实际界面脚本及版本；不复用旧程序默认值作实测。
3. 在隔离副本对照单次重采样路径，核原始时间戳/触发、切窗、采样数量、滤波延迟和真实显示时序。
4. 实现真实适配器及校准模型加载路径；审查现有offline_replay边界，不能把live数据改标签来绕过验证。
5. 独立确定任务、probe与确认块；实际理解/遵从使用独立标注，未知填UNKNOWN；提示不等于遵从真值。
6. 按用户/会话/原始记录组划分校准、holdout和pilot，先验证无重叠/泄漏；正式测试保持独立。
7. 对原始文件做SHA256，记录通道顺序、时间窗、时钟、菜单/刺激/解码版本及真实校准条件。
8. 采TASK/PROBE/CONFIRM数据，并保留无控制、无响应、无效信号、传输失败和超时，不删除失败试次。
9. 分别拟合和验证任务/探测/确认观测关系；实际遵从不确定时保留区间，不从有效混淆矩阵武断分解。
10. 用独立TASK holdout检验probe到task预测误差及适用条件；不支持时收窄或修改协议。
11. 全程记录阅读、切换、采集、解码、网络、重试、返回、确认、计算，缺测项保留null。
12. 机器人保持虚拟端，对照主动与固定probe；覆盖稳定/单因子/双变化/别名及低变化频率，在线算法不读真值计划。
13. 所有数据标pilot，调参记录单列；没有完整证据时不通过G3，不声明疲劳改善或跨BCI泛化。
当前状态：以上真实采集步骤均未执行；用户已明确本轮不进行真实脑机测试。
''')
coverage={'scope':'nonreal_BCI_software_only','result':'PASS','tests_run':tests['tests_run'],
    'step13_whole_completed':False,'nonreal_tests_completed':True,'real_pilot_completed':False,
    'hardware_calls_after_user_optout':0,'cloud_calls':0,'robot_commands':0,'real_pilot_trials':0,
    'prior_receive_only_stream_connections':1,'G2_passed':False,'G3_passed':False,
    'items':[
        {'plan_item':'13.1','software':'three-mode presentation and static preview PASS','real':'not_measured'},
        {'plan_item':'13.2','software':'mock TCP, single resample, timestamps, contexts, no-response PASS','real':'device clock, channels, screen and trigger not verified'},
        {'plan_item':'13.3','software':'label/group/split validator and synthetic CLI PASS','real':'calibration, compliance and transfer not measured'},
        {'plan_item':'13.4','software':'complete/missing synthetic timing PASS','real':'human timings and paired utility not measured'},
        {'plan_item':'13.5-13.7','software':'new lifecycle PASS; previous12R synthetic conditions retained','real':'cross-session changes and diagnosis utility not measured'},
        {'plan_item':'13.8','software':'explicit synthetic provenance, real pilot empty, leakage checks PASS','real':'pilot not collected'}],
    'evidence_run':RUN,'decoder_accuracy':None,'probe_compliance':None,'probe_to_task_transfer':None,'user_time_s':None}
write('execution/13_coverage.json',coverage)
txt('execution/13_acceptance.txt',f'''步骤13指定范围验收：非真实BCI测试PASS；真实G3未验证。
完整回归{tests['tests_run']}项，模拟TCP3条件通过，虚拟三模式闭环及逐检查点重放通过。
用户最新范围外的真实采集/闪烁/机器人均未开展；前期一次接收工程流单列，不计pilot。
全步骤13仍进行中，不能用本软件验收替代G3或绕过G2。
详见docs/13_G3_report.txt和execution/13_coverage.json。
''')
p=read('execution/progress.json');p.update(updated_at='2026-10-16',authorization='完成除真实脑机以外的其他测试',
    current_step='13',current_step_status='非真实测试完成，真实BCI试运行未开展',last_completed_step='12',
    next_step='13真实pilot待恢复或继续G2研究',next_step_blocked_by='真实BCI已由用户排除；G2研究效用仍未通过')
s=next(s for s in p['steps'] if s['step_id']=='13')
s.update(status='进行中',nonreal_tests_completed=True,real_pilot_completed=False,
    checks=[f'{tests["tests_run"]}软件回归PASS','3模拟TCP条件PASS','三模式FakeVLA闭环与重放PASS'],
    evidence_paths=['docs/13_G3_report.txt','execution/13_coverage.json','execution/13_test_result.json',RUN+'/summary.json'],
    changed_files=['bci_vla/pilot/','tests/test_pilot_preparation.py','tests/test_pilot_software_ui.py','execution/*step13*.py'],
    unresolved=['真实理解/遵从/迁移/解码/显示时序/用户时间未测','真实适配器及校准模型实现待补','G2未通过'],
    next_action='已完成本轮非真实BCI范围；保留真实pilot未开展状态',blocked_by=['G2','real_BCI_excluded_by_user'])
p['research_gates']['G3'].update(status='未验证',nonreal_software_checks='PASS',real_pilot_trials=0,
    evidence_paths=['docs/13_G3_report.txt','execution/13_coverage.json'],verdict='非真实软件分支完成，真实G3未验证')
write('execution/progress.json',p)
txt('README.txt',f'''BCI—VLA框架V2；更新2026-10-16。
最新：步骤13按用户“除真实脑机以外”范围完成软件测试；真实pilot未开展，G3未验证，G2研究效用仍未通过。
先读docs/13_G3_report.txt及13_pilot_protocol.txt。{tests['tests_run']}项完整软件回归PASS，模拟TCP3条件与三模式虚拟执行/日志重放PASS。
最终非真实证据：{RUN}/，ui_preview.png为静态中文版界面预览，未呈现刺激。
仅新隔离采样处理路径实现单次重采样；原D:\\脑机项目未修改。前期1次接收工程流不计真实pilot。
最新指令后不连接8712、不启动旧刺激界面、不接机器人/云；真实准确率、遵从、迁移和用户耗时均未测。
步骤12R工程一致性通过、1440合成流程保留；M与B9等效，M2错误增加，不能宣称研究优越性。
01—12R冻结证据保留，步骤14—17未开展。进度execution/progress.json；交接execution/handoff.txt。
冻结来源execution/13_source_manifest.json；验证python -B execution/verify_step13_delivery.py。
不要运行旧runner覆盖旧证据。非真实验收重跑必须新目录，见13_pilot_protocol.txt。
''')
txt('execution/handoff.txt',f'''交接2026-10-16：步骤13非真实BCI指定范围完成。
当前授权：用户不涂导电膏，要求完成除真实脑机以外的测试。不要继续连接8712或启动真实刺激/采集/机器人。
cwd D:\\Scene_Aware\\bci_vla_framework；Python3.13，numpy/scipy/PyQt5可用，旧程序所需pygame/mne/sklearn在本解释器不可用。
新模块bci_vla/pilot/：记录与采样链验证、浮点流解析、单次重采样、离线呈现模型和确定性虚拟闭环。
{tests['tests_run']}软件测试PASS；3模拟TCP条件PASS；Fake VLA三模式全链路终点成功且纯日志逐检查点重放PASS。
证据{RUN}/；docs/13_G3_report.txt、13_pilot_protocol.txt、13_data_collection_checklist.txt。
原设备代码位于D:\\脑机\\yjj（2）\\yjj，未修改/未启动。最新范围收窄之前曾1次接收3080帧原始流，文件仅工程证据。
raw→250Hz离线新路径770样本；旧代码重复抽取反事实162样本；硬件时钟/通道/触发未独立核实，I01未完全解决。
nonreal_acceptance是开发中超时断言失败现场，v2是中间通过版本，v3最终版；保留前两者，不混用最终证据。
G2研究未通过、G3真实未验证，步骤13整体进行中（非真实分支完成），14—17未开始。
真实profile不可加载，filter/authorizer只接offline_replay，SSVEPAdapterShell未实现真实采集/呈现，不可改source_kind绕过。
来源13_source_manifest；交付验证verify_step13_delivery.py；旧12R复验另存12R_integrity_recheck_after_step13.json，不覆盖旧结果。
所有TXT UTF-8 BOM+CRLF；原01—12R源/结果不要覆盖；只读验证优先，重跑实验用新输出目录。
''')
old=(ROOT/'execution/worklog.txt').read_text(encoding='utf-8-sig')
entry=f'\n2026-10-16 步骤13：用户排除真实脑机后完成非真实软件范围。{tests["tests_run"]}回归、3模拟TCP、采样/时序/记录/计时和三模式FakeVLA重放PASS；真实pilot0，G3未验证。修复recv截止等待与帧周期浮点舍入；原设备代码未改。最终证据{RUN}/。\n'
if '用户排除真实脑机后完成非真实软件范围' not in old:txt('execution/worklog.txt',old+entry)
paths=[]
for pattern in ['bci_vla/pilot/*.py','tests/test_pilot*.py','execution/*step13*.py','execution/13_*.json','execution/13_*.txt',
    'docs/13_*.txt','configs/pilot/*.json','examples/pilot13/*.json','runs/pilot_virtual_13_*/**/*']:
    paths.extend(q for q in ROOT.glob(pattern) if q.is_file())
paths.append(ROOT/'execution/12R_source_manifest.json')
excluded={'execution/13_source_manifest.json','execution/13_integrity_result.json','execution/verify_step13_delivery.py'}
files=[{'path':str(q.resolve()),'sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for q in sorted(set(paths))
    if q.relative_to(ROOT).as_posix() not in excluded and '__pycache__' not in q.parts]
write('execution/13_source_manifest.json',{'scope':'nonreal_step13_delivery','G3_passed':False,'files':files})
print(json.dumps({'files_frozen':len(files),'tests_run':tests['tests_run'],'nonreal_scope':'PASS','G3_passed':False}))
