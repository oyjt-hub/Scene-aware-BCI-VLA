"""Freeze Step05 engineering evidence without upgrading research gates."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'execution/v4'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def text(p,s):p.write_bytes(('\ufeff'+s.replace('\r\n','\n').replace('\n','\r\n')).encode('utf-8'))
def main():
 assert not (OUT/'05_acceptance.json').exists(),'Preserve frozen acceptance'
 report=read(OUT/'05_verification.json');assert report['status']=='PASS'
 summary=read(OUT/'05_runs/summary.json');now=datetime.now(timezone.utc).isoformat()
 doc=ROOT/'docs/v4/05_fairness_table.txt';text(doc,doc.read_text(encoding='utf-8-sig'))
 handoff='''V4-05交接（2026-10-17）
已完成基线登记、公共询问/带噪回应/时钟/模拟确认提交runner、B0及适用二候选原过滤器参考。
平台为真实LIBERO冻结采集数据回放；本步不推进新动力学，不调用VLA动作，不是机器人任务成功结果。
B0运行32轮，B1_binary_filter_fixed_policy运行16轮；后者实际原JointBeliefFilter成功更新176次。多候选B1明确不适用。
神经B2/B3/B4/Full及同aux对照仅登记，未实现/训练；B2只有完整文本序列化样例，真实token数未测。
登记：configs/v4/baseline_registry.json；说明：docs/v4/05_fairness_table.txt；日志：execution/v4/05_runs。
复验：python execution/v4/verify_step05.py。500项数据/软件完整性检查通过，不是模型能力验收。
原完整DiagnosticPolicy缺匹配后果表，本步使用公共fixed_inquiry；没有伪造NOIR或神经结果。
模拟请求始终execution_authorized=false/VLA_called=false/physical_dispatched=false。STOP不调用策略，PROBE不改确认任务。
当前2秒统一交互计费，壁钟计算耗时另记；正式神经评估前须共同冻结推理延时规则。
下一步06 CPU模块；步骤03高层门槛、步骤04主任务/放置监督缺口保留，不能直接开始08训练。
正式test未解封，真实EEG和实体动作排除。后续实际VLA动力学闭环仍固定在LIBERO。
'''
 text(OUT/'handoff_step05.txt',handoff)
 write(OUT/'05_command_log.json',[{'command':'python execution/v4/prepare_step05.py','purpose':'baseline registration only'},{'command':'python execution/v4/run_step05.py --refresh-unfrozen','purpose':'development B0/actual binary legacy-filter runs; one V2 response TTL issue repaired before final runs'},{'command':'python execution/v4/verify_step05.py','status':'PASS','checks':report['checks']}])
 acceptance={'step_id':'V4-05','status':'PASS_ENGINEERING_SCOPE','completed_at_utc':now,'checks':report['checks'],'methods_run':summary['methods'],'legacy_filter_successful_updates':176,'neural_baselines_registered_not_run':True,'full_legacy_planner_run':False,'VLA_actions_run':False,'simulation_dynamics_advanced':False,'LIBERO_capture_source':True,'training_run':False,'training_data_ready':False,'high_level_gate_passed':False,'formal_test_unsealed':False,'next_step':'V4-06'}
 write(OUT/'05_acceptance.json',acceptance)
 files=list((ROOT/'bci_vla/evaluation_v4').glob('*.py'))+list(OUT.glob('*step05*.py'))+list((OUT/'05_runs').glob('*.json'))+[ROOT/'configs/v4/baseline_registry.json',doc,OUT/'handoff_step05.txt']
 files+=[p for p in OUT.glob('05*.json') if p.name!='05_source_manifest.json']
 files=sorted(set(files))
 write(OUT/'05_source_manifest.json',{'step':'V4-05','files':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]})
 progress=read(ROOT/'execution/v4_plan_progress.json');old=read(OUT/'05_progress_before.json');assert progress['historical_v2_gates']==old['historical_v2_gates'];assert progress['steps'][5:]==old['steps'][5:]
 progress['steps'][4].update(status=progress['steps'][0]['status'],completed_at_utc=now,completion_scope='基线登记/适用规则参考/真实LIBERO采集上的公共交互runner；神经和动作结果未运行',actual_artifacts=['configs/v4/baseline_registry.json','docs/v4/05_fairness_table.txt','execution/v4/05_runs/summary.json','execution/v4/05_verification.json','execution/v4/05_acceptance.json','execution/v4/handoff_step05.txt'],verification={'status':'PASS_ENGINEERING_SCOPE','report':'execution/v4/05_verification.json','checks':report['checks']})
 progress.update(current_stage='V4-05_registry_and_LIBERO_capture_interaction_runner_complete',next_step='V4-06',latest_handoff='execution/v4/handoff_step05.txt',neural_baseline_training_run=False,formal_experiments_run=False,high_level_gate_passed=False,training_data_ready=False,full_VLA_closed_loop_run=False)
 write(ROOT/'execution/v4_plan_progress.json',progress)
 text(ROOT.parent/'BCI_VLA当前计划入口_V4.txt','当前V4-plan-2；2026-10-17。\n步骤01/02完成；03审计完成但高层能力缺口保留；04 LIBERO开发数据完成但训练监督未就绪；05基线登记/适用规则参考/公共交互runner完成。\n下一步06 CPU小型Transformer与匹配结构原型。\n最新交接：D:\\Scene_Aware\\bci_vla_framework\\execution\\v4\\handoff_step05.txt。\n对照说明：D:\\Scene_Aware\\bci_vla_framework\\docs\\v4\\05_fairness_table.txt。\n主计划：D:\\Scene_Aware\\BCI_VLA框架_意图更新与澄清_执行计划_V4.txt。\n后续平台LIBERO；VLA动作未在本步运行，神经基线未训练，正式test封存；真实EEG/实体动作排除。\n')
 print(json.dumps({'status':acceptance['status'],'checks':report['checks'],'frozen_files':len(files),'next_step':'V4-06'}))
if __name__=='__main__':main()
