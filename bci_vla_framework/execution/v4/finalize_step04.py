"""Freeze the real-LIBERO development data; keep training-readiness gaps explicit."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'execution/v4';DATA=ROOT/'data/v4'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def text(p,s):p.write_bytes(('\ufeff'+s.replace('\r\n','\n').replace('\n','\r\n')).encode('utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert not (OUT/'04_acceptance.json').exists(),'Preserve acceptance'
 verification=read(OUT/'04_verification.json');assert verification['status']=='PASS'
 manifest=read(DATA/'data_manifest.json')
 manifest.update(training_data_ready=False,main_task_generation_supervision_ready=False,place_phase_supervision_ready=False,full_VLA_closed_loop_run=False,split_manifest_sha256=sha(DATA/'split_manifest.json'),capture_manifest_sha256=sha(DATA/'captures/capture_manifest.json'),schema_sha256=sha(ROOT/'configs/v4/interfaces_data_4_1.json'),formal_test_unsealed=False)
 files=[p for p in DATA.rglob('*') if p.is_file() and p.name!='data_manifest.json']
 manifest['files']=[{'path':p.relative_to(DATA).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(files)]
 write(DATA/'data_manifest.json',manifest)
 report='''V4-04 LIBERO开发数据状态（2026-10-17）
已完成：实际LIBERO_90开发场景采集、含噪BCI数据生成、公开/私有分离、分组划分及可复验数据读取。
48个底层回合，96张实际渲染图；分区train/validation/calibration_fit/calibration_audit=24/8/8/8。
4224条公开历史快照、4608条私有标签、384组同公开输入反事实对。增强行不是独立机器人回合。
另预留8个test回合，只冻结编号，未渲染、未生成噪声或评价。正式test留到步骤13。
数据入口：D:\\Scene_Aware\\bci_vla_framework\\data\\v4\\data_manifest.json。
详细规则：D:\\Scene_Aware\\bci_vla_framework\\data\\v4\\annotation_rules.txt。
复验命令：python execution/v4/verify_step04.py；222/222项数据/来源检查通过。
当前视觉为实际LIBERO图像+特权场景投影，ROI只用RGB统计，语义特征缺失；并非Florence/SAM检测结果。
当前仿真只复位、稳定并中性步进，没有成功抓取/放置轨迹。真实脑机未采集，未训练模型，未运行完整VLA任务闭环。
训练级缺口：歧义意图标签保守；主任务重述不等于主任务proposal监督；缺放置阶段真实帧与标注。
默认专家HDF5目录不可用，另发现现有LeRobot LIBERO10候选；需要核实其状态/标注并先检查血缘，不能跨入预留test场景。
步骤04完成范围是开发数据管线；training_data_ready=false。步骤03高层文本能力门槛仍未通过。
下一步05公共runner和06 CPU模块可继续；步骤08高层训练之前必须补齐上述监督及能力缺口。
后续数据、评估与VLA动力学闭环统一使用LIBERO；纯夹具仅用于接口单元测试。
'''
 text(ROOT/'docs/v4/04_data_status.txt',report);text(OUT/'handoff_step04.txt',report)
 commands=[{'command':'python execution/v4/inspect_step04.py','purpose':'existing LIBERO task catalogue'}, {'command':'python execution/v4/prepare_step04.py','purpose':'scene groups and split frozen before capture/noise'}, {'command':'python execution/v4/run_step04_capture.py','purpose':'actual 48 LIBERO reset/neutral-step captures'}, {'command':'python execution/v4/build_step04_data.py --rebuild-unfrozen','purpose':'last unfrozen data generation with canonical option order/covered future probe'}, {'command':'python execution/v4/inspect_step04.py --datasets','purpose':'bounded demonstration path search'}, {'command':'python execution/v4/verify_step04.py','result':'PASS; 222 checks'}]
 write(OUT/'04_command_log.json',commands)
 acceptance={'step_id':'V4-04','status':'DEVELOPMENT_PIPELINE_PASS_WITH_TRAINING_DATA_GAPS','completed_at_utc':datetime.now(timezone.utc).isoformat(),'actual_simulation_platform':'LIBERO','capture_status':'PASS','integrity_checks':verification['checks'],'integrity_passed':True,'base_episodes':48,'rendered_images':96,'public_rows':manifest['public_rows'],'private_rows':manifest['private_label_rows'],'counterfactual_pairs':384,'training_data_ready':False,'place_phase_supervision_ready':False,'successful_manipulation_episodes':0,'test_sealed':True,'training_run':False,'real_EEG':False,'physical_robot':False,'full_VLA_closed_loop_run':False,'next_step':'V4-05'}
 write(OUT/'04_acceptance.json',acceptance)
 source_files=list(DATA.rglob('*'))+[ROOT/'configs/v4/interfaces_data_4_1.json',ROOT/'configs/v4/platform_binding.json',ROOT/'docs/v4/04_data_status.txt']
 source_files+=[p for p in OUT.glob('*step04*.py')]+[p for p in OUT.glob('04*.json') if p.name!='04_source_manifest.json']+[OUT/'handoff_step04.txt']
 source_files+=list((ROOT/'bci_vla/data_v4').glob('*.py'))
 source_files=sorted(set(p for p in source_files if p.is_file()))
 write(OUT/'04_source_manifest.json',{'step':'V4-04','files':[{'path':str(p),'sha256':sha(p)} for p in source_files]})
 progress=read(ROOT/'execution/v4_plan_progress.json');old=read(OUT/'04_progress_before.json')
 assert progress['historical_v2_gates']==old['historical_v2_gates']
 assert progress['steps'][4:]==old['steps'][4:]
 progress['steps'][3].update(status=progress['steps'][0]['status'],completion_scope='LIBERO开发数据管线完成；训练级主任务/放置监督未就绪',training_data_ready=False,actual_artifacts=['data/v4/data_manifest.json','data/v4/split_manifest.json','data/v4/annotation_rules.txt','execution/v4/04_verification.json','execution/v4/04_acceptance.json','execution/v4/handoff_step04.txt'],verification={'status':acceptance['status'],'scope':'actual LIBERO capture/data integrity, not successful manipulation or training readiness','checks':verification['checks'],'report':'execution/v4/04_verification.json'},completed_at_utc=acceptance['completed_at_utc'])
 progress.update(current_stage='V4-04_LIBERO_development_data_complete_with_training_gaps',next_step='V4-05',latest_handoff='execution/v4/handoff_step04.txt',training_data_ready=False,full_VLA_closed_loop_run=False,future_simulation_platform='LIBERO',formal_experiments_run=False)
 write(ROOT/'execution/v4_plan_progress.json',progress)
 text(ROOT.parent/'BCI_VLA当前计划入口_V4.txt','当前计划V4-plan-2；2026-10-17。\n步骤01/02完成；步骤03审计完成、高层能力门槛未通过；步骤04实际LIBERO开发数据管线完成、训练级监督有缺口。\n下一步05公共runner；后续数据、评价、VLA动力学闭环统一使用LIBERO。\n最新交接：D:\\Scene_Aware\\bci_vla_framework\\execution\\v4\\handoff_step04.txt。\n数据：D:\\Scene_Aware\\bci_vla_framework\\data\\v4\\data_manifest.json。\n主计划：D:\\Scene_Aware\\BCI_VLA框架_意图更新与澄清_执行计划_V4.txt。\ntraining_data_ready=false；正式test未解封；新Transformer未训练；真实EEG及实体动作排除。\n')
 print(json.dumps({'status':acceptance['status'],'frozen_files':len(source_files),'checks':verification['checks'],'platform':'LIBERO','training_ready':False}))
if __name__=='__main__':main()
