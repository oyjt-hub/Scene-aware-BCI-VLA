"""Freeze actual prefix engineering, explicitly retain failed untrained semantics."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,ast,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'execution/v4'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def text(p,s):p.write_bytes(('\ufeff'+s.replace('\r\n','\n').replace('\n','\r\n')).encode('utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert not (OUT/'07_acceptance.json').exists(),'Preserve completion evidence'
 real=read(OUT/'07_real_attempt2.json');local=read(OUT/'07_local_result.json');language=read(OUT/'03R_text_capability.json')
 assert real['status']=='PASS_REAL_PREFIX_ENGINEERING' and real['checks_passed']==28 and all(c['passed'] for c in real['checks'])
 assert local['status']=='PASS_PREWORK_ONLY' and local['tests_run']==38
 assert not real['interactions_trained'] and real['optimizer_steps']==0 and not real['untrained_on_language']['semantic_match']
 before=read(OUT/'07_before_manifest.json')['files'];assert all(digest(Path(v['path']))==v['sha256'] for v in before)
 dependency=read(OUT/'03R_source_manifest.json')['files'];assert all(digest(Path(v['path']))==v['sha256'] for v in dependency)
 source=read(OUT/'07_real_code_manifest2.json')['files'];assert all(digest(ROOT/v['path'])==v['sha256'] for v in source)
 now=datetime.now(timezone.utc).isoformat();checks={v['name']:v for v in real['checks']}
 shutil.copyfile(OUT/'07_integration_audit.json',OUT/'07_prework_audit.json')
 audit={'step_id':'V4-07','status':'PASS_REAL_PREFIX_ENGINEERING_ONLY','completed_at_utc':now,
  'dependency':'V4-03R conditional high-level language LoRA; original checkpoint alone still NOT_PASSED','dependency_report':'execution/v4/03R_text_capability.json','dependency_scope':language['gate_scope'],
  'actual_checkpoint':language['base_checkpoint'],'high_level_language_adapter':language['adapter'],
  'local_tests':38,'real_GPU_engineering_checks':28,'real_report':'execution/v4/07_real_attempt2.json',
  'token_path':{'native':'unchanged native image+prompt prefix','boundary':'append 8 evidence vectors after native image+prompt and before causal answer inputs','shape':[1,8,2048],'connector_dtype':'float32','VLM_dtype':'bfloat16','casting':'differentiable, without extra sqrt(D)','on_context_length':400,'off_context_length':392,'off_semantics':'physically remove 8 tokens, native mask/position/cache restored; eight zero vectors are NOT off'},
  'backward':{k:v for k,v in checks.items() if k.startswith('real_language_gradient_')},
  'off_regression':checks['off_exact_native_regression'],'causal_control':checks['real_on_causal_future_barrier'],
  'BF16_cache_control':{k:v for k,v in checks.items() if k.startswith('teacher_KV_argmax_')},
  'cache_key':['full public input digest including scene/mapping/goal/history','base checkpoint sha','language adapter sha','on/off mode','image+prompt+soft-embedding digest','valid/ar mask digest'],
  'scope_isolation':'trained language LoRA enabled only around high-level VLM forward, restored after every call; no action entry monkey patch; shared-model mixed-call concurrency remains to be implemented/verified before deployment',
  'instance_binding':{'neural_object_row_equivariance':'PASS','neural_alias_independent_prefix_and_physical_ID_metadata':'PASS','native_trained_language_joint_alias_generation':'PASS','on_untrained_prefix_semantic_instance_binding':'NOT_PASSED_NOT_CLAIMED; revisit after Step08'},
  'untrained_prefix_generation':real['untrained_on_language'],'semantics_status':'ON_SEMANTICS_NOT_PASSED; engineering-only acceptance, not trained reasoning',
  'first_attempt':'Preserved as FAIL: incorrect acceptance assertion expected meaningful language from random untrained prefix. Off native answer was correct; on returned task task. Corrected test scope without hiding this observed failure.',
  'action_expert':checks['base_and_action_expert_immutable'],'action_sampling_run':False,'new_simulation_dynamics':False,'full_VLA_closed_loop':False,
  'training_run_this_step':False,'optimizer_steps_this_step':0,'training_data_ready':False,'formal_test_unsealed':False,'real_EEG':False,'physical_robot':False,'peak_GPU_bytes':real['peak_GPU_bytes'],
  'next_gate':'V4-08 needs V4-04 intent/language/slot/abstention supervision repaired; no automatic training; all baselines use same high-level LoRA initialization/budget'}
 write(OUT/'07_integration_audit.json',audit)
 config={'version':'V4-07-prefix-1','implementation':'bci_vla.prefix_integration_v4.adapter.Pi05HighLevelPrefix','capability_report':'execution/v4/03R_text_capability.json','language_branch_config':'configs/v4/high_level_branch_step03R.json','K':8,'D_vlm':2048,'insertion':'after_native_image_prompt_before_causal_answer','prefix_extra_embedding_scale':1,'off':'remove_tokens_and_invalidate_cache','default_execution_authorized':False,'generation_maximum':128,'single_session_generation_only':True,'requires_bound_language_adapter':True,'requires_serial_shared_model_ownership':True,'trained_interaction_prefix_available':False,'on_semantics_validated':False,'notes':'engineering gate only; invalid generated text must never be dispatched as a task; confirmation/authorization remain separate Step11 work'}
 write(ROOT/'configs/v4/prefix_integration_step07.json',config)
 doc='''V4-07 真实π0.5内部token接入工程验收（2026-10-17）

一、结论与明确限制
真实π0.5高层VLM已经接收小模块输出的8×2048内部向量，关闭回归、梯度、缓存、mask/位置与工程绑定检查通过。
这是接入工程完成，不是小模块已学会理解脑机。开启随机未训练prefix时实际输出“task task.”，语义未通过，必须保留到08训练后再验。
原计划07验收允许“尚无性能结论”。首次把开启后也必须输出正确任务写成断言，超出了本步工程边界；首次失败和代码保留。
后续修正只区分工程正确与未训练语义，没有训练小模块或将错误输出改写成答案。

二、先补齐的03依赖
用户要求优先修复03后，已用实际LIBERO开发图像/公开字典进行高层LoRA补训并完成独立复验。
使用原LIBERO SFT＋rank16/alpha32语言q/v/o LoRA，302万参数；并非原checkpoint突然具备可靠任务文字能力。
新高层门槛只覆盖显式选中对象/可用pick/缺动作澄清及确认pick/put＋明确夹爪状态，详见docs/v4/03R_backbone_remediation.txt。
通用规划、从RGB自行识别抓取状态、同类物体低层执行仍未证明。

三、实际接入路径
先从PI0Pytorch.embed_prefix得到原图像与prompt embedding。
在其后、因果answer输入之前追加8个小模块向量；输入不经过字符串tokenizer，不额外乘sqrt(D)。
小模块float32输出经可微cast变为实际VLM bfloat16。图像、原prompt、原生语言token缩放保持原路径。
上下文为bidirectional prefix；teacher forcing输入target[:-1]，最后有效context位置预测target[0]，answer位置逐个预测后续token，EOS有监督。
padding/attention/position_ids随真实token数量重建；KV decode区分物理cache长度与有效逻辑位置。
off真的移除8个token并清理cache，而不是把8个token置零留在序列中。
每个高层forward在high_level_scope中启用已训练语言LoRA，结束恢复关闭。动作采样入口没有被patch。
共享模型并发的高低层调用仍需后续串行化/加锁和真实动作回归，本步不宣称部署安全或动作成功。

四、实际结果
本地38项测试通过，包含18项prefix工程测试和步骤06的20项回归。
真实GPU模型28项工程检查通过，来源为服务器已有public LIBERO夹具和对应真实图像，只上传代码/schema，没有上传数据或私有标签。
原生上下文392 token；on为400。off与原生logits最大差0；八个零token与off最大差32.5，证明确实必须移除token。
原生off输出Pick up OBJ_0.；随机prefix on输出task task.。这不是可执行指令，不授予动作权限。
Full/B3/B4均实际通过语言CE反传，小模块prefix梯度L1分别约608.10/1857.12/377.81；底座/高层LoRA/动作专家均冻结、无梯度和优化器更新。
同样的目标变化、映射变化、prefix变动、on/off切换都会拒绝旧KV并清空，之后需要重新prefill。
padding与对象行重排保持输出/输入绑定一致；联合alias变换在已训练原生off语言分支正确，on训练后语义绑定尚未通过。
真实on未来答案屏蔽影响0，解除屏蔽负对照影响2.75；teacher/cache greedy argmax一致。
BF16最大logits差off=1.75、on=1.0，保留原值，不宣称严格数值等价。03R的FP32隔离因果/缓存数学检查另有证据。
所有基座参数版本计数与冻结状态保持不变；动作专家结构/参数形状未改，实际动作采样未运行。
实测peak GPU allocated约8.72GB，仅此开发形状，不能据此推断多样本训练成本。

五、代码与调用约束
bci_vla/prefix_integration_v4/adapter.py；配置configs/v4/prefix_integration_step07.json。
构造时必须使用03R能力报告、正确base SHA和已加载高层LoRA SHA；不允许拿原始未适配checkpoint搭配“通过”报告。
先安装/加载高层语言LoRA，再构造Pi05HighLevelPrefix；传入high_level_adapters和adapter_sha256，底座会被冻结但不会以整体no_grad截断teacher-forcing梯度。
prepare处理原图/原prompt及可选soft_tokens；teacher_forcing返回可微CE；generate只返回有界原始token，execution_authorized始终false。
本步没有构建可靠的on语义输出校验/确认执行器；其输出不得直接送给机器人。
复现：本地python execution/v4/run_step07_local.py；真实脚本execution/v4/step07_real_integration.py。
为保留证据，远程runner要求新的attempt编号并新建隔离目录；现有1/2次结果不可覆盖。

六、接下来
步骤08前仍要补齐04的监督缺口：当前保守intent SET包括所有候选和outside，不能形成有效目标区分；缺槽/弃权及放置监督也不完整。
所有对照必须共用本轮03R高层初始化/预算，不得把它的语言能力增益算成交互小模块贡献。
本步没有optimizer.step、小模块训练、正式test、新仿真动力学、真实EEG或实体机器人动作。旧G2/G3/G4不升级。
完整VLA闭环、on语义实例绑定和任务成功率仍需要后续训练与LIBERO验证。
'''
 text(ROOT/'docs/v4/07_integration.txt',doc)
 text(OUT/'handoff_step07.txt','''V4-07真实prefix工程接入完成，开启语义仍未通过。
按用户要求先补齐03R限定高层语言门槛：原SFT＋单独高层LoRA；不是通用规划能力验收。
真实8×2048内部token接入、off精确回归、因果/缓存/位置/padding、三个结构真实语言CE梯度通过；38本地测试＋28真实GPU工程检查。
off输出Pick up OBJ_0.；随机未训练prefix on实际输出task task.，保留失败，不作有用推理/实例绑定声明。
首次错误的on语义通过断言已修正为工程范围，原始失败与源码在07_real_attempt1*；最终证据07_real_attempt2.json。
模块代码bci_vla/prefix_integration_v4；审计07_integration_audit.json；说明docs/v4/07_integration.txt。
下一步08前须修复04监督缺口；06小模块仍未训练。对照统一03R高层初始化与预算。
共享高低层并发/动作回归未完成；动作专家结构未改，未采样动作或推进新LIBERO动力学。
正式test封存；真实EEG/实体动作排除；同类实例低层执行和完整VLA闭环未验证。
''')
 write(OUT/'07_acceptance.json',{'step_id':'V4-07','status':'PASS_ENGINEERING_ONLY_ON_SEMANTICS_NOT_PASSED','completed_at_utc':now,'local_tests':38,'real_checks':28,'on_semantics_passed':False,'trained_interaction_prefix':False,'instance_binding_scope':'native off language + tensor/metadata engineering, on semantics deferred','training_data_ready':False,'next_step':'V4-08 after missing supervision is repaired'})
 progress=read(ROOT/'execution/v4_plan_progress.json');old=read(OUT/'07_progress_before.json');assert progress['historical_v2_gates']==old['historical_v2_gates']
 assert all(progress['steps'][i]==old['steps'][i] for i in range(len(old['steps'])) if i not in [2,6])
 progress['steps'][6].update(status='工程接入已完成（开启语义待训练后验收）',completed_at_utc=now,completion_scope='真实π0.5内部token/off/梯度/cache工程通过；随机prefix开启语义未通过，不是系统能力完成',actual_artifacts=['bci_vla/prefix_integration_v4','configs/v4/prefix_integration_step07.json','execution/v4/07_integration_audit.json','execution/v4/07_real_attempt2.json','docs/v4/07_integration.txt','execution/v4/handoff_step07.txt'],verification={'status':'PASS_ENGINEERING_ONLY_ON_SEMANTICS_NOT_PASSED','local_tests':38,'real_checks':28,'report':'execution/v4/07_integration_audit.json'})
 progress.update(current_stage='V4-07_real_prefix_engineering_complete_on_semantics_not_passed',next_step='V4-08',latest_handoff='execution/v4/handoff_step07.txt',interaction_prefix_integrated=True,interaction_prefix_trained=False,on_prefix_semantics_passed=False,training_data_ready=False,formal_experiments_run=False,full_VLA_closed_loop_run=False,neural_baseline_training_run=False,step08_dependency='Repair V4-04 target/slot/abstention/language supervision before meaningful comparison training')
 write(ROOT/'execution/v4_plan_progress.json',progress)
 text(ROOT.parent/'BCI_VLA当前计划入口_V4.txt','当前V4-plan-2；2026-10-17。\n03R限定高层语言协议通过（原SFT＋高层LoRA）；06小模块仍未训练。\n07真实8×2048 prefix工程接入完成：38本地测试＋28真实模型工程检查；off logits差0。\n开启随机prefix输出task task.，语义未通过；不可宣称理解脑机或完整VLA闭环。\n最新交接：D:\\Scene_Aware\\bci_vla_framework\\execution\\v4\\handoff_step07.txt。\n03R说明：D:\\Scene_Aware\\bci_vla_framework\\docs\\v4\\03R_backbone_remediation.txt。\n07说明：D:\\Scene_Aware\\bci_vla_framework\\docs\\v4\\07_integration.txt。\n下一步08之前须补齐04监督缺口。所有神经对照共用03R高层初始化/预算。\n正式test封存；真实EEG/实体动作排除；同类实例低层执行未验证。\n')
 files=list((ROOT/'bci_vla/prefix_integration_v4').glob('*.py'))+[ROOT/'tests/test_prefix_integration_v4.py',ROOT/'configs/v4/prefix_integration_step07.json',ROOT/'docs/v4/07_integration.txt',OUT/'handoff_step07.txt']
 files+=list(OUT.glob('*step07*.py'))+list(OUT.glob('07*.json'))+list(OUT.glob('07*.log'))+[OUT/'03R_source_manifest.json']
 files=sorted(set(p for p in files if p.name!='07_source_manifest.json'))
 for p in files:
  if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
 write(OUT/'07_source_manifest.json',{'step':'V4-07','files':[{'path':str(p),'sha256':digest(p)} for p in files]})
 print(json.dumps({'status':'PASS_ENGINEERING_ONLY_ON_SEMANTICS_NOT_PASSED','local_tests':38,'real_checks':28,'prior_files_unchanged':len(before),'source_files':len(files),'next':'V4-08 gated on supervision repair'}))
if __name__=='__main__':main()
