"""Promote only the evidenced restricted language gate; retain all other gaps."""
import json,hashlib,ast
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'execution/v4'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def text(p,s):p.write_bytes(('\ufeff'+s.replace('\r\n','\n').replace('\n','\r\n')).encode('utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert not (OUT/'03R_acceptance.json').exists(),'Preserve acceptance'
 pilot=read(OUT/'03R_language_pilot_v1.json');reload=read(OUT/'03R_reload_audit.json');phase=read(OUT/'03R_state_precision_audit.json');hashes=read(OUT/'03R_final_weight_hashes.json')['files']
 assert pilot['status']=='PILOT_COMPLETE_NOT_GENERAL_CAPABILITY_ACCEPTANCE' and pilot['optimizer_steps']==240 and pilot['all_base_parameters_frozen']
 assert reload['status']==phase['status']=='AUDIT_COMPLETE'
 assert all(v['exact']==v['count'] for v in list(reload['metrics'].values())+list(phase['metrics'].values()))
 assert pilot['off_regression_max_abs']==reload['off_regression_max_abs']==phase['off_regression_max_abs']==0
 precision=phase['fp32_math_control'];assert precision['max_abs_cached_vs_teacher_logit']<1e-4 and precision['masked_future_difference']==0 and precision['unmasked_negative_control_difference']>1
 assert precision['all_argmax_agree'] and reload['cache_argmax_all_agree'] and phase['actual_after_grasp_images']
 assert hashes[0]['sha256']=='4616e4c1966a5fb063b442ea5d69857f45e873222d839b794634d87d252ab9e7'
 artifact=ROOT/'artifacts/v4/high_level_lora/step240.safetensors';assert digest(artifact)==hashes[1]['sha256']
 prior=read(OUT/'07_before_manifest.json')['files'];assert all(digest(Path(v['path']))==v['sha256'] for v in prior),'Prior evidence changed'
 now=datetime.now(timezone.utc).isoformat()
 limitations=[
  'This is a restricted protocol language gate, not general scene-grounded planning or learned BCI intent inference.',
  'Main task is demonstrated for an explicitly available pick action with a selected object; unspecified action asks for clarification. Arbitrary main-task/place proposals are not established.',
  'Next-subtask checks condition on explicit confirmed pick/put task and known public gripper state; the model is not shown to infer grasp state from RGB alone.',
  'High-level alias correctness does not verify same-class low-level target reaching; that original gate remains NOT_VERIFIED.',
  'Only two legacy calibration endpoint images support the actual held-state check; eight prompt/alias variants are not eight independent rollouts.',
  '120 validation prompts use eight frozen LIBERO episodes from two scene groups; template variants are correlated development cases, not independent task-success trials.',
  'BF16 maximum teacher/cache logit difference is .9375 despite matching greedy argmax; strict BF16 equivalence is NOT claimed. FP32 isolated math control is not a low-level deployment precision change.',
  'The Step06 interaction connector remains untrained; it was absent from this language pilot. No internal interaction prefix has been accepted on the real model yet.',
  'No new LIBERO dynamics, placing rollout, EEG, physical action or formal test was run. V4-04 interaction training-data gaps remain.',
  'All future B2/B3/B4/Full comparisons must use the same high-level language initialization/budget; do not attribute this common LoRA language gain to the interaction connector.',
  'High-level-only LoRA is disabled by default and scope-restored, but shared-model concurrency/real action regression must still be verified in Step07/11 before mixed use.'
 ]
 capability={'step_id':'V4-03R','completed_at_utc':now,'status':'PASS_RESTRICTED_HIGH_LEVEL_LANGUAGE_GATE','supersedes_only':'original V4-03 main-task/subtask language blocker within stated protocol; frozen original audit retained',
  'base_checkpoint':hashes[0],'adapter':{'local_path':str(artifact),'remote_path':hashes[1]['path'],'sha256':hashes[1]['sha256'],'bytes':hashes[1]['bytes'],'rank':16,'alpha':32,'targets':['q_proj','v_proj','o_proj'],'layers':18,'adapted_linears':54,'parameters':3022848,'optimizer_steps':240,'learning_rate':.001,'default_enabled':False},
  'high_level_main_task_gate':'PASSED','high_level_subtask_gate':'PASSED',
  'gate_scope':'canonical observable-public-state language protocol only; explicitly selected object/allowed pick, missing-action clarification, confirmed pick/put + empty/holding-selected gripper -> next subtask/Done',
  'original_checkpoint_without_adapter_gate':'NOT_PASSED','general_scene_grounded_planning_gate':'NOT_ESTABLISHED','same_class_low_level_grounding_gate':'NOT_VERIFIED','real_pi05_interaction_prefix_gate':'NOT_RUN',
  'before_after_same_development_subset':{'before':{'exact':0,'count':40},'after':{'exact':40,'count':40}},
  'reload_development_metrics':reload['metrics'],'actual_held_phase_metrics':phase['metrics'],'FP32_math_control':precision,
  'BF16_math':{'max_abs_logit_difference':reload['cache_max_abs_logit_difference'],'argmax_all_agree':reload['cache_argmax_all_agree'],'strict_equivalence':False},
  'off_regression_max_abs':0.,'base_checkpoint_full_hash_matches_history':True,'base_optimizer_updates':0,
  'training_data':{'train_episodes':24,'train_scene_groups':6,'train_prompts':480,'validation_episodes':8,'validation_scene_groups':2,'validation_prompts':120,'vision':'actual existing LIBERO captures + privileged public object dictionaries; 6D ROI/semantic limitations unchanged','after_grasp_training':'explicit symbolic text-only examples; real held images used only in later development audit','old_private_BCI_labels_used':False,'data_uploaded':False},
  'limitations':limitations,'allowed_next_step':'V4-07 real-checkpoint prefix engineering within this language protocol, adapter loaded/enabled only for high-level calls; no automatic training/performance claim',
  'formal_test_unsealed':False,'real_EEG':False,'physical_robot':False,'new_dynamics_run':False}
 write(OUT/'03R_text_capability.json',capability)
 branch={'version':'V4-03R-language-1','capability_audit':'execution/v4/03R_text_capability.json','base_checkpoint_sha256':hashes[0]['sha256'],'adapter_sha256':hashes[1]['sha256'],'local_adapter':str(artifact),'remote_adapter':hashes[1]['path'],'rank':16,'alpha':32,'targets':['q_proj','v_proj','o_proj'],'default_enabled':False,'runtime':'keep original BF16 low-level configuration; FP32 used only for isolated math audit','high_level_call':'load exact base then install_high_level_lora; copy matching A/B tensors from safetensors; freeze; use high_level_scope only around the high-level call','low_level_call':'adapters must be disabled; serial ownership/locking and real action regression remain required before deployment','LoRA_scope_note':'shared PaliGemma language layers are adapted conditionally, not a change to action-expert architecture','all_baselines_same_high_level_initialization':True,'all_baselines_common_language_train_steps':240,'comparison_training_budget_unfrozen':True,'no_interaction_connector_training_in_this_pilot':True}
 write(ROOT/'configs/v4/high_level_branch_step03R.json',branch)
 write(OUT/'03R_remaining_gaps.json',{'closed_in_restricted_protocol':['native high-level text/alias generation after explicit LoRA','teacher-forcing/EOS/position/cache causal control','conditional next-subtask semantics with public state','reload/off-path regression and immutable base checkpoint'], 'not_closed':limitations,'must_not_claim':'general planner, visual-only grasp recognition, BCI denoising success, same-class low-level execution, complete VLA closed loop or independent formal-test gains'})
 doc='''V4-03R 高层文字能力修复与限定验收（2026-10-17）

一、结论
已补齐阻止步骤07工程接入的“限定协议下主任务/下一子任务文字输出”缺口。
本次不是让未训练的步骤06小Transformer变聪明，而是先补训π0.5共享PaliGemma高层语言支路。
原checkpoint本身仍没有通过；新的可用分支是“原LIBERO SFT＋单独可开关的高层LoRA”。
只对本文件列出的协议给出通过，开放式规划、真实脑机去噪与低层执行能力不随之升级。

二、诊断与分支选择
原始π0.5 PyTorch权重和LIBERO SFT均经过原生caption/answer格式、纯文本因果、少样例提示对照。
两者都能输出自然语言，但仍无法稳定保留OBJ实例或给出正确的抓取后子任务。
因此不是简单“没文本头”，也不能把此前数字输出全归咎于接入错误；选择原计划允许的高层LoRA补训分支。
第一轮双权重探针360秒总超时，只保留了原始权重结果；后续SFT单独复验完成，未改原始失败证据。

三、实际训练
基座：/home/oyjt/workspace/RLinf-RECAP/models/baseline_sft/RLinf-Pi05-LIBERO-SFT/model.safetensors。
完整SHA256仍为4616e4c1966a5fb063b442ea5d69857f45e873222d839b794634d87d252ab9e7，和历史一致。
高层18层语言self-attention的q/v/o共54个线性层，rank16、alpha32，共3,022,848个可训练参数。
240步AdamW、lr=0.001；基座/视觉塔/embedding/文本头/动作专家不参加优化。
24个LIBERO train底层回合、6个场景组、480条模板监督；validation为8回合、2场景组、120条。
监督在服务器根据已有公开物体字典、明确选择、确认任务与显式状态生成，没有上传本地私有标签或读取旧隐藏真值。
抓取后训练样例是纯文本符号状态，不伪装成真实抓取图像。实际抓取端点图只用于后续开发检查。
本次从240步模型取得结果；120步和240步都在预定40条开发样例上40/40。

四、通过范围
1. 主任务：明确可用动作pick及选中OBJ_i，输出Pick up OBJ_i.
2. 缺动作：已经选对象但没有指定动作，输出Which task do you want?；不能偷偷猜任意动作。
3. 已确认pick/put任务＋明确公开夹爪状态：空夹爪输出Pick up；持有目标且put任务输出Place；持有目标且pick任务输出Done。
4. gripper state必须由可观察状态提供；本轮没有证明模型能仅从RGB识别抓取状态，也没有证明任意任务/场景分解。
5. 高层协议提供清楚的对象别名、当前状态和任务字段；本轮是语言能力启动，不是含噪BCI意图推断评测。
后续Full/B3/B4的意图证据须按计划经内部token提供；不能把训练时显式选中对象当成不可观测真值泄漏到正式评估。

五、开发检查结果
补训前同一40条开发样例0/40，补训后40/40。
独立重新加载：120/120原始验证样例、40/40联合别名置换、40/40未见措辞改写。
实际抓取端点：来自既有LIBERO校准init0的两个端点图，接触且抬升>0.03m、末3步持有成立；4/4原始状态问法和4/4别名变体通过。
这些是2张端点图的8种问法，不是8次独立机器人实验；没有新动作或新的成功轨迹。
FP32隔离数学检查：cache/teacher最大logits差约6.485e-5，argmax全部一致；屏蔽未来答案改动影响0，解除屏蔽负对照影响8.183。
BF16缓存与teacher的greedy argmax一致，但最大logits差0.9375，不宣称严格等价。FP32审计没有改变低层部署配置。
LoRA关闭后原始sentinel logits差0；原基座参数版本计数不变，原checkpoint完整哈希不变。

六、权重和复现入口
本地适配权重：D:\\Scene_Aware\\bci_vla_framework\\artifacts\\v4\\high_level_lora\\step240.safetensors，12,102,152字节。
SHA256：c1dbc8bd23e8e2590ac2670839a068d82423aa0c4620e734d2d99618afaec380。
服务器：/home/oyjt/Scene-aware-BCI-VLA-runs/v4_step03R_language_20261017_v1/high_level_lora_step240.safetensors。
配置configs/v4/high_level_branch_step03R.json；能力审计execution/v4/03R_text_capability.json；原始结果03R_language_pilot_v1.json、03R_reload_audit.json、03R_state_precision_audit.json。
代码bci_vla/high_level_v4/lora.py；复现训练脚本execution/v4/step03R_language_pilot.py；独立重载脚本step03R_reload_audit.py。
加载同一基座，install_high_level_lora(rank=16,alpha=32)，严格匹配safetensors中的A/B键和形状，冻结后只在high_level_scope中启用。
默认关闭并在异常退出时恢复开关。后续共享模型并发必须串行化/加锁，并测试低层实际动作回归，不能认为语言off检查等于动作验收。

七、后续约束
步骤07仍未完成：此前只有37项本地预备/回归测试，下一步必须实际接入8个内部token并跑真实模型on/off、梯度、cache和实例绑定测试。
步骤06模块未训练；V4-04保守intent SET、任务/放置监督缺口仍在，不能直接宣布步骤08训练就绪。
所有B2/B3/B4/Full后续比较必须共享本轮高层语言初始化及预算，单列共同的302万LoRA参数，不能把它的语言增益归给小模块结构。
同类多实例的低层到达/抓取仍NOT_VERIFIED；高层ID正确不等于动作专家会对准实例。
正式test未解封，真实EEG与实体机器人未运行，原G2/G3/G4保持原结论。训练数据变体和措辞样例不是独立机器人回合。
'''
 text(ROOT/'docs/v4/03R_backbone_remediation.txt',doc)
 handoff='''V4-03R：限定高层语言协议通过，原始03审计完整保留。
实际π0.5 LIBERO SFT＋高层LoRA（q/v/o rank16 alpha32，302万参数），训练240步，基座冻结且完整哈希未变。
原生提示仍失败，因此采用LoRA而不是将模板输出冒充原checkpoint能力。
原始验证120/120、别名40/40、改写40/40；两个真实抓取端点图的8个状态/别名检查通过。均为开发结果，不是独立正式测试。
主任务仅验证显式允许pick+已选对象；缺动作须澄清。子任务验证确认pick/put＋明确公开夹爪状态，不是从RGB自行推断抓取或通用规划。
LoRA关闭的sentinel logits差0；FP32 cache/teacher差6.485e-5且未来屏蔽检查通过；BF16严格数值等价未证明。
本地权重artifacts/v4/high_level_lora/step240.safetensors；配置configs/v4/high_level_branch_step03R.json。
下一步07可在该协议内做真实prefix工程接入；07尚未完成，06小模块尚未训练。04训练监督缺口保留。
同类实例低层执行未验证；高层LoRA只在高层scope启用，低层须关闭且后续验证并发/动作回归。
后续所有神经对照共享这份高层初始化；不把共同LoRA训练增益归因于交互小模块。
正式test、真实脑机、实体动作排除。当前不是完整VLA闭环验收。
'''
 text(OUT/'handoff_step03R.txt',handoff)
 write(OUT/'03R_acceptance.json',{'status':'PASS_RESTRICTED_HIGH_LEVEL_LANGUAGE_GATE','completed_at_utc':now,'scope':capability['gate_scope'],'same_class_low_level_grounding':'NOT_VERIFIED','general_planning':'NOT_ESTABLISHED','next_step':'V4-07 real prefix integration; not yet accepted','formal_test_unsealed':False})
 old=read(OUT/'07_progress_before.json');progress=read(ROOT/'execution/v4_plan_progress.json')
 assert progress['historical_v2_gates']==old['historical_v2_gates']
 assert all(progress['steps'][i]==old['steps'][i] for i in range(len(old['steps'])) if i not in [6])
 progress['steps'][2].update(remediated_at_utc=now,remediation_scope='原LIBERO SFT＋高层LoRA的限定公共状态语言协议通过；保留泛化/低层实例限制',remediation_artifacts=['execution/v4/03R_text_capability.json','configs/v4/high_level_branch_step03R.json','artifacts/v4/high_level_lora/step240.safetensors','docs/v4/03R_backbone_remediation.txt'],high_level_language_gate='PASSED_RESTRICTED_PROTOCOL')
 progress['steps'][6].update(status='接入预备完成，待真实prefix验收',dependency_remediation='V4-03R restricted language protocol passed; use exact adapter and preserve scope')
 progress.update(current_stage='V4-03R_restricted_high_level_language_gap_remediated',next_step='V4-07',latest_handoff='execution/v4/handoff_step03R.txt',high_level_gate_passed=True,high_level_gate_scope=capability['gate_scope'],high_level_gate_report='execution/v4/03R_text_capability.json',high_level_lora_training_run=True,training_started=True,training_data_ready=False,neural_baseline_training_run=False,formal_experiments_run=False,full_VLA_closed_loop_run=False,new_experiments_run_meaning='Development high-level LoRA pilot and scoped language/cache audits; no formal evaluation or interaction-connector training')
 write(ROOT/'execution/v4_plan_progress.json',progress)
 text(ROOT.parent/'BCI_VLA当前计划入口_V4.txt','当前V4-plan-2；2026-10-17。\n步骤03R限定高层语言协议已通过：原LIBERO SFT＋高层LoRA，非原checkpoint直接能力。\n新能力审计：D:\\Scene_Aware\\bci_vla_framework\\execution\\v4\\03R_text_capability.json。\n说明：D:\\Scene_Aware\\bci_vla_framework\\docs\\v4\\03R_backbone_remediation.txt。\n最新交接：D:\\Scene_Aware\\bci_vla_framework\\execution\\v4\\handoff_step03R.txt。\n步骤06原型工程完成但未训练；步骤07预备完成、真实prefix验收未完成。下一步07须加载限定高层LoRA并遵守协议。\n同类物体低层执行、泛化规划未验证；04训练监督缺口保留。正式test封存，真实EEG/实体动作排除。\n')
 files=list((OUT).glob('*03R*.py'))+list(OUT.glob('03R*.json'))+list(OUT.glob('03R*.log'))+list((ROOT/'bci_vla/high_level_v4').glob('*.py'))+[ROOT/'tests/test_high_level_lora_v4.py',ROOT/'configs/v4/high_level_branch_step03R.json',ROOT/'docs/v4/03R_backbone_remediation.txt',OUT/'handoff_step03R.txt',artifact]
 files=sorted(set(p for p in files if p.name!='03R_source_manifest.json'))
 for p in files:
  if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
 write(OUT/'03R_source_manifest.json',{'step':'V4-03R','files':[{'path':str(p),'sha256':digest(p)} for p in files]})
 print(json.dumps({'status':'PASS_RESTRICTED_HIGH_LEVEL_LANGUAGE_GATE','development_reload_cases':200,'actual_held_prompt_cases':8,'base_unchanged':True,'prior_files_unchanged':len(prior),'next_step':'V4-07'}))
if __name__=='__main__':main()
