"""Freeze CPU mechanism evidence; preserve prior research gates and sealed test."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,ast
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'execution/v4'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def text(p,s):p.write_bytes(('\ufeff'+s.replace('\r\n','\n').replace('\n','\r\n')).encode('utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert not (OUT/'06_acceptance.json').exists(),'Preserve completed evidence'
 result=read(OUT/'06_cpu_result.json');assert result['status']=='PASS' and result['tests_run']==20 and result['execution_location']=='local_windows'
 assert result['device']=='cpu' and not result['cuda_initialized'] and result['optimizer_steps']==0
 frozen=read(OUT/'06_before_manifest.json')['files']
 assert all(Path(r['path']).exists() and digest(Path(r['path']))==r['sha256'] for r in frozen),'Prior source changed'
 for row in read(OUT/'06_cpu_input_manifest.json')['files']:assert digest(ROOT/row['path'])==row['sha256'],row['path']
 now=datetime.now(timezone.utc).isoformat();models=result['models'];base=models['Full']['parameters']
 fixture=read(OUT/'06_fixtures/manifest.json')
 audit={'step_id':'V4-06','status':'PASS_CPU_ENGINEERING_ONLY','completed_at_utc':now,
  'runtime':{k:result[k] for k in ['python','torch','platform','device','cuda_initialized','execution_location']},
  'models':models,'baseline_parameter_difference_fraction':{k:(v['parameters']-base)/base for k,v in models.items()},
  'architecture':{'hidden_size':256,'heads':4,'history_layers_per_group':1,'fusion_blocks_per_group':2,'ffn_dim':1024,'history_limit':8,'candidate_limit':16,'query_groups':[4,4],'prefix_composition':'4 intent query latents + 4 typed diagnostic summary latents; raw channel queries do not cross','prefix_shape':[8,2048],'vlm_dimension_source':'V4-03 actual pi0.5 config; no VLM loaded here','typed_diagnostic_fields':['attempt_count/64','correct_count/64','no_response_count/64','covered_symbols/16','age/30 clamped4','truncated','has_evidence','sufficient','expired','insufficient','masked_uncalibrated_reliability','has_current_or_unexpired_summary'],'channel_allowed_memory':'deidentified diagnostic history + public diagnostic summary only; task/object inputs have no gradient path into reliability','outputs':['candidate+outside logits/probabilities','missing slots logits [source,destination,action]','abstain logit','reliability logit/probability/validity','interaction logits [PROPOSE_TASK,ASK_INTENT,CHECK_CHANNEL,WAIT]','prefix [B,8,2048]']},
  'data':fixture,'tests_run':result['tests_run'],'gradient_scope':'backward only, real masked auxiliary losses plus explicitly synthetic wiring objective; no optimizer step or claim of learned reasoning',
  'supervision_limits':['conservative current SET includes all candidates and outside; intent loss gives no discriminative supervision','missing-slot and abstain heads only wiring-tested, no current private labels for these heads','high-level language loss and VLA integration belong to later gated stages','B3/B4 with and without auxiliary supervision reuse same architectures/parameters; no variants have been trained'],
  'controls':'STOP/CANCEL uses validated pure contract routing before neural call; this module has no device dispatch or authorization',
  'first_attempt':'17 tests; 2 test-fixture failures (PROBE kind DIAGNOSTIC and incomplete control events), fixed; preserved log',
  'second_attempt':'19 tests passed on existing project server CPU; final extended 20 tests rerun locally',
  'approval_event':'Expanded simulator-label upload refused by auto-review twice, including after source audit; no expanded payload uploaded. Final verification completed locally using official downloaded CPU dependency; no unresolved upload is needed for Step06.',
  'not_claimed':['trained reasoning','calibrated confidence','pi0.5 high-level integration','successful manipulation','new LIBERO dynamics','formal test performance'],
  'high_level_gate_passed':False,'training_data_ready':False,'training_run':False,'formal_test_unsealed':False,'real_EEG':False,'physical_robot':False}
 write(OUT/'06_module_audit.json',audit)
 registry={'version':'V4-06-registry-1','parent_registry':'configs/v4/baseline_registry.json','parent_sha256':digest(ROOT/'configs/v4/baseline_registry.json'),'status':'implemented_untrained_CPU_only','methods':[],'matching':'same tensor inputs, 8x2048 prefix, all actual parameters counted; parameter match only, not equal FLOPs/latency; future training and VLM budget not yet frozen'}
 for key in ['B3_aux','B4_aux','Full']:
  registry['methods'].append({'id':key,'implementation':{'Full':'IntentChannelConnector','B3_aux':'PooledMLPConnector','B4_aux':'SharedQueryConnector'}[key],**models[key]})
 for name in ['B3','B4']:registry['methods'].append({'id':name,'same_architecture_as':name+'_aux','parameters':models[name+'_aux']['parameters'],'status':'implemented_architecture_untrained','auxiliary_losses':'disabled in future no-aux comparison; not trained in Step06'})
 write(ROOT/'configs/v4/baseline_registry_step06.json',registry)
 doc='''V4-06 小型交互Transformer CPU原型（2026-10-17）

一、这一步完成了什么
输入仍为步骤04实际LIBERO图像裁剪统计、位置、模拟选择证据和公共交互历史。
输出8个2048维内部token以及候选/候选外/缺槽/弃权/可靠性/询问辅助头。token尚未送入π0.5。
Full实际7,281,930参数；B3池化MLP 7,269,034参数（宽992）；B4共享查询7,281,930参数。
B3相差约0.1771%，B4完全相同，未添加虚假占位参数。这只匹配参数，不表示计算量或时延相同。

二、代码入口与逐步调用
1. 先调用bci_vla.neural_interaction_v4.control.route_control(public)。STOP/CANCEL返回已有直接控制路径，call_neural_module=false；不得继续调用模型。
2. collate_public([public])先执行4.1/4.0合同校验，然后返回张量和独立metadata。
3. IntentChannelConnector()(tensors)返回原始可微输出。metadata中的ID/alias/sample hash不传给forward，只在输出与loss绑定使用。
4. 实例化matched_models()可取得Full、B3_aux、B4_aux。无aux对照B3/B4复用相同模型结构，之后训练时关闭辅助损失。
5. auxiliary_losses(output,publics,private_labels,metadata)在loss端验证私有标签和公开输入哈希；forward拒绝额外key。
6. inspect_prediction生成module-inspection/0.1调试记录，不能作为冻结执行合同或机器人执行指令。

三、架构
d=256，4头，FFN=1024，H<=8，N<=16。bbox、角色、时间通过输入数值/one-hot联合投影编码；没有ID或别名数字嵌入。
任务与诊断先分组，再分别做一层历史Transformer；4个意图query与4个通道query各经过两个融合块。
通道只读去物体语义的PROBE统计、分数、时效和公开诊断摘要，不能读取TASK选择、对象ROI或任务历史。
只有有字段定义的12维摘要可经4个投影token单向送入意图分支。原始通道query不送VLM。
最终prefix前4个来自意图query，后4个来自摘要。不同角色在self-attention前就分开，所有融合层保持隔离。
全mask组使用有效EMPTY哨兵，padding在投影前归零；有效非有限值或超出归一化范围的值报错。
没有诊断历史且没有未过期摘要时，可靠性有效mask=false，导出值为null；任何结果均uncalibrated。
窗口外已确认任务通过public_state独立编码，不依赖它是否仍在8条历史中。
STOP/CANCEL依赖既有直接路由；模型只能提出建议，execution_authorized恒为false。

四、数据与验证范围
最终66条public、68条private，均来自冻结train/validation，含真实同公开输入/不同隐藏目标的反事实对。
这些private仅是程序生成的仿真监督，不含EEG或受试者数据。正式test和校准集未读取。
20项CPU测试全部通过：输出形状/有限值/候选概率；参数匹配；空输入与N/H=0；批次隔离；对象重排等变及梯度等变；别名不进入特征；padding NaN；非法值/字段/维度；逐层通道隔离；PROBE物体引用变化不变性；诊断数值正向影响；单向梯度；持久摘要/证据不足；9次PROBE后确认任务保留；STOP/CANCEL无torch旁路；私有反事实标签隔离；全部模型反传；选项重排；超限历史；普通共享基线负对照。
来源特征仍只是6维实际RGB均值/标准差，semantic缺失并mask=false；不是Florence/SAM或预训练视觉向量。
空历史、控制与>H机制测试另使用步骤02的明确合成夹具，不视为新LIBERO任务结果。
反向传播含真实辅助loss和显式机制测试目标；没有optimizer.step，没有训练、保存权重或性能评价。

五、必须保留的限制
当前保守intent SET包含全部候选和outside，loss没有区分目标的信息；不能据此开始有效目标绑定训练。
缺槽与弃权目前只有输出及梯度接线测试，没有对应私有监督；语言损失也尚未接入。
首次17项测试有两项夹具错误，修正后服务器19项通过。扩展反事实上传被自动审批拒绝后，最终20项改在本机CPU完成。
本机CPU依赖位于D:\\Scene_Aware\\.runtime\\step06_cpu；项目脚本不修改全局Python包，不上传测试数据。
复验命令（项目目录）：python execution/v4/run_step06_local.py；该命令会刷新CPU结果和输入哈希清单。
审计：execution/v4/06_module_audit.json；逐项日志：execution/v4/06_cpu_local.log；原始结果：execution/v4/06_cpu_result.json。
下一步07仍依赖03高层文本能力通过。现有π0.5生成探针未证明主任务/子任务能力，不能把06原型通过称为接入或完整LIBERO动作闭环成功。
真实EEG、实体机器人、GPU训练、正式test均未运行。历史G2/G3/G4及01–05冻结证据保持不变。
'''
 text(ROOT/'docs/v4/06_module.txt',doc)
 handoff='''V4-06已完成：CPU模块工程验收，非训练/推理能力验收。
主模块7,281,930参数，B3 7,269,034，B4 7,281,930；输出8×2048 prefix，尚未接入π0.5。
66条LIBERO train/validation public及68条模拟private，20项本地CPU机制/梯度测试通过。
本机复验：在D:\\Scene_Aware\\bci_vla_framework运行python execution/v4/run_step06_local.py。
代码bci_vla/neural_interaction_v4；审计execution/v4/06_module_audit.json；说明docs/v4/06_module.txt。
私有标签仅用于loss；PROBE物体语义不进入诊断支路；STOP/CANCEL走既有直接控制路由；任何输出不授权执行。
自动审批拒绝扩展仿真标签上传，已改为本地CPU完成，不再需要外发这批数据。
未训练/未校准/未调用VLA/未推进新仿真动力学/未开test。03高层门槛未通过，04主任务/放置监督缺口仍在。
下一步07须先补齐03门槛；不能直接宣称07/08完成。真实脑机、实体动作仍排除，后续平台LIBERO。
'''
 text(OUT/'handoff_step06.txt',handoff)
 checks={'prior_frozen_files_unchanged':len(frozen),'final_tests':20,'local_CPU_no_cuda':True,'hash_bound_inputs':True,'formal_test_sealed':True,'no_training':True,'no_VLA_or_new_dynamics':True,'parameters_match_within_5_percent':True}
 write(OUT/'06_verification.json',{'status':'PASS_CPU_ENGINEERING_ONLY','checks':checks,'result':'execution/v4/06_cpu_result.json'})
 write(OUT/'06_acceptance.json',{'step_id':'V4-06','status':'PASS_ENGINEERING_SCOPE','completed_at_utc':now,'tests':20,'claims':'interface/masks/routing/equivariance/numerics/gradients only','high_level_gate_passed':False,'training_data_ready':False,'next_step':'V4-07 with unresolved V4-03 dependency'})
 write(OUT/'06_command_log.json',[{'command':'python execution/v4/prepare_step06.py','final_public_records':66,'final_private_records':68},{'command':'python execution/v4/run_step06_cpu.py --attempt 1','status':'17 tests, 2 fixture failures'},{'command':'python execution/v4/run_step06_cpu.py --attempt 2','status':'PASS 19 CPU tests'},{'command':'python execution/v4/run_step06_cpu.py --attempt 3','status':'NOT EXECUTED; auto-review rejected expanded synthetic-private upload twice'},{'command':'python -m pip install --target D:\\Scene_Aware\\.runtime\\step06_cpu --index-url https://download.pytorch.org/whl/cpu torch==2.7.1','scope':'public official CPU dependency download, no project data transfer'},{'command':'python execution/v4/run_step06_local.py','status':'PASS 20 local CPU tests'}])
 progress=read(ROOT/'execution/v4_plan_progress.json');old=read(OUT/'06_progress_before.json')
 assert progress['historical_v2_gates']==old['historical_v2_gates'] and progress['steps'][:5]==old['steps'][:5] and progress['steps'][6:]==old['steps'][6:]
 progress['steps'][5].update(status=progress['steps'][0]['status'],completed_at_utc=now,completion_scope='小型Transformer及B3/B4未训练CPU原型；接口、路由、等变、数值与梯度工程验收',actual_artifacts=['bci_vla/neural_interaction_v4','tests/test_neural_interaction_v4.py','configs/v4/baseline_registry_step06.json','execution/v4/06_module_audit.json','execution/v4/06_cpu_result.json','execution/v4/06_verification.json','execution/v4/handoff_step06.txt'],verification={'status':'PASS_ENGINEERING_SCOPE','tests':20,'report':'execution/v4/06_verification.json'})
 progress.update(current_stage='V4-06_CPU_module_complete_untrained',next_step='V4-07',latest_handoff='execution/v4/handoff_step06.txt',review_status='V4-06_CPU_structure_verified_not_trained',implementation_started=True,training_started=False,formal_experiments_run=False,high_level_gate_passed=False,training_data_ready=False,full_VLA_closed_loop_run=False,neural_baseline_training_run=False)
 write(ROOT/'execution/v4_plan_progress.json',progress)
 text(ROOT.parent/'BCI_VLA当前计划入口_V4.txt','当前V4-plan-2；2026-10-17。\n步骤06 CPU模块及匹配B3/B4原型完成；20项本地CPU结构/梯度测试通过。\n主模块约728万参数，8×2048 prefix；尚未训练、接入π0.5或运行VLA动作。\n最新交接：D:\\Scene_Aware\\bci_vla_framework\\execution\\v4\\handoff_step06.txt。\n说明：D:\\Scene_Aware\\bci_vla_framework\\docs\\v4\\06_module.txt。\n下一步07依赖03高层文本能力门槛通过；04训练监督缺口保留。\n后续平台LIBERO；真实EEG/实体动作排除；正式test封存。\n')
 files=list((ROOT/'bci_vla/neural_interaction_v4').glob('*.py'))+list((OUT/'06_fixtures').glob('*.json'))+[ROOT/'tests/test_neural_interaction_v4.py',ROOT/'configs/v4/baseline_registry_step06.json',ROOT/'docs/v4/06_module.txt',OUT/'handoff_step06.txt']
 files+=list(OUT.glob('*step06*.py'))+list(OUT.glob('06*.json'))+list(OUT.glob('06*.log'))
 files=sorted(set(p for p in files if p.name!='06_source_manifest.json'))
 for p in files:
  if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
 write(OUT/'06_source_manifest.json',{'step':'V4-06','files':[{'path':str(p),'sha256':digest(p)} for p in files]})
 print(json.dumps({'status':'PASS_ENGINEERING_SCOPE','tests':20,'frozen_prior_files':len(frozen),'source_files':len(files),'next_step':'V4-07 gated by V4-03'}))
if __name__=='__main__':main()
