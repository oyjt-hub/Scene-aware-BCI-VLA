"""Preserve the audit and report capability gates without promoting old results."""
import ast,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'execution/v4'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def text(p,s):p.write_bytes(('\ufeff'+s.replace('\r\n','\n').replace('\n','\r\n')).encode('utf-8'))
def main():
    assert not (OUT/'03_acceptance.json').exists(),'Preserve existing acceptance evidence'
    audit=read(OUT/'03_remote_source_audit.json')
    candidates=[OUT/n for n in ['03_text_probe_precision_raw.json','03_text_probe_mask_raw.json','03_text_probe_tied_raw.json','03_text_probe_fast_raw.json','03_text_probe_overlay_raw.json','03_text_probe_raw.json']]
    existing=[p for p in candidates if p.exists()]
    assert existing,'Probe must be attempted and saved first'
    chosen=next((p for p in existing if read(p).get('execution_status')=='completed'),existing[0])
    probe=read(chosen);stages=probe.get('stages',[])
    if not stages:
        for line in probe.get('stdout_before_marker','').splitlines():
            if line.startswith('V4_STAGE_JSON='):
                try:stages.append(json.loads(line[len('V4_STAGE_JSON='):]))
                except json.JSONDecodeError:pass
    by_name={s['name']:s for s in stages}
    tests=[]
    def check(name,condition):tests.append({'name':name,'passed':bool(condition)})
    for row in read(OUT/'03_before_manifest.json')['files']:
        check('preserved:'+row['path'],hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'])
    for row in read(OUT/'02_source_manifest.json')['files']:
        check('step02_frozen:'+row['path'],hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'])
    for path in [OUT/n for n in ['inspect_step03.py','step03_text_probe.py','run_step03_probe.py','finalize_step03.py']]:
        ast.parse(path.read_text(encoding='utf-8'));check('syntax:'+path.name,True)
    for row in audit['sources']:
        check('remote_snapshot_hash:'+row['path'],hashlib.sha256(row['text'].encode()).hexdigest()==row['sha256'])
    h=audit['safetensors']['header']
    check('vlm_head_2048',h['paligemma_with_expert.paligemma.lm_head.weight']['shape']==[257152,2048])
    check('expert_head_not_vlm',h['paligemma_with_expert.gemma_expert.lm_head.weight']['shape']==[257152,1024])
    check('action_only_training_loss','return F.mse_loss(u_t, v_t, reduction="none")' in audit['sources'][4]['text'])
    check('vlm_inputs_embeds_branch','inputs_embeds[1] is None' in audit['sources'][5]['text'])
    check('probe_no_actions_or_training',probe.get('training') is False and probe.get('action_sampling') is False if 'training' in probe else probe.get('execution_status')=='failed')
    previous=read(OUT/'03_progress_before.json');progress=read(ROOT/'execution/v4_plan_progress.json')
    check('historical_gates_unchanged',previous['historical_v2_gates']==progress['historical_v2_gates'])
    check('later_steps_unstarted',all(s['status']=='未开始' for s in progress['steps'][3:]))
    assert all(t['passed'] for t in tests),[t for t in tests if not t['passed']]
    completed=probe.get('execution_status')=='completed'
    case_names=['main_task','next_subtask','next_after_grasp','same_class_instance']
    cases=[by_name[n] for n in case_names if n in by_name]
    parity=by_name.get('cache_position_and_causal_mask',{})
    text_engineering=completed and len(cases)==4 and all(s['loss_finite'] for s in cases) and parity.get('within_tolerance',False) and parity.get('future_answer_first_logit_difference',1) <= 0.25 and by_name.get('synthetic_image_path',{}).get('finite',False)
    controls=by_name.get('causal_precision_controls',{})
    if controls:
        text_engineering=text_engineering and controls['masked_future_change_max_diff']<=controls['identical_input_repeat_max_diff']+1e-6 and controls['unmasked_future_negative_control_max_diff']>controls['masked_future_change_max_diff'] and by_name.get('cache_argmax_control',{}).get('argmax_agrees',False)
    case_summary=[{'name':n,'generated_text':by_name[n]['generated_text'],'target_fixture':by_name[n]['target_fixture'],'exact_fixture_match':by_name[n]['generated_text'].strip()==by_name[n]['target_fixture'].strip(),'loss':by_name[n]['teacher_forcing_loss'],'stopped_at_eos':by_name[n]['stopped_at_eos'],'tokens_per_second':by_name[n]['tokens_per_second']} for n in case_names if n in by_name]
    gaps=['No verified main-task/subtask language-supervision ledger for this checkpoint.','Synthetic images and explicit text fixtures cannot establish scene-grounded planning.','OBJ_i text correctness cannot establish same-class low-level instance grounding.','Text backward/LoRA training, long-context cache, interaction prefix gradients and action regressions have not been tested.']
    if not text_engineering:gaps.insert(0,'Text runtime probe did not complete: '+str(probe.get('error')))
    elif not all(c['exact_fixture_match'] for c in case_summary):gaps.insert(0,'At least one greedy development answer differs from its expected fixture; review raw outputs, do not promote planning capability.')
    report={'step_id':'V4-03','audit_status':'COMPLETE_WITH_CAPABILITY_GAPS','selected_probe':str(chosen),'checkpoint':audit['safetensors']['path'],'checkpoint_bytes':audit['safetensors']['bytes'],'full_checkpoint_hash_verified':False,'vlm_hidden_size':2048,'planned_prefix_shape':['B',8,2048],'source_only_interface_available':True,'runtime_completed':completed,'text_loss_and_decode_engineering_pass':text_engineering,'cache_check':parity,'case_outputs':case_summary,'high_level_main_task_gate':'NOT_ESTABLISHED','high_level_subtask_gate':'NOT_ESTABLISHED','same_class_low_level_grounding_gate':'NOT_VERIFIED','image_probe_source':'synthetic black/white only','external_llm_used':False,'training_run':False,'action_sampling_run':False,'real_eeg':False,'physical_robot':False,'gaps':gaps,'integration_branch':{'preferred':'Same PaliGemma VLM: add separate causal text entry and observable high-level supervision; evaluate high-level LoRA before interaction prefix integration.','alternative_not_selected':'A verified text-capable checkpoint or separate VLM, explicitly report changed model boundary and compute.','training_budget':'unmeasured; requires development data and pilot, not inferred from inference speed','steps_allowed_without_gate':['V4-04 dataset protocol/development fixtures','V4-06 CPU module'],'steps_gated':['V4-07 integration success claim','end-to-end text training/closed-loop high-level success']}}
    report['image_probe_source']='synthetic constant zero/one normalized images only'
    report['runtime_stages']=stages
    report['precision_controls']=by_name.get('causal_precision_controls',{})
    report['cache_argmax_control']=by_name.get('cache_argmax_control',{})
    report['strict_numeric_equivalence_confirmed']=False
    report['numeric_scope']='Short-sequence BF16 coarse logit tolerance only. Single-position head readout vs batched teacher-forcing head can change GEMM rounding; no strict equivalence claim.'
    if completed:
        report['high_level_main_task_gate']='NOT_PASSED_ON_DEVELOPMENT_PROBES'
        report['high_level_subtask_gate']='NOT_PASSED_ON_DEVELOPMENT_PROBES'
    commands=[{'command':'python execution/v4/inspect_step03.py','scope':'read-only source/tensor-header audit','result':'saved; initial git timeout recorded and fixed with bounded exception handling'}]
    for path in reversed(existing):
        item=read(path)
        flag=' --precision-check' if 'precision' in path.name else ' --mask-overlay' if 'mask' in path.name else ' --tied-overlay' if 'tied' in path.name else ' --fast-overlay' if 'fast' in path.name else ' --existing-overlay' if 'overlay' in path.name else ''
        commands.append({'command':'python execution/v4/run_step03_probe.py'+flag,'trace':str(path),'execution_status':item.get('execution_status'),'remote_exit_code':item.get('remote_exit_code'),'error':item.get('error')})
    for name,flag in [('03_load_trace.json','--load-trace'),('03_io_diagnostic.json','--io-check')]:
        if (OUT/name).exists():commands.append({'command':'python execution/v4/run_step03_probe.py '+flag,'trace':str(OUT/name)})
    write(OUT/'03_command_log.json',commands)
    write(OUT/'03_text_capability.json',report)
    write(OUT/'03_verification.json',{'scope':'source/provenance/preservation checks, separate from neural capability gates','status':'PASS','checks':len(tests),'results':tests})
    doc=ROOT/'docs/v4/03_backbone_gate.txt'
    content=doc.read_text(encoding='utf-8-sig')
    content+='\n六、本轮实际结果\n审计完成，能力门槛保留缺口。最小文本运行完成：'+str(completed)+'；文本loss/解码工程检查：'+str(text_engineering)+'。\n'
    for row in case_summary:content+=row['name']+'：生成='+repr(row['generated_text'])+'；夹具期望='+repr(row['target_fixture'])+'；loss='+str(row['loss'])+'。\n'
    content+='本轮实际选用原始trace：'+chosen.name+'。主任务/子任务语义未建立可靠性，同类实例低层落地未验证。\n'
    if not completed:content+='运行错误：'+str(probe.get('error'))+'。\n'
    text(doc,content)
    handoff='V4-03交接（2026-10-17）\n审计已完成，能力门槛有缺口，不能宣称现有π0.5已经稳定生成主任务/subtask。\nVLM维数2048，后续交互前缀[B,8,2048]；当前动作训练入口没有主任务/子任务文本CE。\n严格加载、文本loss、EOS限长解码已实际运行；主任务输出1000、下一子任务连续0，未通过本轮开发提示。\n短序列BF16 cache argmax一致，仍不宣称严格数值等价或长上下文正确。\n结果：docs/v4/03_backbone_gate.txt、execution/v4/03_text_capability.json、原始probe trace。\n默认环境缺补丁；已有隔离overlay用于重跑；显式恢复checkpoint声明的embedding共享，修正探测mask类型，不改全局依赖或权重。\n下一步V4-04可做数据/开发夹具；步骤06可做CPU模块；步骤07等需先解决高层能力缺口。\n未训练、未调用动作、未接真实脑机和实体机器人；旧G2/G3/G4结论不升级。\n'
    text(OUT/'handoff_step03.txt',handoff)
    source_paths=[OUT/'inspect_step03.py',OUT/'run_step03_probe.py',OUT/'step03_text_probe.py',OUT/'finalize_step03.py',doc,OUT/'handoff_step03.txt',OUT/'03_remote_source_audit.json',OUT/'03_text_capability.json',OUT/'03_source_locations.json',OUT/'03_verification.json',OUT/'03_command_log.json']+existing+sorted((OUT/'03_sources').iterdir())
    source_paths+=[OUT/n for n in ['03_load_trace.json','03_io_diagnostic.json'] if (OUT/n).exists()]
    write(OUT/'03_source_manifest.json',{'step':'V4-03','files':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in source_paths]})
    now=datetime.now(timezone.utc).isoformat()
    write(OUT/'03_acceptance.json',{'step_id':'V4-03','status':'AUDIT_COMPLETE_WITH_CAPABILITY_GAPS','completed_at_utc':now,'audit_integrity_checks':len(tests),'audit_integrity_passed':True,'neural_capability_gate_passed':False,'runtime_completed':completed,'text_loss_decode_engineering_pass':text_engineering,'next_step':'V4-04','no_training_or_actions':True,'historical_gates_upgraded':False})
    progress['steps'][2].update(status='已完成',completion_scope='审计完成，能力门槛存在缺口',capability_gate_passed=False,completed_at_utc=now,actual_artifacts=['execution/v4/03_text_capability.json','docs/v4/03_backbone_gate.txt','execution/v4/03_source_manifest.json','execution/v4/03_acceptance.json','execution/v4/handoff_step03.txt'],verification={'scope':'capability audit; planning not established','status':'AUDIT_COMPLETE_WITH_CAPABILITY_GAPS','report':'execution/v4/03_text_capability.json','integrity_checks':len(tests)})
    progress.update(current_stage='V4-03_audit_complete_capability_gaps',next_step='V4-04',latest_handoff='execution/v4/handoff_step03.txt',development_capability_probe_run=True,formal_experiments_run=False,high_level_gate_passed=False)
    progress['new_experiments_run']=True
    progress['new_experiments_run_meaning']='V4-03 development capability probe attempted; not formal evaluation or training.'
    write(ROOT/'execution/v4_plan_progress.json',progress)
    text(ROOT.parent/'BCI_VLA当前计划入口_V4.txt','当前计划：V4-plan-2。\n2026-10-17：步骤01/02完成；步骤03审计完成，能力门槛有缺口。\n下一步V4-04的数据协议/开发夹具可继续；步骤07等不能绕过高层能力门槛。\n最新交接：D:\\Scene_Aware\\bci_vla_framework\\execution\\v4\\handoff_step03.txt。\n能力报告：D:\\Scene_Aware\\bci_vla_framework\\docs\\v4\\03_backbone_gate.txt。\n当前进度：D:\\Scene_Aware\\bci_vla_framework\\execution\\v4_plan_progress.json。\n仅有开发能力探测；新Transformer未实现/训练，高层规划未通过开发探测，实例动作能力未验证；真实EEG和实体动作排除。\n')
    print(json.dumps({'status':'AUDIT_COMPLETE_WITH_CAPABILITY_GAPS','integrity_checks':len(tests),'runtime_completed':completed,'engineering_text_pass':text_engineering,'selected_trace':chosen.name},ensure_ascii=False))
if __name__=='__main__':main()
