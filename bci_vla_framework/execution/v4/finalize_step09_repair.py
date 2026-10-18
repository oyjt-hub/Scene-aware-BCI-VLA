"""Publish verified Step09 development evidence without upgrading research gates."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT.parent/'.runtime/step06_cpu'), str(PROJECT)]
OUT = PROJECT/'execution/v4'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--cpu-result', type=Path, default=OUT/'09_cpu_test_result_final.json')
    args = parser.parse_args()
    from bci_vla.calibration_v4 import policy_hash
    from bci_vla.calibration_v4.policy import _validate_config
    report_path = OUT/f'09_calibration_attempt{args.attempt}.json'
    report = load(report_path)
    if report['status'] == 'FAILED':
        raise ValueError('Do not publish failed execution as acceptance')
    cpu = load(args.cpu_result)
    trained = load(OUT/'09_trained_mechanism_audit.json')
    weights = load(OUT/'09_checkpoint_manifest3.json')
    training = load(OUT/'09_binding_attempt3.json')
    run = training['runs'][0]
    if not (cpu['status']=='PASS_CPU_MECHANISMS_ONLY' and trained['passed'] and weights['all_verified'] and report['binding_gate']['passed']):
        raise ValueError('Required repair mechanisms did not pass')
    archive = Path(report['local_evidence_archive'])
    if hashlib.sha256(archive.read_bytes()).hexdigest() != report['local_evidence_archive_sha256']:
        raise ValueError('Downloaded evidence archive SHA mismatch')
    evidence_dir = OUT/f'09_calibration_evidence{args.attempt}'
    evidence_dir.mkdir(exist_ok=True)
    wanted = {'calibration_result.json','binding_gate.json','online_audit_preregistered_criteria.json',
              'policy.json','temperatures.json','fixed_history_audit.json','validation_summary.json',
              'selected_policy.json','audit_result.json','probability_rows.json'}
    contract_checks = {'confirmations_checked':0,'passed':True,'scope':'selected-policy online development traces'}
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if name.endswith('/') or Path(name).name not in wanted:
                continue
            dest = (evidence_dir/name).resolve()
            if evidence_dir.resolve() not in dest.parents:
                raise ValueError('Archive path outside evidence directory')
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(name))
        for pid, artifact in report['calibrations'].items():
            config = json.loads(z.read(f'results/artifacts/{pid}/policy.json'))
            _validate_config(config)
            if policy_hash(config) != artifact['binding']['policy_sha256']:
                raise ValueError('Candidate policy SHA mismatch')
            for split in ('calibration_fit','calibration_audit'):
                rows = json.loads(z.read(f'results/artifacts/{pid}/{split}_rows.json'))
                if rows['binding'] != artifact['binding'] or any(
                        r['metadata']['binding'] != artifact['binding'] for task in ('intent','channel') for r in rows[task+'_rows']):
                    raise ValueError('Calibration header/row binding mismatch')
        selected_id = report['validation_selection']['selected_policy_id']
        if selected_id and args.attempt >= 6:
            from bci_vla.calibration_v4.evidence_gate import confirmation_evidence_gate
            config = json.loads(z.read(f'results/artifacts/{selected_id}/policy.json'))
            for name in z.namelist():
                if not name.startswith('results/online_audit/') or not name.endswith('.json'):
                    continue
                trace = json.loads(z.read(name))
                if not isinstance(trace,dict) or 'trace' not in trace:
                    continue
                for event in trace['trace']['journal']:
                    if event['kind']=='INTERACTION' and event['decision']['action']=='CONFIRM':
                        if confirmation_evidence_gate(event['input'],event['decision']['source_id'],config) is not None:
                            raise ValueError('A recorded confirmation bypassed its public evidence contract')
                        contract_checks['confirmations_checked'] += 1
    selected = report['validation_selection']['selected_policy_id']
    usable = bool(report.get('policy_operating_constraints_passed', False))
    primary = selected or next(iter(report['calibrations']))
    calibration = report['calibrations'][primary]
    diagnostic_path = OUT/f'09_diagnostic_attempt{args.attempt}.json'
    diagnostic = load(diagnostic_path) if diagnostic_path.exists() else None
    engineering = bool(report.get('step09_implementation_complete', False) or
                       diagnostic and diagnostic.get('implementation_verified'))
    activation = {'status':'SIMULATED_DEVELOPMENT_READY' if usable else 'NOT_ACTIVATED_DEVELOPMENT_CONSTRAINTS_FAILED',
                  'selected_policy_id':selected, 'enabled':usable, 'simulated_interaction_only':True,
                  'VLA_execution_enabled':False,'physical_actions':False,'formal_test_accessed':False,
                  'report':str(report_path.relative_to(PROJECT)), 'binding':calibration['binding'] if selected else None}
    activation['audit_role'] = report.get('audit_role','historical development audit')
    if selected:
        exact_config = load(evidence_dir/'results/selected_policy.json')['config']
        if policy_hash(exact_config) != calibration['binding']['policy_sha256']:
            raise ValueError('Selected policy hash changed')
        dump(PROJECT/'configs/v4/calibration_and_policy.json', exact_config)
    else:
        dump(PROJECT/'configs/v4/calibration_and_policy.json', {
            'version':'V4-09-calibrated-policy-catalog/0.1','status':'NO_ELIGIBLE_POLICY_NOT_ACTIVATED',
            'active_policy':None,'execution_authorized':False,'formal_test_accessed':False,
            'calibration_parameters_fitted':True,'fit_scope':'synthetic BCI on actual LIBERO captures',
            'candidate_evidence':str(evidence_dir.relative_to(PROJECT)),
            'report':str(report_path.relative_to(PROJECT)),
            'reproduction_template':'configs/v4/calibration_and_policy_template.json'})
    dump(PROJECT/'configs/v4/policy_activation.json', activation)
    audit = {'version':'V4-09-repair/0.1','updated_at_utc':datetime.now(timezone.utc).isoformat(),
             'engineering_repair_verified':engineering, 'step09_full_acceptance':usable,
             'status':'RESTRICTED_DEVELOPMENT_PASSED' if usable else 'CALIBRATION_REPAIRED_POLICY_CONSTRAINTS_FAILED',
             'cpu':cpu,'trained_mechanisms':trained,'checkpoint_manifest':'execution/v4/09_checkpoint_manifest3.json',
             'confirmation_contract_trace_audit':contract_checks,
             'binding_gate':report['binding_gate'],'language_exact':run['selected_validation']['exact'],
             'language_count':run['selected_validation']['count'],
             'remaining_language_failure_families':dict(Counter(t['family'] for t in run['selected_validation']['traces'] if not t['exact'])),
             'calibration':calibration,'policy_search':report['validation_selection'],
             'online_policy_audit':report.get('online_policy_audit'), 'diagnostic_only_audit':diagnostic,
             'activation':activation,'evidence_archive_sha256':report['local_evidence_archive_sha256'],
             'calibration_source_units':report['source_partition_counts'],
             'historical_gates_upgraded':False,'full_step08_complete':False,
             'full_VLA_closed_loop':False,'formal_test_accessed':False,'real_EEG':False,'physical_actions':False}
    dump(OUT/'09_repair_audit.json', audit)
    lines = ['V4-09 修复与开发复验报告（2026-10-18）','',
        '一、结论',
        ('受限合成域开发策略达到本轮点估计约束。' if usable else '对象绑定和校准代码已修复；策略尚未达到运行约束，步骤09整体验收未通过。'),
        f'工程路径是否完整复验：{engineering}；实际策略是否可用：{usable}。真实脑机、实体动作、正式test未使用。',
        '本轮仅验证公开固定pick动作下的对象选择和确认，不代表VLM主任务/缺槽推理闭环，也没有推进机器人动力学。','',
        '二、已修复的问题',
        '1. 在VLM接入边界加入公开对象编号嵌入，由预测的对象概率软混合；正确目标不进入模型输入。8个接入token按4诊断摘要+4绑定意图排列。',
        '2. 同场景覆盖全部候选目标及联合编号置换；降低学习率并补足原family均衡采样的训练覆盖。最终实际采用旧采样器2400步，已覆盖432/432行、72/72选择组、96/96编号对。',
        '3. 校准使用独立fit/audit分组，绑定模型、数据、schema、策略及实施代码SHA；无支持的通道概率保留null。',
        '4. 用户拒绝建议后必须重新询问；比较8问/3PROBE和12问/6PROBE预算，所有等待、无响应和拒绝保留。',
        '5. STOP/CANCEL优先旁路；在线审计使用真实calibration_audit分区，不冒充validation。',
        '6. 修正独立统计单位、零解决候选筛除及缓存审计文件的版本头；校准数值与每条记录的绑定均已核验。','',
        '三、实际模型复验',
        '对象切换完整组24/24、编号重绑定双侧正确24/24；均为seed0开发诊断，不是24个独立场景。',
        f"语言生成{run['selected_validation']['exact']}/{run['selected_validation']['count']}，8条声明冲突样本仍失败；不能称全部主任务语义通过。",
        f"CPU机制测试{cpu['tests_run']}项通过；真实选中连接器的PROBE对象引用置换造成prefix/候选/通道/交互输出最大差均为0；STOP/CANCEL触发模型调用0次。",
        '选中权重2个文件共41,244,424字节，均已取回本地并核验SHA。',
        '动作专家、主干权重和动作采样范围未扩展；模块及高层语言LoRA参与了开发训练。','',
        '四、合成域校准结果（固定历史分布）',
        'fit、audit、validation各8个底层回合、2个场景组，三者互斥；重复快照不作为新的独立受试/场景。']
    if args.attempt >= 6:
        lines.insert(2,'继续修复：使用此前已查看的校准审计组进行开发回归，不是新的独立测试。上轮25%错误结果与旧配置保存在execution/v4/09_repair_v2_before及attempt5不可变记录。')
        checks, freshness, window = (3,8,12) if args.attempt>=7 else (2,6,10)
        lines.append(f'本轮新规则：相同预算下对比plain与recheck。recheck要求至少2次近期同对象选择、最近一次选择之后至少{checks}次已知目标PROBE均正确且覆盖至少2个符号，再走原CONFIRM；失败保留WAIT/重询问，不修改模型预测为正确目标。')
        lines.append(f'所有新候选同为12问/6PROBE/4意图询问/4重试，选择新鲜度{freshness}秒、一致性窗口{window}秒；门控不读取隐藏换目标时间或真实通道参数。')
    for task, label in (('intent','意图'),('channel','通道')):
        fit = calibration['temperatures'][task]; checked = calibration['fixed_history_audit'][task]
        lines.append(f"{label}：T={fit['temperature']:.6f}；fit有效快照{fit['samples']}，audit有效快照{checked['samples']}。")
        lines.append(f"audit NLL {checked['before']['nll']:.6f}→{checked['after']['nll']:.6f}；Brier {checked['before']['brier']:.6f}→{checked['after']['brier']:.6f}；ECE {checked['before']['ece']:.6f}→{checked['after']['ece']:.6f}。")
    lines.extend(['通道定义仅是同映射/布局、已覆盖符号、+2秒的已知目标PROBE正确解码；不外推真人TASK/CONFIRM可靠性。',
        '同公开输入的隐藏反事实合并成soft outcome标签；未来probe结果仅在损失/审计端，不送预测器。','',
        '五、策略比较与冻结',
        f"本轮{len(report['validation_selection']['scores'])}个冻结候选；selected_policy={selected}；约束为独立底层回合任一错误提交比例≤10%，且任务解决率>0。",
        '同一底层回合的4种条件先取解决率均值，再按场景/回合宏平均；错误风险仍按回合任一次错误计。all-four同时成功另列，不膨胀独立N。',
        '8个独立回合中出现1个错误即12.5%，因此本轮10%点估计约束要求验证中零错；这不是统计风险保证。'])
    for pid, score in report['validation_selection']['scores'].items():
        lines.append(f"{pid}: 解决率={score['resolved_fraction']:.4f}, 错误回合率={score['wrong_episode_rate']:.4f}, N={score['episodes']}, 场景组={score['groups']}。")
    online = report.get('online_policy_audit', {})
    if 'acceptance' in online:
        accepted = online['acceptance']
        wrong_base = sum(x['wrong_submission'] for x in online['base_episodes'])
        correct_conditions = sum(x['resolved_correctly'] for x in online['condition_summaries'])
        lines.append(f"冻结后在线开发回归：条件解决{correct_conditions}/{len(online['condition_summaries'])}；底层错误回合{wrong_base}/{len(online['base_episodes'])}；宏平均解决率{accepted['resolved_fraction']:.4f}、错误回合率{accepted['wrong_episode_rate']:.4f}。")
        for task, label in (('intent','意图'),('channel','通道')):
            metric = online['probability_metrics'][task]
            lines.append(f"在线{label}校准NLL {metric['before']['nll']:.6f}→{metric['after']['nll']:.6f}；这不代替闭环风险验收。")
    lines.extend(['每种条件固定24秒，等待同样计时，所以完整时长恒为24秒；逐条件日志另报问题/等待/无响应/拒绝次数，不声称恒定时长有降低。',
        '辅助头策略没有因持续等待而获得成功计分；不宣称已学会主动澄清。',
        f"在线审计状态：{report.get('online_policy_audit',{}).get('status')}。",
        ('已选策略完成冻结后在线审计，审计后未改阈值和温度。' if selected else '没有可行候选时不自动启用策略；若另有保守候选审计，明确只用于诊断，不冒充合格策略。'),'',
        '六、尝试记录与边界',
        '绑定attempt1在真实tokenizer对齐检查前失败、无正式训练更新；attempt2进行了三个机制对照，未过对象组门槛；attempt3延长覆盖后通过。',
        '最终成功绑定训练使用加权完整答案CE、残差尺度0.1。新增additive alias bonus、完整行覆盖采样器和残差1.0配置仅为未执行的后续草案，不计本轮实际结果。',
        '校准attempt1使用all-four严格解决口径；attempt2改为独立回合内条件均值；attempt3扩大严格阈值；attempt4修正缓存输出版本头；attempt5在validation细化0.85到0.90间阈值。全部结果和源代码快照保留。',
        *(['attempt6针对已观察到的单次含噪接受风险增加公开证据合约，先在validation比较同预算策略，随后冻结选中策略作开发回归；本轮没有再训练神经模块，不声称神经策略学会了这套规则。'] if args.attempt>=6 else []),
        *(['attempt6的两次检查方案：validation解决9/32、错误0/8；开发回归解决9/32、错误1/8。attempt7按新的三次检查协议与相同时效原策略比较，所有旧失败保留。'] if args.attempt>=7 else []),
        '校准分区多次用于开发，不能称为多次独立正式测试；尚未读取预留test。',
        '当前图像与物体来自实际LIBERO冻结观察，bbox为特权投影、特征为6维RGB统计，BCI为模拟；Florence/SAM及真人EEG效果不在本轮结论内。',
        '完整放置阶段、含噪隐藏意图泛化、其他基线的同等新绑定训练、VLM交互语言闭环及实际VLA动作仍待后续验证；步骤08整体和旧G2/G3/G4不升级。','',
        '后续仍需回到步骤08补齐原始含噪轨迹下的意图更新/通道变化训练及冲突声明语义；先在train/validation改进，再重新校准与冻结审计，不能继续根据本次audit细调阈值并称独立通过。','',
        '七、复现与证据',
        f'主报告：execution/v4/09_calibration_attempt{args.attempt}.json',
        '修复审计：execution/v4/09_repair_audit.json',
        '真实权重机制：execution/v4/09_trained_mechanism_audit.json',
        '本地权重：runs/neural_interaction_v4/dev/step09_attempt3/Full_seed0/step2400/',
        f'完整逐条件日志归档：execution/v4/09_calibration_results{args.attempt}.zip',
        '冻结状态：configs/v4/policy_activation.json；不可绕过enabled字段直接启用候选。',
        '复现模板：configs/v4/calibration_and_policy_template.json；code_bundle及manifest保存每轮完整实际来源。'])
    (PROJECT/'docs/v4/09_policy_validation.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    progress_path = PROJECT/'execution/v4_plan_progress.json'
    before = OUT/'09_progress_before.json'
    if not before.exists():
        before.write_bytes(progress_path.read_bytes())
    progress = load(progress_path)
    for step in progress['steps']:
        if step['step_id']=='V4-08':
            step['object_binding_remediation'] = {'passed':True,'scope':'seed0 restricted public-object binding only',
                'choice_groups':'24/24','joint_alias_pairs':'24/24','report':'execution/v4/09_repair_audit.json',
                'full_step08_complete':False,'conflicting_declarations_language_passed':False}
        if step['step_id']=='V4-09':
            step['status'] = '受限开发范围完成' if usable else '部分完成（绑定和校准修复，策略运行约束未通过）'
            step['actual_artifacts'] = ['configs/v4/calibration_and_policy.json','configs/v4/policy_activation.json',
                'docs/v4/09_policy_validation.txt','execution/v4/09_repair_audit.json',
                f'execution/v4/09_calibration_attempt{args.attempt}.json','execution/v4/09_checkpoint_manifest3.json']
            step['verification'] = {'engineering_repaired':engineering,'policy_constraints_passed':usable,
                'CPU_tests':cpu['tests_run'],'formal_test':False,'full_VLA_closed_loop':False,
                'scope':'simulated BCI, public fixed-pick object confirmation; actual frozen LIBERO captures'}
    progress.update(current_stage='V4-09_development_repaired' if usable else 'V4-09_calibration_repaired_policy_constraints_failed',
        next_step='V4-10' if usable else 'V4-09',latest_handoff='execution/v4/handoff_step09.txt',
        authorization_this_turn='修复步骤9；延续排除真实EEG和实体动作，允许开发训练补救及fit/audit/validation复验',
        calibration_run=True, object_binding_remediation_passed=True, full_step08_complete=False,
        step09_engineering_repaired=engineering, step09_policy_constraints_passed=usable,
        full_VLA_closed_loop_run=False,formal_experiments_run=False)
    dump(progress_path, progress)
    handoff = ('V4-09：'+('受限开发策略通过。' if usable else '校准和绑定已修复，策略运行门槛未通过。')+'\n'
        f"CPU {cpu['tests_run']}项通过；对象组24/24、编号对24/24；语言112/120，声明冲突8条仍失败。\n"
        f"选中策略：{selected}；策略可用：{usable}；禁止把校准指标改善称为完整BCI→VLA成功。\n"
        '报告docs/v4/09_policy_validation.txt；状态execution/v4/09_repair_audit.json。\n'
        '下一步按实际策略失败与缺失监督处理；不从正式test调参；steps08广义监督/放置仍缺，旧G2/G3/G4保留。\n'
        'config实际启用状态见configs/v4/policy_activation.json。真实EEG/实体动作排除。\n')
    (OUT/'handoff_step09.txt').write_text(handoff,encoding='utf-8')
    (OUT/'handoff.txt').write_text(handoff,encoding='utf-8')
    for relative in ('execution/handoff.txt','execution/worklog.txt','README.txt'):
        with (PROJECT/relative).open('a',encoding='utf-8') as f:
            f.write('\n2026-10-18 V4-09修复更新：'+handoff)
    print(json.dumps({'status':audit['status'],'engineering':engineering,'usable':usable,
                      'report':str(PROJECT/'docs/v4/09_policy_validation.txt')},ensure_ascii=False))


if __name__=='__main__':
    main()
