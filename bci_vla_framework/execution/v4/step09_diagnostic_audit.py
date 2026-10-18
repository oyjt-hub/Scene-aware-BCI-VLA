"""Audit a frozen conservative fallback after validation finds NO usable policy.

This is explicitly a diagnostic, not selection of an eligible/winning policy.
The original calibration result and all failed constraints remain unchanged.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from execution.v4.step09_calibrate_and_policy import (
    load, dump, sha, SCENARIOS, ConnectorPredictor, CalibratedPolicy,
    episode_summary, aggregate_base_summaries, online_audit_rows, online_audit_acceptance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous diagnostic evidence')
    args.output.mkdir(parents=True)
    report = load(ROOT/'results/calibration_result.json')
    if report['status'] != 'NO_POLICY_MEETS_DEVELOPMENT_CONSTRAINTS' or report['validation_selection']['selected_policy_id'] is not None:
        raise ValueError('Diagnostic fallback is only allowed after an explicit no-eligible result')
    scores = report['validation_selection']['scores']
    fixed = [k for k, s in scores.items() if s['variant'] == 'fixed_threshold']
    bound = report['online_audit_preregistered_criteria']['max_wrong_episode_rate']
    conservative = [k for k in fixed if scores[k]['wrong_episode_rate'] <= bound]
    if not conservative:
        raise ValueError('No fixed fallback meets even the validation error bound; do not deploy one')
    pid = min(conservative, key=lambda k: (-scores[k]['resolved_fraction'], scores[k]['wrong_episode_rate'],
                                           scores[k]['interaction_cost'], k))
    config = load(ROOT/'results/artifacts'/pid/'policy.json')
    artifact = report['calibrations'][pid]
    binding, temperatures = artifact['binding'], artifact['temperatures']
    import torch
    from safetensors.torch import load_file
    from bci_vla.calibration_v4 import evaluate_calibration, policy_hash
    from bci_vla.interaction_contract_v4.validation import canonical_digest
    from bci_vla.neural_interaction_v4.model import matched_models
    from bci_vla.evaluation_v4.runner import run_calibration_audit_episode, make_public
    from execution.v4.build_step04_data import generate
    torch.set_num_threads(4); torch.manual_seed(0)
    if policy_hash(config) != binding['policy_sha256']:
        raise ValueError('Frozen fallback policy SHA mismatch')
    weights = Path(report['checkpoint'])/'connector.safetensors'
    if sha(weights) != report['model_binding']['checkpoint_files']['connector.safetensors']:
        raise ValueError('Frozen connector SHA mismatch')
    model = matched_models()['Full'].eval().requires_grad_(False)
    model.load_state_dict(load_file(str(weights), device='cpu'), strict=True)
    predictor = ConnectorPredictor(model)
    captures = [c for c in load(args.capture_root/'captures/capture_manifest.json')['episodes'] if c['partition']=='calibration_audit']
    frozen = canonical_digest({'config': config, 'binding': binding, 'temperatures': temperatures})
    conditions, trajectories = [], []
    for capture in captures:
        for scenario in SCENARIOS:
            _, world = generate(deepcopy(capture), scenario)
            policy = CalibratedPolicy(predictor, config, binding, temperatures['intent'], temperatures['channel'], pid)
            trace = run_calibration_audit_episode(capture, world, policy, args.capture_root, horizon=24, declared_action='pick')
            summary = episode_summary(trace, world, capture['group_id'], scenario)
            conditions.append(summary); trajectories.append((trace, world, capture['group_id']))
            dump(args.output/'episodes'/f"{capture['base_episode_id']}_{scenario}.json",
                 {'summary': summary, 'trace': trace, 'policy_decisions': policy.decisions})
    base = aggregate_base_summaries(conditions, 'calibration_audit')
    rows = online_audit_rows(trajectories, predictor, binding)
    metrics = {task: evaluate_calibration(rows[task+'_rows'], temperatures[task], task=task,
               binding=binding, bins=config['calibration']['ece_bins']) for task in ('intent','channel')}
    acceptance = online_audit_acceptance(base, metrics, report['online_audit_preregistered_criteria'])
    controls = {}
    for control in ('STOP','CANCEL'):
        public = make_public(captures[0], 0., [], None, declared_action='pick')
        event = {'event_id':'direct-'+control, 'sequence':1, 'role':control, 'original_role':None,
                 'question':None, 'started_at_s':0., 'ended_at_s':0., 'valid':True,
                 'invalid_reason':None, 'evidence':None}
        public.update(history=[event], current_event_id=event['event_id'])
        public['public_state']['stop_latched'] = control == 'STOP'
        policy = CalibratedPolicy(predictor, config, binding, temperatures['intent'], temperatures['channel'], pid)
        before = predictor.forward_calls
        decision = policy.decide(public)
        controls[control] = {'decision':decision, 'predictor_forward_calls':predictor.forward_calls-before,
                             'passed':decision['action']=='WAIT' and predictor.forward_calls==before and decision.get('call_neural_module') is False}
    unchanged = frozen == canonical_digest({'config':config,'binding':binding,'temperatures':temperatures})
    result = {'status':'DIAGNOSTIC_ONLY_NO_ELIGIBLE_POLICY', 'diagnostic_only':True,
              'fallback_policy_id':pid, 'validation':scores[pid], 'binding':binding,
              'config':config, 'temperatures':temperatures, 'online_probability_metrics':metrics,
              'online_acceptance':acceptance, 'condition_summaries':conditions, 'base_episodes':base,
              'controls':controls, 'frozen_policy_unchanged':unchanged,
              'implementation_verified':unchanged and all(x['passed'] for x in controls.values()),
              'policy_usable':False, 'full_step09_accepted':False,
              'selected_policy_id':None, 'selection_or_temperature_retuned_on_audit':False,
              'formal_test_accessed':False, 'VLA_called':False, 'physical_actions':False,
              'scope':'public fixed-pick object confirmation; conservative no-eligible fallback diagnosis only',
              'source_script_sha256':sha(__file__), 'source_calibration_report_sha256':sha(ROOT/'results/calibration_result.json')}
    dump(args.output/'probability_rows.json', rows)
    dump(args.output/'diagnostic_result.json', result)
    print(json.dumps({'status':result['status'],'policy':pid,'implementation_verified':result['implementation_verified'],
                      'resolved':acceptance['resolved_fraction'],'wrong':acceptance['wrong_episode_rate']}), flush=True)


if __name__ == '__main__':
    main()
