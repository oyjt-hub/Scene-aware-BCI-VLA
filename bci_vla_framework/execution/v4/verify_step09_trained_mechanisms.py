"""Verify PROBE semantic isolation and direct controls with collected weights."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
WORKSPACE = PROJECT.parent
sys.path[:0] = [str(WORKSPACE/'.runtime/step06_cpu'), str(PROJECT)]


def main():
    import torch
    from safetensors.torch import load_file
    from bci_vla.neural_interaction_v4.model import matched_models
    from bci_vla.neural_interaction_v4.tensorizer import collate_public
    from bci_vla.interaction_contract_v4.validation import canonical_digest
    from execution.v4.step09_calibrate_and_policy import ConnectorPredictor, CalibratedPolicy
    torch.set_num_threads(4)
    manifest = json.loads((PROJECT/'execution/v4/09_checkpoint_manifest3.json').read_text())
    row = next(x for x in manifest['files'] if x['name']=='connector.safetensors')
    weights = Path(row['local_path'])
    actual = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert actual == row['sha256'] and row['verified']
    model = matched_models()['Full'].eval().requires_grad_(False)
    model.load_state_dict(load_file(str(weights)), strict=True)
    records = json.loads((PROJECT/'execution/v4/06_fixtures/public.json').read_text())
    original = deepcopy(next(p for p in records if p['public_state']['diagnostics']['status']=='SUFFICIENT'))
    changed = deepcopy(original)
    ids = [o['instance_id'] for o in changed['scene']['objects']]
    for events in (changed['history'], changed['public_state']['diagnostics']['receipts']):
        for event in events:
            q = event.get('question')
            if q and q['role']=='PROBE':
                for option in q['options']:
                    option['reference'] = ids[(ids.index(option['reference'])+1)%len(ids)]
    x, _ = collate_public([original]); y, _ = collate_public([changed])
    assert canonical_digest(original) != canonical_digest(changed)
    assert all(torch.equal(x[k],y[k]) for k in x)
    with torch.inference_mode():
        a, b = model(x), model(y)
    deltas = {key:float((a[key]-b[key]).abs().max()) for key in ('prefix','candidate_logits','reliability_logit','interaction_logits')}
    predictor = ConnectorPredictor(model)
    controls = {}
    for control in ('stop','cancel'):
        public = json.loads((PROJECT/f'examples/v4/{control}_clears_task.json').read_text())
        # Deliberately missing calibration/config: direct controls must bypass
        # model inference and all probabilistic thresholds before validation.
        policy = CalibratedPolicy(predictor, {}, None, None, None, 'direct-control-audit')
        result = policy.decide(public)
        controls[control] = {'action':result['action'], 'model_calls':predictor.forward_calls,
                             'route':result['control_route'], 'execution_authorized':result['execution_authorized']}
        assert result['action']=='WAIT' and result['call_neural_module'] is False and predictor.forward_calls==0
    result = {'status':'PASS_TRAINED_CONNECTOR_MECHANISMS', 'connector_sha256':actual,
              'probe_reference_changed':True, 'tensor_inputs_identical':True, 'output_max_abs_deltas':deltas,
              'controls':controls, 'passed':all(v==0 for v in deltas.values()),
              'scope':'Actual selected trained connector; PROBE object references only, not hidden-intent correctness or VLM/robot success',
              'real_eeg':False,'VLA_called':False,'formal_test_accessed':False}
    path = PROJECT/'execution/v4/09_trained_mechanism_audit.json'
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'deltas':deltas,'control_model_calls':predictor.forward_calls}))


if __name__ == '__main__':
    main()
