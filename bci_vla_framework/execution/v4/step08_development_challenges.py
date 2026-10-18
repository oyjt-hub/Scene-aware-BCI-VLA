"""Post-training evidence/alias binding probes on allowed validation groups only."""
import os
os.environ.update(JAX_PLATFORMS='cpu', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TORCHDYNAMO_DISABLE='1')
import argparse, hashlib, json, sys, time, traceback
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(8*1024**2):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-job', type=Path, required=True)
    parser.add_argument('--capture-root', type=Path, required=True)
    args = parser.parse_args()
    output = ROOT / 'challenge_result.json'
    assert not output.exists(), 'Preserve challenge evidence'
    result = {'scope': 'Post-training validation-only diagnostic; no model selection or weight update',
        'formal_test_accessed': False, 'calibration_accessed': False, 'optimizer_steps': 0,
        'action_sampling': False, 'runs': []}
    start = time.monotonic()
    try:
        import torch, numpy as np, sentencepiece as spm
        from PIL import Image
        from safetensors import safe_open
        from safetensors.torch import load_file, load_model
        from transformers.modeling_utils import no_init_weights
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from bci_vla.high_level_v4.lora import install_high_level_lora
        from bci_vla.neural_interaction_v4.model import matched_models
        from bci_vla.neural_interaction_v4.tensorizer import collate_public
        from bci_vla.prefix_integration_v4.adapter import Pi05HighLevelPrefix
        from bci_vla.training_v4.data import load_partition
        from bci_vla.training_v4.challenges import build_development_challenges
        from bci_vla.training_v4.text_baseline import build_b2_prompt
        from bci_vla.interaction_contract_v4.validation import canonical_digest
        torch.set_num_threads(4)
        upload = json.loads((ROOT/'code_manifest.json').read_text())
        assert all(sha(ROOT/row['path'])==row['sha256'] for row in upload['files']), 'Challenge source hash mismatch'
        report = json.loads((args.training_job/'results/training_result.json').read_text())
        config = report['config']
        validation = load_partition(args.training_job/'training_data', 'validation')
        challenge = build_development_challenges(validation)
        result['counts'] = challenge['counts']
        checkpoint = '/home/oyjt/workspace/RLinf-RECAP/models/baseline_sft/RLinf-Pi05-LIBERO-SFT/model.safetensors'
        assert sha(checkpoint) == config['base_sha256']
        with no_init_weights():
            base = PI0Pytorch(Pi0Config(pi05=True, dtype='bfloat16'))
        with safe_open(checkpoint, framework='pt', device='cpu') as f:
            assert f.metadata()['paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight'] == 'paligemma_with_expert.paligemma.lm_head.weight'
        base.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight = base.paligemma_with_expert.paligemma.lm_head.weight
        load_model(base, checkpoint, strict=True, device='cpu')
        device = torch.device('cuda:0'); base.eval().requires_grad_(False).to(device)
        adapters = install_high_level_lora(base, rank=16, alpha=32)
        bridge = Pi05HighLevelPrefix(base, {}, engineering_only=True, checkpoint_tag=config['base_sha256'], high_level_adapters=adapters)
        sp = spm.SentencePieceProcessor(model_file='/home/oyjt/.cache/openpi/big_vision/paligemma_tokenizer.model')
        image_cache = {}
        for trained in report['runs']:
            if trained['status'] != 'DEVELOPMENT_TRAINING_COMPLETE':
                continue
            method = trained['method']; seed = trained['seed']; connector = None
            folder = Path(trained['selected_checkpoint'])
            record = next(c for c in trained['checkpoints'] if c['step'] == trained['selected_step'])
            assert sha(folder/'language_lora.safetensors') == record['files']['language_lora.safetensors']
            weights = load_file(str(folder/'language_lora.safetensors'))
            with torch.no_grad():
                for name, module in adapters.items():
                    for side in ('A','B'):
                        getattr(module, side).copy_(weights[name+'.'+side].to(device))
            bridge.adapter_sha256 = sha(folder/'language_lora.safetensors')
            if method != 'B2':
                assert sha(folder/'connector.safetensors') == record['files']['connector.safetensors']
                models = matched_models(); connector = models[method].to(device).eval()
                connector.load_state_dict(load_file(str(folder/'connector.safetensors'))); del models
            for p in base.parameters():
                p.requires_grad_(False)
            decoded = {}

            @torch.no_grad()
            def predict(row, off=False):
                public = row['public']; language = row['supervision']['language']
                key = (canonical_digest(public), language['prompt'], off)
                if key in decoded:
                    return decoded[key]
                prompt = language['prompt']
                if method == 'B2' and not off:
                    prompt = build_b2_prompt(public, prompt, tokenizer=lambda t:sp.encode(t,add_bos=True), max_tokens=config['max_text_tokens'])['prompt']
                ids = torch.tensor([sp.encode(prompt, add_bos=True)], dtype=torch.long, device=device)
                ikey = (public['scene']['image_ref'], row['supervision']['source']['image_sha256'])
                if ikey not in image_cache:
                    path = args.capture_root/ikey[0]
                    assert sha(path) == ikey[1]
                    rgb = np.array(Image.open(path).convert('RGB').resize((224,224)), copy=True)
                    image_cache[ikey] = torch.tensor(rgb, device=device).permute(2,0,1)[None].float()/127.5-1.
                x, metadata = collate_public([public]); x = {k:v.to(device) for k,v in x.items()}
                soft = connector(x)['prefix'] if connector is not None and not off else None
                pack = bridge.prepare([image_cache[ikey]], [torch.tensor([language['visual']],dtype=torch.bool,device=device)],
                    ids, torch.ones_like(ids,dtype=torch.bool), soft_tokens=soft, enabled=soft is not None)
                generation = bridge.generate(public, pack, eos_id=sp.eos_id(), max_new_tokens=config['max_answer_tokens'])
                text = sp.decode(generation['token_ids']).strip()
                decoded[key] = {'sample_id':public['sample_id'], 'generated':text, 'target':language['target'],
                    'exact':text==language['target'], 'EOS':generation['stopped_at_eos'], 'text_tokens':ids.shape[1]}
                return decoded[key]

            run = {'method':method,'seed':seed,'selected_step':trained['selected_step'],'pairs':{},'auxiliary':{}}
            for kind in ('paired_evidence','joint_alias'):
                traces = []
                for pair in challenge[kind]:
                    a = predict(pair['first']); b = predict(pair['second'])
                    off = predict(pair['first'], off=True)
                    traces.append({'family':pair['family'], 'invariants':pair['invariants'], 'first':a, 'second':b, 'first_without_interaction':off,
                        'both_correct':a['exact'] and b['exact'], 'output_changed':a['generated']!=b['generated'],
                        'expected_changed':a['target']!=b['target']})
                run['pairs'][kind] = {'both_correct':sum(t['both_correct'] for t in traces),'count':len(traces),
                    'output_changed':sum(t['output_changed'] for t in traces),'traces':traces}
            if connector is not None:
                pointer, channel, slot, abstain = [], [], [], []
                confusion = {name:{'TP':0,'TN':0,'FP':0,'FN':0} for name in ('ACTION','SOURCE','DESTINATION','ABSTAIN')}
                def add_confusion(name, predicted, target):
                    cell = ('TP' if target else 'FP') if predicted else ('FN' if target else 'TN')
                    confusion[name][cell] += 1
                with torch.no_grad():
                    for row in validation:
                        x, metadata = collate_public([row['public']]); x = {k:v.to(device) for k,v in x.items()}
                        pred = connector(x); labels = row['private_labels']['labels']; aux = row['supervision']['auxiliary']
                        target = labels['intent']
                        if target['mask'] and target['mode']=='POINT':
                            j = int(pred['candidate_logits'][0].argmax()); ids=metadata[0]['instance_ids']
                            pointer.append(j<len(ids) and ids[j] in target['candidate_ids'])
                        if labels['channel']['mask']:
                            assert bool(pred['reliability_valid'][0])
                            channel.append((float(pred['reliability_probability'][0])-float(labels['channel']['next_probe_success']))**2)
                        for j, valid in enumerate(aux['missing_slots_mask']):
                            if valid:
                                predicted = bool(pred['missing_slot_logits'][0,j]>0); target = bool(aux['missing_slots'][j])
                                slot.append(predicted==target)
                                add_confusion(('ACTION','SOURCE','DESTINATION')[j], predicted, target)
                        if aux['abstain_mask']:
                            predicted = bool(pred['abstain_logit'][0]>0); target = bool(aux['abstain'])
                            abstain.append(predicted==target); add_confusion('ABSTAIN', predicted, target)
                run['auxiliary'] = {'pointer':{'correct':sum(pointer),'count':len(pointer)},
                    'channel_Brier':{'mean':sum(channel)/len(channel) if channel else None,'count':len(channel)},
                    'train_frequency_channel_Brier_reference': 0.19140625,
                    'missing_slots':{'correct':sum(slot),'count':len(slot)},
                    'abstain_at_uncalibrated_0_5':{'correct':sum(abstain),'count':len(abstain)},
                    'confusion_at_uncalibrated_0_5':confusion}
            result['runs'].append(run)
            print('CHALLENGE_STAGE='+json.dumps({'method':method,'seed':seed,
                'paired':run['pairs']['paired_evidence']['both_correct'],'paired_n':run['pairs']['paired_evidence']['count'],
                'alias':run['pairs']['joint_alias']['both_correct'],'alias_n':run['pairs']['joint_alias']['count']}), flush=True)
            connector=None; decoded.clear(); bridge.reset_generation()
            import gc
            gc.collect(); torch.cuda.empty_cache()
        result.update(status='DEVELOPMENT_CHALLENGES_COMPLETE', elapsed_s=time.monotonic()-start,
            no_weight_updates=True, full_hidden_intent_robustness_proven=False)
    except Exception as exc:
        result.update(status='FAIL',error=str(exc),traceback=traceback.format_exc(),elapsed_s=time.monotonic()-start)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print('CHALLENGE_RESULT='+json.dumps({k:result.get(k) for k in ('status','error','elapsed_s')}),flush=True)
    if result['status']!='DEVELOPMENT_CHALLENGES_COMPLETE':
        raise SystemExit(1)


if __name__=='__main__':
    main()
