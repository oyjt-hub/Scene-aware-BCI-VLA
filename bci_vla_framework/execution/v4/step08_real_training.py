"""Bounded, auditable real-Pi05 development training; never dispatches actions."""
import os
os.environ.update(JAX_PLATFORMS='cpu', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TORCHDYNAMO_DISABLE='1')
import argparse, copy, hashlib, json, math, random, sys, time, traceback
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(8 * 1024 ** 2):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/v4/train_step08_pilot.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Append-only run; choose a new output path')
    args.output.mkdir(parents=True)
    config = json.loads(args.config.read_text())
    report = {'scope': config['scope'], 'status': 'STARTED', 'config': config,
              'formal_test_accessed': False, 'calibration_accessed': False,
              'action_sampling': False, 'physical_actions': False, 'runs': []}
    start = time.monotonic()

    def event(name, **values):
        row = {'event': name, 'elapsed_s': time.monotonic() - start, **values}
        with (args.output / 'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        print('TRAIN_STAGE=' + json.dumps(row), flush=True)
        dump(args.output / 'progress.json', report)

    try:
        import numpy as np
        import torch
        import sentencepiece as spm
        from PIL import Image
        from safetensors import safe_open
        from safetensors.torch import load_model, load_file, save_file
        from transformers.modeling_utils import no_init_weights
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from bci_vla.high_level_v4.lora import install_high_level_lora, adapter_state
        from bci_vla.neural_interaction_v4.model import matched_models
        from bci_vla.neural_interaction_v4.tensorizer import collate_public
        from bci_vla.prefix_integration_v4.adapter import Pi05HighLevelPrefix, PackedPrefix
        from bci_vla.training_v4.data import load_partition
        from bci_vla.training_v4.engine import (compute_training_losses, episode_balanced_schedule,
            validation_selection, scene_group_mean, parameter_snapshot, parameter_change_audit,
            backward_and_step)
        from bci_vla.training_v4.text_baseline import build_b2_prompt
        torch.set_num_threads(4)
        torch.manual_seed(0)
        device = torch.device('cuda:0')
        assert torch.cuda.mem_get_info()[0] > 26 * 1024 ** 3, 'Insufficient free GPU memory'
        upload = json.loads((ROOT / 'code_manifest.json').read_text())
        assert all(sha(ROOT / row['path']) == row['sha256'] for row in upload['files']), 'Uploaded source hash mismatch'
        train = load_partition(args.data, 'train')
        validation = load_partition(args.data, 'validation')
        data_manifest = json.loads((args.data / 'manifest.json').read_text())
        assert all(sha(args.data / row['path']) == row['sha256'] for row in data_manifest['files']), 'Training file hash mismatch'
        assert not ({r['supervision']['source']['group_id'] for r in train} &
                    {r['supervision']['source']['group_id'] for r in validation})
        assert all(r['supervision']['language']['mask'] for r in train + validation), 'This pilot filters to valid language rows'
        val = [validation[i] for i in validation_selection(validation, seed=0, max_examples=config['validation_examples'])]
        event('data_loaded', train=len(train), validation=len(val),
              families=dict(Counter(r['supervision']['family'] for r in train)),
              data_manifest_sha256=sha(args.data / 'manifest.json'))
        language = Path('/home/oyjt/Scene-aware-BCI-VLA-runs/v4_step03R_language_20261017_v1')
        checkpoint = Path('/home/oyjt/workspace/RLinf-RECAP/models/baseline_sft/RLinf-Pi05-LIBERO-SFT/model.safetensors')
        initial_adapter = language / 'high_level_lora_step240.safetensors'
        assert sha(checkpoint) == config['base_sha256']
        assert sha(initial_adapter) == config['initial_language_adapter_sha256']
        with no_init_weights():
            base = PI0Pytorch(Pi0Config(pi05=True, dtype='bfloat16'))
        with safe_open(checkpoint, framework='pt', device='cpu') as f:
            meta = f.metadata()
        alias = 'paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight'
        source = 'paligemma_with_expert.paligemma.lm_head.weight'
        assert meta[alias] == source
        base.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight = base.paligemma_with_expert.paligemma.lm_head.weight
        load_model(base, str(checkpoint), strict=True, device='cpu')
        base.eval().requires_grad_(False).to(device)
        adapters = install_high_level_lora(base, rank=16, alpha=32)
        initial = load_file(str(initial_adapter))
        capability = {'high_level_main_task_gate': 'PASSED', 'high_level_subtask_gate': 'PASSED',
                      'adapter': {'sha256': config['initial_language_adapter_sha256'], 'parameters': 3022848},
                      'base_checkpoint': {'sha256': config['base_sha256']}}
        bridge = Pi05HighLevelPrefix(base, capability, checkpoint_tag=config['base_sha256'],
            high_level_adapters=adapters, adapter_sha256=config['initial_language_adapter_sha256'])
        lora = {f'{name}.{side}': getattr(module, side) for name, module in adapters.items() for side in ('A', 'B')}
        assert set(initial) == set(lora)
        assert all(initial[k].shape == p.shape and torch.isfinite(initial[k]).all() for k, p in lora.items())
        lora_ids = {id(p) for p in lora.values()}
        frozen_versions = {name: p._version for name, p in base.named_parameters() if id(p) not in lora_ids}
        expert_shapes = {name: list(p.shape) for name, p in base.paligemma_with_expert.gemma_expert.named_parameters()}
        sp = spm.SentencePieceProcessor(model_file='/home/oyjt/.cache/openpi/big_vision/paligemma_tokenizer.model')
        report['tokenizer_sha256'] = sha('/home/oyjt/.cache/openpi/big_vision/paligemma_tokenizer.model')
        encode = lambda text: sp.encode(text, add_bos=True)
        image_cache, native_cache = {}, {}

        def reset_lora():
            bridge.reset_generation()
            bridge.adapter_sha256 = config['initial_language_adapter_sha256']
            with torch.no_grad():
                for name, p in lora.items():
                    p.copy_(initial[name].to(device))
            assert all(torch.equal(p.detach().cpu(), initial[name]) for name, p in lora.items())
            for p in base.parameters():
                p.requires_grad_(id(p) in lora_ids)
                p.grad = None

        def inputs(row, method):
            public = row['public']; lang = row['supervision']['language']
            prompt = lang['prompt']
            if method == 'B2':
                prompt = build_b2_prompt(public, prompt, tokenizer=encode,
                    max_tokens=config['max_text_tokens'])['prompt']
            ids = encode(prompt)
            if len(ids) > config['max_text_tokens']:
                raise ValueError(f'No truncation: {len(ids)} text tokens')
            key = (public['scene']['image_ref'], bool(lang['visual']), row['supervision']['source']['image_sha256'])
            if key not in image_cache:
                path = args.capture_root / key[0]
                if not path.is_file() or sha(path) != key[2]:
                    raise ValueError('Actual image binding mismatch')
                rgb = np.array(Image.open(path).convert('RGB').resize((224, 224)), copy=True)
                image_cache[key] = (torch.tensor(rgb, device=device).permute(2, 0, 1)[None].float() / 127.5 - 1.,
                    torch.tensor([key[1]], dtype=torch.bool, device=device))
            image, imask = image_cache[key]
            native_key = (key, prompt)
            if native_key not in native_cache:
                t = torch.tensor([ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    p = bridge.prepare([image], [imask], t, torch.ones_like(t, dtype=torch.bool), enabled=False)
                native_cache[native_key] = PackedPrefix(p.embeddings.detach().cpu(), p.valid.cpu(), p.ar.cpu(), False, p.native_length)
            target = torch.tensor([sp.encode(lang['target'], add_eos=True)], dtype=torch.long, device=device)
            tensors, metadata = collate_public([public])
            tensors = {k: v.to(device) for k, v in tensors.items()}
            cached = native_cache[native_key]
            native = PackedPrefix(cached.embeddings.to(device), cached.valid.to(device), cached.ar.to(device), False, cached.native_length)
            return native, target, tensors, metadata, len(ids)

        def pack(native, soft):
            if soft is None:
                return native
            soft = soft.to(device=device, dtype=native.embeddings.dtype)
            return PackedPrefix(torch.cat([native.embeddings, soft], 1),
                torch.cat([native.valid, torch.ones(1, 8, dtype=torch.bool, device=device)], 1),
                torch.cat([native.ar, torch.zeros(1, 8, dtype=torch.bool, device=device)], 1), True, native.native_length)

        def forward(row, method, connector):
            native, target, x, metadata, text_length = inputs(row, method)
            output = connector(x) if connector is not None else None
            ce = bridge.teacher_forcing(pack(native, output['prefix'] if output is not None else None),
                target, torch.ones_like(target, dtype=torch.bool), eos_id=sp.eos_id())['loss']
            terms = compute_training_losses(output, [row], metadata, language_loss=ce,
                auxiliary_enabled=connector is not None, weights=config['loss_weights'])
            return terms, output, text_length

        @torch.no_grad()
        def evaluate(rows, method, connector, decode=False):
            if connector is not None:
                connector.eval()
            losses, metrics, traces = [], defaultdict(list), []
            lengths = []
            for row in rows:
                terms, output, length = forward(row, method, connector)
                losses.append(float(terms['language'])); lengths.append(length)
                family = row['supervision']['family']
                metrics[family].append(float(terms['language']))
                if decode:
                    native, target, x, metadata, _ = inputs(row, method)
                    result = bridge.generate(row['public'], pack(native, None if output is None else output['prefix']),
                        eos_id=sp.eos_id(), max_new_tokens=config['max_answer_tokens'])
                    text = sp.decode(result['token_ids']).strip()
                    record = {'family': family, 'sample_id': row['public']['sample_id'],
                        'scene_group': row['supervision']['source']['group_id'],
                        'target': row['supervision']['language']['target'], 'generated': text,
                        'exact': text == row['supervision']['language']['target'], 'EOS': result['stopped_at_eos'],
                        'language_CE': float(terms['language']), 'text_tokens': length}
                    if output is not None:
                        intent = row['private_labels']['labels']['intent']
                        if intent['mask'] and intent['mode'] == 'POINT':
                            j = int(output['candidate_logits'][0].argmax())
                            predicted = metadata[0]['instance_ids'][j] if j < len(metadata[0]['instance_ids']) else None
                            record['pointer_exact'] = predicted in intent['candidate_ids']
                        record['abstain_probability'] = float(output['abstain_logit'].sigmoid()[0])
                    traces.append(record)
            result = {'language_CE': scene_group_mean(losses, rows),
                'family_language_CE': {k: sum(v) / len(v) for k, v in metrics.items()},
                'text_tokens': {'min': min(lengths), 'max': max(lengths), 'mean': sum(lengths) / len(lengths)}}
            if decode:
                result.update(exact=sum(r['exact'] for r in traces), count=len(traces), traces=traces)
            return result

        event('strict_checkpoint_loaded', base_frozen_except_language_lora=True,
              lora_parameters=sum(p.numel() for p in lora.values()), gpu_free_bytes=torch.cuda.mem_get_info()[0])
        # Real tokenizer preflight, no hidden overflow/truncation. This happens before optimizer work.
        token_lengths = {}
        for method in config['methods']:
            values = []
            for row in train + val:
                prompt = row['supervision']['language']['prompt']
                if method == 'B2':
                    prompt = build_b2_prompt(row['public'], prompt, tokenizer=encode,
                        max_tokens=config['max_text_tokens'])['prompt']
                values.append(len(encode(prompt)))
            if max(values) > config['max_text_tokens']:
                raise ValueError('Text preflight overflow')
            token_lengths[method] = {'min': min(values), 'max': max(values), 'mean': sum(values) / len(values)}
        report['text_token_preflight'] = token_lengths
        event('text_token_preflight', lengths=token_lengths)

        for seed in config['seeds']:
            schedule = episode_balanced_schedule(train, seed=seed, steps=config['optimizer_steps'])
            schedule_digest = hashlib.sha256(json.dumps([train[i]['public']['sample_id'] for i in schedule]).encode()).hexdigest()
            for method in config['methods']:
                run = {'method': method, 'seed': seed, 'status': 'STARTED', 'optimizer_steps': 0,
                       'schedule_sha256': schedule_digest, 'history': [], 'checkpoints': []}
                report['runs'].append(run)
                folder = args.output / f'{method}_seed{seed}'
                folder.mkdir()
                run_start = time.monotonic()
                connector = None
                try:
                    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
                    reset_lora()
                    if method != 'B2':
                        models = matched_models()
                        connector = models[method].to(device)
                        del models
                    trainable = dict(lora)
                    if connector is not None:
                        trainable.update({'connector.' + k: p for k, p in connector.named_parameters()})
                    parameter_count = {kind: sum(p.numel() for k, p in trainable.items() if k.startswith('connector.') == (kind == 'connector'))
                                       for kind in ('connector', 'language_lora')}
                    run['trainable_parameters'] = parameter_count
                    initial_model = copy.deepcopy(connector.state_dict()) if connector is not None else None
                    initial_snapshot = parameter_snapshot(trainable)

                    def optimizer():
                        groups = [{'params': list(lora.values()), 'lr': config['language_lora_learning_rate']}]
                        if connector is not None:
                            groups.append({'params': list(connector.parameters()), 'lr': config['connector_learning_rate']})
                        return torch.optim.AdamW(groups, weight_decay=config['weight_decay'])

                    opt = optimizer(); smoke_row = next(r for r in train if r['supervision']['family'] == 'declared_pick')
                    smoke = []; smoke_details = []
                    for step in range(config['smoke_steps']):
                        if connector is not None:
                            connector.train()
                        terms, output, _ = forward(smoke_row, method, connector)
                        smoke.append(float(terms['total'].detach()))
                        detail = {'step': step+1, 'loss': {k: float(v.detach()) for k, v in terms.items() if isinstance(v, torch.Tensor)}}
                        if output is not None:
                            detail['prefix_RMS'] = float(output['prefix'].detach().square().mean().sqrt())
                            detail['prefix_max_abs'] = float(output['prefix'].detach().abs().max())
                        gradient = backward_and_step(terms['total'], opt, trainable.values(), max_grad_norm=config['max_grad_norm'])
                        detail.update(gradient); smoke_details.append(detail)
                        if not gradient['stepped']:
                            raise ValueError('No real optimizer update')
                        del terms, output
                    run['single_batch_smoke'] = {'losses': smoke, 'details': smoke_details, 'loss_decreased': smoke[-1] < smoke[0],
                        'parameter_audit': parameter_change_audit(trainable, initial_snapshot, require_trainable_change=True)}
                    changed = run['single_batch_smoke']['parameter_audit']['trainable_changed']
                    assert any(not k.startswith('connector.') for k in changed), 'Language LoRA did not update'
                    if connector is not None:
                        assert any(k.startswith('connector.') for k in changed), 'Connector did not update'
                    if not run['single_batch_smoke']['loss_decreased']:
                        raise ValueError('Single-batch smoke loss did not decrease')
                    # Do not leak smoke overfitting into development comparisons.
                    reset_lora()
                    if connector is not None:
                        connector.load_state_dict(initial_model)
                    opt = optimizer()
                    run['initial_validation'] = evaluate(val, method, connector)
                    event('method_smoke_pass', method=method, seed=seed, first_loss=smoke[0], last_loss=smoke[-1], parameters=parameter_count)
                    torch.cuda.reset_peak_memory_stats()
                    run_snapshot = parameter_snapshot(trainable)
                    best = float('inf'); best_step = None
                    for step, index in enumerate(schedule, 1):
                        if connector is not None:
                            connector.train()
                        terms, output, length = forward(train[index], method, connector)
                        log = {'step': step, 'sample_id': train[index]['public']['sample_id'], 'family': train[index]['supervision']['family'],
                            'loss': {k: float(v.detach()) for k, v in terms.items() if isinstance(v, torch.Tensor)},
                            'supervised_counts': terms['supervised_counts'], 'text_tokens': length}
                        log.update(backward_and_step(terms['total'], opt, trainable.values(), max_grad_norm=config['max_grad_norm']))
                        if not log['stepped']:
                            raise ValueError('Training step had zero gradient')
                        run['optimizer_steps'] = step; run['history'].append(log)
                        with (folder / 'training.jsonl').open('a') as f:
                            f.write(json.dumps(log) + '\n')
                        del terms, output
                        if step % 20 == 0:
                            event('training', method=method, seed=seed, step=step,
                                mean_last20_loss=sum(v['loss']['total'] for v in run['history'][-20:]) / min(20, step))
                        if step in config['checkpoint_steps']:
                            measured = evaluate(val, method, connector)
                            path = folder / f'step{step}'
                            path.mkdir()
                            save_file({k: p.detach().cpu().contiguous() for k, p in adapter_state(adapters).items()}, str(path / 'language_lora.safetensors'))
                            if connector is not None:
                                save_file({k: p.detach().cpu().contiguous() for k, p in connector.state_dict().items()}, str(path / 'connector.safetensors'))
                            torch.save({'optimizer': opt.state_dict(), 'torch_rng': torch.get_rng_state(), 'schedule_sha256': schedule_digest,
                                        'completed_steps': step, 'config': config}, path / 'optimizer.pt')
                            score = measured['language_CE']['mean']
                            candidate = {'step': step, 'validation': measured, 'path': str(path),
                                'files': {p.name: sha(p) for p in path.iterdir() if p.is_file()}}
                            run['checkpoints'].append(candidate)
                            dump(path / 'validation.json', measured)
                            if score < best:
                                best, best_step = score, step
                            event('validation', method=method, seed=seed, step=step, group_language_CE=score)
                    run['parameter_audit'] = parameter_change_audit(trainable, run_snapshot, require_trainable_change=True)
                    changed = run['parameter_audit']['trainable_changed']
                    assert any(not k.startswith('connector.') for k in changed), 'Language LoRA did not update'
                    if connector is not None:
                        assert any(k.startswith('connector.') for k in changed), 'Connector did not update'
                    best_dir = folder / f'step{best_step}'
                    selected_lora = load_file(str(best_dir / 'language_lora.safetensors'))
                    with torch.no_grad():
                        for name, p in lora.items():
                            p.copy_(selected_lora[name].to(device))
                    assert all(torch.equal(p.detach().cpu(), selected_lora[name]) for name, p in lora.items())
                    bridge.adapter_sha256 = sha(best_dir / 'language_lora.safetensors')
                    if connector is not None:
                        connector.load_state_dict(load_file(str(best_dir / 'connector.safetensors')))
                    run['selected_step'] = best_step
                    run['selected_validation'] = evaluate(val, method, connector, decode=True)
                    # Token on/off diagnostic: development-only, not action success.
                    if connector is not None:
                        subset = [r for r in val if r['supervision']['family'] == 'declared_pick'][:4]
                        probes = []
                        with torch.no_grad():
                            for row in subset:
                                native, target, x, metadata, _ = inputs(row, method)
                                soft = connector(x)['prefix']
                                on = bridge.teacher_forcing(pack(native, soft), target, torch.ones_like(target, dtype=torch.bool), eos_id=sp.eos_id())
                                off = bridge.teacher_forcing(pack(native, None), target, torch.ones_like(target, dtype=torch.bool), eos_id=sp.eos_id())
                                probes.append({'sample_id': row['public']['sample_id'], 'on_CE': float(on['loss']), 'off_CE': float(off['loss']),
                                    'max_abs_target_logit_delta': float((on['logits'] - off['logits']).abs().max())})
                        run['token_on_off_development_probe'] = probes
                    run.update(status='DEVELOPMENT_TRAINING_COMPLETE', elapsed_s=time.monotonic()-run_start,
                        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(), selected_checkpoint=str(best_dir),
                        model_trained=True, general_hidden_intent_robustness_proven=False, full_action_loop=False)
                    dump(folder / 'result.json', run)
                    event('method_complete', method=method, seed=seed, selected_step=best_step,
                        validation_exact=run['selected_validation']['exact'], validation_count=len(val), elapsed_s=run['elapsed_s'])
                except Exception as exc:
                    run.update(status='FAIL', error=str(exc), traceback=traceback.format_exc(), elapsed_s=time.monotonic()-run_start)
                    dump(folder / 'failure.json', run)
                    event('method_failed', method=method, seed=seed, error=str(exc))
                finally:
                    bridge.reset_generation()
                    for p in base.parameters():
                        p.grad = None
                    connector = None
                    if 'models' in locals():
                        del models
                    if 'opt' in locals():
                        del opt
                    native_cache.clear(); image_cache.clear()
                    import gc
                    gc.collect(); torch.cuda.empty_cache()
        unchanged = all(p._version == frozen_versions[n] and p.grad is None and not p.requires_grad
                        for n, p in base.named_parameters() if id(p) not in lora_ids)
        assert unchanged, 'Frozen backbone changed'
        assert expert_shapes == {n: list(p.shape) for n, p in base.paligemma_with_expert.gemma_expert.named_parameters()}
        assert all(not a.enabled for a in adapters.values()), 'Language adapter leaked outside scope'
        report.update(status='DEVELOPMENT_PILOT_COMPLETE' if all(r['status']=='DEVELOPMENT_TRAINING_COMPLETE' for r in report['runs']) else 'PILOT_WITH_FAILURES',
            frozen_base_unchanged=True, action_expert_unchanged=True, elapsed_s=time.monotonic()-start,
            full_step08_complete=False, seeds_1_2_not_run=any(not any(r['seed']==s and r['status']=='DEVELOPMENT_TRAINING_COMPLETE' for r in report['runs']) for s in (1,2)), place_phase_ready=False)
    except Exception as exc:
        report.update(status='FAIL', error=str(exc), traceback=traceback.format_exc(), elapsed_s=time.monotonic()-start)
    dump(args.output / 'training_result.json', report)
    print('TRAINING_RESULT=' + json.dumps({k: report.get(k) for k in ('status','error','elapsed_s')}), flush=True)
    if report['status'] != 'DEVELOPMENT_PILOT_COMPLETE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
