"""Derive an isolated repair runner; preserve the old frozen step08 source."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
source=(ROOT/'execution/v4/step08_real_training.py').read_text(encoding='utf-8-sig')


def replace(old,new):
    global source
    if old not in source:
        raise ValueError('Expected source anchor missing: '+old[:100])
    source=source.replace(old,new)


replace('configs/v4/train_step08_pilot.json','configs/v4/train_step09_binding_repair.json')
replace('from bci_vla.training_v4.text_baseline import build_b2_prompt',
'''from bci_vla.training_v4.text_baseline import build_b2_prompt
        from bci_vla.prefix_integration_v4.binding import ObjectReferenceBinding, build_reference_bank
        from torch.nn import functional as F''')
replace('image_cache, native_cache = {}, {}','image_cache, native_cache, reference_cache = {}, {}, {}')
replace('def inputs(row, method):', '''def bind(output, row, method):
            if output is None:
                return None
            mode = 'reorder_only' if method.endswith('_reorder_only') else 'reference'
            head = ObjectReferenceBinding(mode=mode, residual_scale=config['binding']['residual_scale'], reference_scale=config['binding']['reference_scale'])
            x, _ = collate_public([row['public']])
            candidate_mask = x['candidate_mask'].to(device)
            refs = None
            if mode == 'reference':
                key = tuple((o['alias'],o['object_valid']) for o in row['public']['scene']['objects'])
                if key not in reference_cache:
                    reference_cache[key] = build_reference_bank([row['public']], lambda text:sp.encode(text,add_bos=False),
                        base.paligemma_with_expert.embed_language_tokens, hidden_size=2048, device=device, dtype=torch.float32)
                refs = reference_cache[key]
            return head(output, refs, candidate_mask)

        def inputs(row, method):''')
replace("ce = bridge.teacher_forcing(pack(native, output['prefix'] if output is not None else None),\n                target, torch.ones_like(target, dtype=torch.bool), eos_id=sp.eos_id())['loss']", '''language = bridge.teacher_forcing(pack(native, bind(output, row, method)),
                target, torch.ones_like(target, dtype=torch.bool), eos_id=sp.eos_id())
            ce = language['loss']
            token_losses = F.cross_entropy(language['logits'][0], target[0], reduction='none')
            loss_weights = torch.ones_like(token_losses)
            import re
            for alias in re.findall(r'OBJ_\\d+', row['supervision']['language']['target']):
                alias_ids = sp.encode(alias, add_bos=False)
                ids_list = target[0].tolist()
                locations = [i for i in range(len(ids_list)-len(alias_ids)+1) if ids_list[i:i+len(alias_ids)]==alias_ids]
                if not locations:
                    raise ValueError('Alias span did not align with real tokenizer')
                for i in locations:
                    loss_weights[i:i+len(alias_ids)] = config['alias_loss_weight']
            weighted_ce = (token_losses*loss_weights).sum()/loss_weights.sum()''')
replace('language_loss=ce,','language_loss=weighted_ce,')
replace('return terms, output, text_length',"terms['raw_language_CE'] = ce.detach()\n            return terms, output, text_length")
replace("losses.append(float(terms['language']))", "losses.append(float(terms['raw_language_CE']))")
replace("metrics[family].append(float(terms['language']))", "metrics[family].append(float(terms['raw_language_CE']))")
replace("pack(native, None if output is None else output['prefix'])", "pack(native, bind(output, row, method))")
replace("'language_CE': float(terms['language'])", "'language_CE': float(terms['raw_language_CE'])")
replace("models[method].to(device)", "models['Full' if method.endswith('_reorder_only') else method].to(device)")
replace('initial_model = copy.deepcopy(connector.state_dict()) if connector is not None else None', '''previous_report = json.loads((Path(config['previous_job'])/'results/training_result.json').read_text())
                    original_method = 'Full' if method.endswith('_reorder_only') else method
                    previous_run = next(r for r in previous_report['runs'] if r['method']==original_method and r['seed']==seed)
                    previous_dir = Path(previous_run['selected_checkpoint'])
                    previous_hashes = next(r['files'] for r in previous_run['checkpoints'] if r['step']==previous_run['selected_step'])
                    assert sha(previous_dir/'language_lora.safetensors')==previous_hashes['language_lora.safetensors']
                    initial = load_file(str(previous_dir/'language_lora.safetensors'))
                    assert set(initial)==set(lora)
                    if connector is not None:
                        assert sha(previous_dir/'connector.safetensors')==previous_hashes['connector.safetensors']
                        connector.load_state_dict(load_file(str(previous_dir/'connector.safetensors')))
                    reset_lora()
                    run['initialization'] = {'path':str(previous_dir),'hashes':previous_hashes,'original_method':original_method}
                    run['binding'] = dict(config['binding'], mode='reorder_only' if method.endswith('_reorder_only') else 'reference')
                    initial_model = copy.deepcopy(connector.state_dict()) if connector is not None else None''')
replace("soft = connector(x)['prefix']", "soft = bind(connector(x), row, method)")
replace('native_cache.clear(); image_cache.clear()', 'native_cache.clear(); image_cache.clear(); reference_cache.clear()')
replace("'scope': config['scope']", "'scope': config['scope'], 'step_id': 'V4-09-binding-remediation'")
target=ROOT/'execution/v4/step09_binding_training.py'
if target.exists():
    raise FileExistsError('Preserve generated repair runner; inspect before replacing')
target.write_text(source,encoding='utf-8')
print(str(target))

# Subsequent corrective patch: see step09_binding_training.py for full-answer proto offset weighting, actual continuation SHA, and matched-residual control. Frozen attempt1 code remains in 09_binding_code_bundle1.zip.
