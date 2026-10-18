"""Reusable Step08 loss/optimizer mechanics, not a LIBERO performance evaluator.

Forward only receives tensorized public inputs. Private/sidecar targets enter at
this loss boundary. Calibration and test partitions are deliberately rejected.
"""
from collections import defaultdict, deque
import math
import random

import torch
from torch.nn import functional as F

from bci_vla.data_v4.contracts import validate_private
from bci_vla.interaction_contract_v4.validation import canonical_digest
from bci_vla.neural_interaction_v4.losses import auxiliary_losses


LOSS_NAMES = ('language', 'intent', 'channel', 'interaction', 'missing_slots', 'abstain')


def _supervision(row):
    supervision = row['supervision']
    if supervision['split'] not in ('train', 'validation'):
        raise ValueError('Step08 may read train/validation only; calibration/test are sealed')
    return supervision


def _check_bindings(rows, metadata):
    if not rows or len(rows) != len(metadata):
        raise ValueError('nonempty row/metadata batch binding required')
    for row, meta in zip(rows, metadata):
        p, s = row['public'], _supervision(row)
        digest = canonical_digest(p)
        # source.sample_id is frozen-source lineage and may differ after public
        # augmentation; the sidecar's own sample_id and hash bind this example.
        if (meta['input_sha256'] != digest or s['public_input_sha256'] != digest
                or s['sample_id'] != p['sample_id']):
            raise ValueError('public/sidecar/tensor binding mismatch')
        validate_private(row['private_labels'], p)
        if row['private_labels']['split'] != s['split']:
            raise ValueError('private/sidecar split mismatch')


def _masked_bce(logits, targets, mask, zero):
    """Gather before evaluating BCE: masked NaN/None targets never enter arithmetic."""
    mask = torch.as_tensor(mask, device=logits.device)
    if mask.dtype != torch.bool or tuple(mask.shape) != tuple(logits.shape):
        raise ValueError('BCE masks must be boolean and match logits')
    values = torch.as_tensor(targets, device=logits.device, dtype=logits.dtype)
    if values.shape != logits.shape:
        raise ValueError('BCE target shape')
    count = int(mask.sum())
    if not count:
        return zero, 0
    valid = values[mask]
    if not torch.isfinite(valid).all() or ((valid < 0) | (valid > 1)).any():
        raise ValueError('valid BCE targets must lie in [0,1]')
    return F.binary_cross_entropy_with_logits(logits[mask], valid), count


def compute_training_losses(output, rows, metadata, *, language_loss=None,
                            language_mask=None, auxiliary_enabled=True, weights=None):
    """Combine existing bound labels, independent sidecar masks, and language CE.

    `language_loss` is a scalar CE from an entirely valid batch, or a [B] tensor
    for independent row masking. A scalar cannot retroactively mask only some
    rows: filter before teacher_forcing, or pass per-example CE. Caller masks
    may only further suppress sidecar labels, never enable masked labels. With no active
    language rows its contribution is a differentiable zero, even if a caller
    supplies a masked NaN CE. B2 uses output=None, auxiliary_enabled=False.
    """
    _check_bindings(rows, metadata)
    if output is not None:
        if output['prefix'].shape[0] != len(rows):
            raise ValueError('output batch binding')
        zero = output['prefix'].sum() * 0.
    elif isinstance(language_loss, torch.Tensor):
        # Do not use language_loss.sum()*0: a completely masked loss may be NaN.
        zero = torch.where(torch.isfinite(language_loss), language_loss,
                           torch.zeros_like(language_loss)).sum() * 0.
    else:
        raise ValueError('a differentiable output or language loss is required')
    if auxiliary_enabled and output is None:
        raise ValueError('auxiliary heads are required; B2 must disable auxiliary loss')

    terms = {name: zero for name in LOSS_NAMES}
    counts = {name: 0 for name in LOSS_NAMES}
    if auxiliary_enabled:
        original = auxiliary_losses(output, [r['public'] for r in rows],
                                    [r['private_labels'] for r in rows], metadata)
        for name in ('intent', 'channel', 'interaction'):
            terms[name] = original[name]
            counts[name] = original['supervised_counts'][name]
        aux = [r['supervision']['auxiliary'] for r in rows]
        slot_mask = [s['missing_slots_mask'] for s in aux]
        # Masked None is legal at this boundary and replaced before tensorization.
        slots = [[v if m else 0. for v, m in zip(s['missing_slots'], s['missing_slots_mask'])]
                 for s in aux]
        terms['missing_slots'], counts['missing_slots'] = _masked_bce(
            output['missing_slot_logits'], slots, slot_mask, zero)
        abstain_mask = [s['abstain_mask'] for s in aux]
        abstain = [s['abstain'] if s['abstain_mask'] else 0. for s in aux]
        terms['abstain'], counts['abstain'] = _masked_bce(
            output['abstain_logit'], abstain, abstain_mask, zero)

    declared_mask = torch.as_tensor([r['supervision']['language']['mask'] for r in rows], device=zero.device)
    if declared_mask.dtype != torch.bool or tuple(declared_mask.shape) != (len(rows),):
        raise ValueError('sidecar language masks must be one boolean per row')
    mask = declared_mask if language_mask is None else torch.as_tensor(language_mask, device=zero.device)
    if mask.dtype != torch.bool or tuple(mask.shape) != (len(rows),):
        raise ValueError('one boolean language mask per row is required')
    if (mask & ~declared_mask).any():
        raise ValueError('caller language mask cannot enable a masked sidecar label')
    counts['language'] = int(mask.sum())
    if counts['language']:
        if not isinstance(language_loss, torch.Tensor):
            raise ValueError('active language labels require caller-computed CE')
        if language_loss.ndim == 0:
            if counts['language'] != len(rows):
                raise ValueError('partial language mask requires per-example CE or a prefiltered batch')
            value = language_loss
        elif tuple(language_loss.shape) == (len(rows),):
            value = language_loss[mask].mean()
        else:
            raise ValueError('language CE must be scalar or [B]')
        if not torch.isfinite(value).all():
            raise ValueError('nonfinite active language CE')
        terms['language'] = value
    coefficient = dict.fromkeys(LOSS_NAMES, 1.)
    if weights is not None:
        if set(weights) - set(LOSS_NAMES):
            raise ValueError('unknown loss weight')
        coefficient.update(weights)
    if any(not math.isfinite(v) or v < 0 for v in coefficient.values()):
        raise ValueError('loss weights must be finite and nonnegative')
    terms['auxiliary_total'] = sum(coefficient[k] * terms[k] for k in LOSS_NAMES if k != 'language')
    terms['total'] = terms['auxiliary_total'] + coefficient['language'] * terms['language']
    terms['supervised_counts'] = counts
    return terms


def _row_keys(row):
    s = _supervision(row)
    source = s['source']
    keys = (source['group_id'], source['base_episode_id'], s['family'])
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError('nonempty scene group / episode / family keys required')
    return keys


def episode_balanced_schedule(rows, *, split='train', seed=0, steps=None):
    """Fixed-seed row indices; each round visits every episode once.

    Families within an episode are also round-robin balanced. Short episodes
    repeat only after cycling their own rows. Call once and reuse the exact
    result for B2/B3_aux/B4_aux/Full, rather than sampling per method.
    """
    if split not in ('train', 'validation'):
        raise ValueError('train/validation only')
    rng = random.Random(seed)
    episodes = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        group, episode, family = _row_keys(row)
        if _supervision(row)['split'] == split:
            episodes[(group, episode)][family].append(index)
    if not episodes:
        raise ValueError('no examples in requested partition')
    steps = sum(len(v) for groups in episodes.values() for v in groups.values()) if steps is None else steps
    if type(steps) is not int or steps < 0:
        raise ValueError('steps must be a nonnegative integer')
    family_queues, row_queues = {}, {}
    result = []
    while len(result) < steps:
        order = sorted(episodes)
        rng.shuffle(order)
        for ep in order:
            if not family_queues.get(ep):
                families = sorted(episodes[ep])
                rng.shuffle(families)
                family_queues[ep] = deque(families)
            family = family_queues[ep].popleft()
            key = (ep, family)
            if not row_queues.get(key):
                indices = list(episodes[ep][family])
                rng.shuffle(indices)
                row_queues[key] = deque(indices)
            result.append(row_queues[key].popleft())
            if len(result) == steps:
                break
    return result


def validation_selection(rows, *, seed=0, max_examples=32):
    """Unique validation indices balanced by scene, then episode and family."""
    if type(max_examples) is not int or max_examples < 1:
        raise ValueError('positive validation limit required')
    rng = random.Random(seed)
    eligible = []
    for index, row in enumerate(rows):
        _row_keys(row)  # Also rejects any sealed partition mixed into rows.
        if _supervision(row)['split'] == 'validation':
            eligible.append(index)
    if not eligible:
        raise ValueError('no validation examples')

    def interleave(indices, level):
        if level == 3:
            rng.shuffle(indices)
            return indices
        groups = defaultdict(list)
        for i in indices:
            groups[_row_keys(rows[i])[level]].append(i)
        queues = {key: deque(interleave(groups[key], level + 1)) for key in sorted(groups)}
        result = []
        while queues:
            keys = sorted(queues)
            rng.shuffle(keys)
            for key in keys:
                result.append(queues[key].popleft())
                if not queues[key]:
                    del queues[key]
        return result
    return interleave(eligible, 0)[:max_examples]


def scene_group_mean(values, rows):
    """Equal scene-group weighting for validation selection, not row micro-mean."""
    if len(values) != len(rows) or not rows:
        raise ValueError('nonempty value/row binding required')
    groups = defaultdict(list)
    for value, row in zip(values, rows):
        if _supervision(row)['split'] != 'validation':
            raise ValueError('validation rows only')
        if not math.isfinite(float(value)):
            raise ValueError('nonfinite validation value')
        groups[_row_keys(row)[0]].append(float(value))
    means = {key: sum(items) / len(items) for key, items in sorted(groups.items())}
    return {'mean': sum(means.values()) / len(means), 'group_means': means,
            'group_counts': {key: len(items) for key, items in sorted(groups.items())}}


def _parameters(module_or_mapping):
    return dict(module_or_mapping.named_parameters()) if isinstance(module_or_mapping, torch.nn.Module) else dict(module_or_mapping)


def parameter_snapshot(module_or_mapping, *, trainable_only=False):
    """CPU clones; for a large backbone pass only audited named LoRA tensors."""
    return {name: {'value': p.detach().cpu().clone(), 'requires_grad': p.requires_grad}
            for name, p in _parameters(module_or_mapping).items()
            if not trainable_only or p.requires_grad}


def parameter_change_audit(module_or_mapping, before, *, require_trainable_change=False,
                           require_frozen_unchanged=True):
    parameters = _parameters(module_or_mapping)
    if not set(before) <= set(parameters):
        raise ValueError('parameter names changed')
    changed, trainable_changed, frozen_changed, groups = [], [], [], defaultdict(float)
    squared = max_delta = 0.
    for name, original in before.items():
        p = parameters[name]
        current, previous = p.detach().cpu(), original['value']
        if current.shape != previous.shape or p.requires_grad != original['requires_grad']:
            raise ValueError('parameter shape/trainability changed during audit')
        delta = current.double() - previous.double()
        if not torch.isfinite(delta).all():
            raise ValueError('nonfinite parameter delta')
        maximum = float(delta.abs().max()) if delta.numel() else 0.
        squared += float(delta.square().sum())
        max_delta = max(max_delta, maximum)
        groups[name.split('.')[0]] = max(groups[name.split('.')[0]], maximum)
        if maximum > 0:
            changed.append(name)
            (trainable_changed if p.requires_grad else frozen_changed).append(name)
    if require_frozen_unchanged and frozen_changed:
        raise ValueError('frozen parameters changed: ' + ', '.join(frozen_changed[:5]))
    if require_trainable_change and not trainable_changed:
        raise ValueError('optimizer did not change an audited trainable parameter')
    return {'changed': changed, 'trainable_changed': trainable_changed,
            'frozen_changed': frozen_changed, 'max_abs_delta': max_delta,
            'l2_delta': math.sqrt(squared), 'groups_max_abs_delta': dict(groups),
            'audited_parameter_tensors': len(before)}


def backward_and_step(loss, optimizer, parameters, *, max_grad_norm=1.):
    """One real update with finite checks; zero supervision cannot cause decay."""
    params = list(parameters)
    if (not params or len({id(p) for p in params}) != len(params)
            or any(not p.requires_grad for p in params)):
        raise ValueError('unique trainable optimizer parameters required')
    optimized = [p for group in optimizer.param_groups for p in group['params']]
    if len(optimized) != len(params) or {id(p) for p in optimized} != {id(p) for p in params}:
        raise ValueError('optimizer parameter scope mismatch')
    if max_grad_norm <= 0 or not math.isfinite(max_grad_norm):
        raise ValueError('finite positive gradient clip required')
    if loss.ndim != 0 or not loss.requires_grad or not torch.isfinite(loss):
        raise ValueError('finite differentiable scalar loss required')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads = [p.grad for p in params if p.grad is not None]
    if not grads or any(not torch.isfinite(g).all() for g in grads):
        optimizer.zero_grad(set_to_none=True)
        raise ValueError('missing/nonfinite gradients; no optimizer step')
    norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm, error_if_nonfinite=True)
    nonzero = bool(norm > 0)
    if nonzero:
        optimizer.step()
    return {'stepped': nonzero, 'grad_norm_before_clip': float(norm),
            'clip_max_norm': float(max_grad_norm), 'gradient_tensors': len(grads)}
