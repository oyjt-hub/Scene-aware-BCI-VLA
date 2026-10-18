"""Versioned public-alias binding at the VLM integration boundary.

The connector still reads no aliases/IDs and emits four intent plus four
diagnostic tokens. This parameter-free head uses *predicted* candidate weights
to bind the intent stream to the current public dictionary, then packs four
unchanged diagnostic tokens followed by four bound intent tokens. No label or
target text enters either API.
"""
from collections.abc import Mapping
import math

import torch
from torch import nn

from bci_vla.data_v4.contracts import validate_public
from bci_vla.interaction_contract_v4.validation import validate_document

BINDING_VERSION = 'public-object-reference/0.1'
PREFIX_ORDER = 'diagnostic[4],bound_intent[4]'


def _embedding_device(embed_tokens):
    owner = embed_tokens if isinstance(embed_tokens, nn.Module) else getattr(embed_tokens, '__self__', None)
    if isinstance(owner, nn.Module):
        parameter = next(owner.parameters(), None)
        if parameter is not None:
            return parameter.device
    return torch.device('cpu')


def _token_ids(tokenizer, alias):
    ids = tokenizer(alias)
    if isinstance(ids, torch.Tensor):
        if ids.ndim != 1 or ids.dtype != torch.long:
            raise ValueError('alias tokenizer must return unbatched integer IDs without BOS/EOS')
        ids = ids.tolist()
    if (not isinstance(ids, (list, tuple)) or not 1 <= len(ids) <= 4
            or any(type(token) is not int or token < 0 for token in ids)):
        raise ValueError('alias must encode to 1..4 nonnegative token IDs; no truncation or BOS/EOS')
    return list(ids)


def build_reference_bank(publics, tokenizer_callable, embed_tokens, *, hidden_size=2048,
                         device=None, dtype=None, max_objects=16):
    """Frozen [B,N,4,D] alias embeddings, right aligned with zero left padding.

    `tokenizer_callable(alias)` must use the runtime tokenizer WITHOUT BOS/EOS.
    `embed_tokens(ids)` receives [1,T] IDs and must return raw [1,T,D] embeddings
    from the same VLM, before native sqrt(D) scaling. Scaling is applied exactly
    once here. No-grad/detach freezes the bank without disabling gradients in
    the later predicted mixture. N defaults to the connector's padded 16 rows.
    GPU callers should pass the runtime device explicitly.
    """
    if (not isinstance(publics, (list, tuple)) or not publics
            or type(hidden_size) is not int or hidden_size < 1
            or type(max_objects) is not int or not 0 <= max_objects <= 16
            or not callable(tokenizer_callable) or not callable(embed_tokens)):
        raise ValueError('nonempty public batch, callables and valid bank dimensions required')
    if dtype is not None and not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError('reference bank requires floating dtype')
    device = torch.device(device) if device is not None else _embedding_device(embed_tokens)
    aliases, sequences = [], {}
    for public in publics:
        if public.get('schema_version') == 'bci-vla-interaction/4.1':
            validate_public(public)
        else:
            validate_document(public)
        objects = public['scene']['objects']
        if len(objects) > max_objects:
            raise ValueError('candidate rows exceed reference bank width; no truncation')
        row = []
        for obj in objects:
            alias = obj['alias'] if obj['object_valid'] else None
            row.append(alias)
            if alias is not None and alias not in sequences:
                sequences[alias] = _token_ids(tokenizer_callable, alias)
        aliases.append(row)
    flat_ids = [token for sequence in sequences.values() for token in sequence]
    with torch.no_grad():
        references = {}
        if flat_ids:
            ids = torch.tensor([flat_ids], device=device, dtype=torch.long)
            raw = embed_tokens(ids)
            if (not isinstance(raw, torch.Tensor) or raw.shape != (1, len(flat_ids), hidden_size)
                    or not raw.is_floating_point() or raw.device != ids.device
                    or not torch.isfinite(raw).all()):
                raise ValueError('native alias embedding shape/device/finite contract')
            # Match native embed_prefix: raw token embedding * sqrt(hidden size).
            scaled = (raw.detach() * math.sqrt(hidden_size)).to(dtype=dtype or raw.dtype)
            if not torch.isfinite(scaled).all():
                raise ValueError('nonfinite scaled reference embeddings')
            offset = 0
            for alias, sequence in sequences.items():
                references[alias] = scaled[0, offset:offset + len(sequence)]
                offset += len(sequence)
            bank_dtype = scaled.dtype
        else:
            bank_dtype = dtype or torch.float32
        bank = torch.zeros(len(publics), max_objects, 4, hidden_size, device=device, dtype=bank_dtype)
        for batch, row in enumerate(aliases):
            for index, alias in enumerate(row):
                if alias is not None:
                    bank[batch, index, -len(sequences[alias]):] = references[alias]
    return bank.detach()


class ObjectReferenceBinding(nn.Module):
    """Predicted soft reference mixture, never argmax or teacher-selected text.

    reference mode: refs = sum_i predicted_p_i * bank_i, without renormalizing
    away outside mass; bound = residual_scale * intent + reference_scale * refs.
    Output order is [original diagnostic 4, bound intent 4]. In reorder_only
    mode the original intent embeddings are unchanged, and bank/probabilities
    are unused. This control isolates readout order from reference information.
    """
    def __init__(self, *, mode='reference', residual_scale=.1, reference_scale=1.):
        super().__init__()
        if mode not in ('reference', 'reorder_only'):
            raise ValueError('mode must be reference or reorder_only')
        if any(not isinstance(value, (float, int)) or isinstance(value, bool)
               or not math.isfinite(value) or value < 0 for value in (residual_scale, reference_scale)):
            raise ValueError('finite nonnegative binding scales required')
        self.mode = mode
        self.residual_scale = float(residual_scale)
        self.reference_scale = float(reference_scale)
        self.binding_version = BINDING_VERSION
        self.prefix_order = PREFIX_ORDER

    def forward(self, module_output, reference_bank, candidate_mask):
        if not isinstance(module_output, Mapping) or 'prefix' not in module_output:
            raise ValueError('connector output with prefix required')
        prefix = module_output['prefix']
        if (not isinstance(prefix, torch.Tensor) or prefix.ndim != 3 or prefix.shape[0] < 1
                or prefix.shape[1] != 8 or prefix.shape[2] < 1 or not prefix.is_floating_point()
                or not torch.isfinite(prefix).all()):
            raise ValueError('eight finite floating connector tokens required')
        batch, _, dimension = prefix.shape
        if (not isinstance(candidate_mask, torch.Tensor) or candidate_mask.ndim != 2
                or candidate_mask.shape[0] != batch or candidate_mask.shape[1] > 16
                or candidate_mask.dtype != torch.bool or candidate_mask.device != prefix.device):
            raise ValueError('candidate mask shape/dtype/device')
        diagnostic, intent = prefix[:, 4:], prefix[:, :4]
        if self.mode == 'reorder_only':
            return torch.cat([diagnostic, intent], 1)
        probability = module_output.get('candidate_probabilities')
        if (not isinstance(probability, torch.Tensor) or probability.shape != candidate_mask.shape
                or not probability.is_floating_point() or probability.device != prefix.device
                or not torch.isfinite(probability).all() or (probability < 0).any()
                or (probability > 1 + 1e-6).any()
                or (probability[~candidate_mask] != 0).any()
                or (probability.sum(1) > 1 + 1e-5).any()):
            raise ValueError('predicted candidate probabilities must preserve valid sub-probability mass')
        if (not isinstance(reference_bank, torch.Tensor)
                or reference_bank.shape != (batch, candidate_mask.shape[1], 4, dimension)
                or not reference_bank.is_floating_point() or reference_bank.device != prefix.device
                or not torch.isfinite(reference_bank).all()):
            raise ValueError('finite [B,N,4,D] reference bank required on prefix device')
        working_dtype = torch.promote_types(probability.dtype, reference_bank.dtype)
        if working_dtype in (torch.float16, torch.bfloat16):
            working_dtype = torch.float32
        references = torch.einsum('bn,bnkd->bkd', probability.to(working_dtype),
                                  reference_bank.detach().to(working_dtype)).to(prefix.dtype)
        bound = self.residual_scale * intent + self.reference_scale * references
        result = torch.cat([diagnostic, bound], 1)
        if not torch.isfinite(result).all():
            raise ValueError('nonfinite bound prefix')
        return result

    def extra_repr(self):
        return f'mode={self.mode}, residual_scale={self.residual_scale}, reference_scale={self.reference_scale}'
