"""Run pinned openpi_with_subtask method bodies against our PyTorch VLM.

The upstream implementation is JAX-only. We preserve its tokenization, greedy
decode, and observation construction bodies; this small interop layer provides
the existing trained PyTorch model and eight interaction tokens. No action call.
Full upstream CE+flow joint training is not reproduced by this adapter.
"""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import logging
import string

COMMIT = '99c738a322300166499efa0b614ffa95dbb9b21e'


def source_functions(root, manifest, *, jax, jnp, np, model_module):
    """Extract exact named AST bodies; do not import a second openpi package.

    This avoids replacing installed openpi/transformers or executing upstream
    initialization/download code. Only hash-pinned requested methods execute.
    """
    root = Path(root)
    required = {'src/openpi/models/pi0.py': ('make_attn_mask', 'generate_subtask', 'build_full_observation'),
                'src/openpi/models/tokenizer.py': ('tokenize_high_level_prefix', 'tokenize_high_low_prompt', 'detokenize')}
    assert manifest['commit'] == COMMIT
    env = {'jax': jax, 'jnp': jnp, 'np': np, '_model': model_module,
           'PALIGEMMA_EOS_TOKEN': 1, 'logging': logging, 'string': string}
    functions = {}
    for name, wanted in required.items():
        source = (root/name).read_bytes()
        if hashlib.sha256(source).hexdigest() != manifest['sha256'][name]:
            raise ValueError('upstream_source_hash_mismatch')
        tree = ast.parse(source, filename=name)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                node.decorator_list = []
                # Annotation names belong to the upstream import namespace.
                # Deferred annotations preserve method statements verbatim.
                module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
                exec(compile(ast.fix_missing_locations(module), name, 'exec'), env)
                functions[node.name] = env[node.name]
        assert all(w in functions for w in wanted)
    return functions


class UpstreamSubtaskInterop:
    """Shares the same bridge/weights as main-task generation; serialized outside.

    JAX array bookkeeping stays on CPU; Torch attention and KV cache stay on GPU.
    No prompt-only subtask cache: every observation/evidence change is recomputed.
    """
    def __init__(self, bridge, tokenizer, upstream_root, manifest):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from openpi.models import model as model_module
        self.bridge = bridge; self.jax = jax; self.jnp = jnp; self.np = np
        self.model_module = model_module
        self.functions = source_functions(upstream_root, manifest, jax=jax, jnp=jnp, np=np, model_module=model_module)
        self.tokenizer = SimpleNamespace(_tokenizer=tokenizer, _max_len=200)
        self.facade = SimpleNamespace(PaliGemma=SimpleNamespace(llm=self.llm), embed_prefix=self.embed_prefix)

    def _torch(self, value, *, floating=False):
        import torch
        array = self.np.array(value, copy=True)
        tensor = torch.from_numpy(array).to(self.bridge.base.paligemma_with_expert.paligemma.lm_head.weight.device)
        return tensor.to(self.bridge.base.paligemma_with_expert.paligemma.lm_head.weight.dtype) if floating else tensor

    def _jax(self, tensor):
        import torch
        # JAX CPU does not accept Torch bfloat16 through numpy directly.
        t = tensor.detach().cpu()
        return self.jnp.asarray(t.float().numpy() if t.dtype == torch.bfloat16 else t.numpy())

    def llm(self, inputs, *, method=None, mask=None, positions=None, kv_cache=None):
        import math
        import torch
        if method == 'embed':
            return self._jax(self.bridge.base.paligemma_with_expert.embed_language_tokens(self._torch(inputs).long()) * math.sqrt(self.bridge.hidden_size))
        if method == 'decode_to_logits':
            return self._jax(self.bridge.logits(self._torch(inputs, floating=True)))
        assert method is None and inputs[1] is None
        embeddings = self._torch(inputs[0], floating=True)
        valid = torch.ones(embeddings.shape[:2], dtype=torch.bool, device=embeddings.device)
        hidden, cache = self.bridge._forward(embeddings, valid, torch.zeros_like(valid), cache=kv_cache,
            position_ids=self._torch(positions).long(), use_cache=True, allowed=self._torch(mask).bool())
        return (self._jax(hidden), None), cache

    def embed_prefix(self, observation):
        import torch
        image = self._torch(observation.images['base_0_rgb'], floating=True).permute(0, 3, 1, 2).float()
        ids = self._torch(observation.tokenized_prompt).long()
        valid = self._torch(observation.tokenized_prompt_mask).bool()
        pack = self.bridge.prepare([image], [self._torch(observation.image_masks['base_0_rgb']).bool()], ids, valid,
                                  soft_tokens=self.soft_tokens, enabled=True)
        return tuple(self._jax(t) for t in (pack.embeddings, pack.valid, pack.ar))

    def generate(self, *, task_text, rgb, soft_tokens, max_tokens=32):
        import torch
        if not 1 <= max_tokens <= 64:
            raise ValueError('bounded_generation_required')
        # Upstream truncates silently; reject overflow before calling it.
        cleaned = task_text.lower().strip().replace('_', ' ').replace('\n', ' ')
        if cleaned and cleaned[-1] in string.punctuation:
            cleaned = cleaned[:-1]
        if len(self.tokenizer._tokenizer.encode(f'Task: {cleaned}. Subtask: ', add_bos=True)) + max_tokens > 200:
            raise ValueError('subtask_context_overflow')
        ids, mask = self.functions['tokenize_high_level_prefix'](self.tokenizer, task_text)
        image = self.np.asarray(rgb, dtype=self.np.float32)[None]
        images = {'base_0_rgb': image, 'left_wrist_0_rgb': self.np.zeros_like(image), 'right_wrist_0_rgb': self.np.zeros_like(image)}
        observation = self.model_module.Observation(images=images,
            image_masks={k:self.np.asarray([k == 'base_0_rgb']) for k in images},
            state=self.np.zeros((1,32), dtype=self.np.float32),
            tokenized_prompt=ids[None], tokenized_prompt_mask=mask[None])
        self.soft_tokens = soft_tokens
        self.bridge.reset_generation()
        try:
            with torch.no_grad():
                tokens = self.functions['generate_subtask'](self.facade, observation, max_tokens=max_tokens)
                array = self.np.asarray(tokens)
                full = self.functions['build_full_observation'](self.facade, observation, tokens)
            return {'raw_text': self.functions['detokenize'](self.tokenizer, array[0]),
                    'token_ids': array[0].tolist(), 'stopped_at_eos': bool(array.shape[1] and array[0,-1] == 1),
                    'full_prompt_tokens': self.np.asarray(full.tokenized_prompt)[0].tolist(),
                    'full_prompt_mask': self.np.asarray(full.tokenized_prompt_mask)[0].tolist(),
                    'upstream_commit': COMMIT, 'implementation':'upstream_exact_method_bodies_with_torch_interop',
                    'action_sampling':False, 'execution_authorized':False}
        finally:
            self.soft_tokens = None
            self.bridge.reset_generation()
