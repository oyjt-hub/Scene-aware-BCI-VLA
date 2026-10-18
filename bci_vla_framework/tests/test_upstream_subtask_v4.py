"""Pinned upstream method mechanics with explicit tiny-model/array stubs."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch

from bci_vla.high_level_v4.upstream_subtask import source_functions,UpstreamSubtaskInterop

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=ROOT/'third_party/openpi_with_subtask'
MANIFEST=json.loads((ROOT/'configs/v4/upstream_subtask_source.json').read_text())

class TinyTokenizer:
    def encode(self,text,add_bos=False):return ([2] if add_bos else [])+[100+ord(c) for c in text]
    def decode(self,ids):return ''.join(chr(i-100) for i in ids if i>=100)

class TinyTorchBridge:
    """Scripted logits, NOT a language capability test."""
    hidden_size=4
    def __init__(self):
        self.outputs=TinyTokenizer().encode('pick up obj 0')+[1]
        self.count=0;self.masks=[];self.positions=[];self.reset_count=0
        vlm=NS(paligemma=NS(lm_head=NS(weight=torch.zeros(256,4))),
               embed_language_tokens=lambda ids:torch.zeros(*ids.shape,4))
        self.base=NS(paligemma_with_expert=vlm)
    def reset_generation(self):self.count=0;self.reset_count+=1
    def prepare(self,images,image_masks,ids,valid,*,soft_tokens,enabled):
        assert enabled and soft_tokens.shape==(1,8,4)
        assert images[0].shape==(1,3,224,224)
        # One unused camera token before text: valid-count is NOT last index.
        mask=torch.cat([torch.tensor([[True,False]]),valid,torch.ones(1,8,dtype=torch.bool)],1)
        return NS(embeddings=torch.zeros(1,mask.shape[1],4),valid=mask,ar=torch.zeros_like(mask))
    def _forward(self,embeddings,valid,ar,*,cache,position_ids,use_cache,allowed):
        self.masks.append(allowed.clone());self.positions.append(position_ids.clone())
        hidden=torch.zeros_like(embeddings);hidden[...,0]=self.count;self.count+=1
        return hidden,NS(step=self.count)
    def logits(self,hidden):
        out=torch.full((*hidden.shape[:-1],256),-100.)
        idx=hidden[...,0].long();out.scatter_(-1,torch.tensor(self.outputs)[idx][...,None],100.)
        return out

def interop():
    obj=UpstreamSubtaskInterop.__new__(UpstreamSubtaskInterop)
    obj.bridge=TinyTorchBridge();obj.np=np;obj.jnp=np
    obj.jax=NS(tree=NS(map=lambda fn,value:value))
    obj.model_module=NS(Observation=NS,preprocess_observation=lambda rng,obs,train:obs)
    obj.functions=source_functions(UPSTREAM,MANIFEST,jax=obj.jax,jnp=np,np=np,model_module=obj.model_module)
    obj.tokenizer=NS(_tokenizer=TinyTokenizer(),_max_len=200)
    obj.facade=NS(PaliGemma=NS(llm=obj.llm),embed_prefix=obj.embed_prefix)
    return obj

class UpstreamTests(unittest.TestCase):
    def test_exact_upstream_decode_with_torch_interop_and_prompt_rebuild(self):
        obj=interop();result=obj.generate(task_text='Pick up OBJ_0.',rgb=np.zeros((224,224,3),np.float32),soft_tokens=torch.zeros(1,8,4))
        self.assertEqual(result['raw_text'],'pick up obj 0');self.assertTrue(result['stopped_at_eos'])
        self.assertEqual(result['upstream_commit'],MANIFEST['commit'])
        self.assertFalse(result['action_sampling']);self.assertFalse(result['execution_authorized'])
        valid=sum(result['full_prompt_mask']);self.assertGreater(valid,len(result['token_ids']))
        self.assertIsNone(obj.soft_tokens);self.assertEqual(obj.bridge.reset_count,2)
        # Cached decode mask includes each preceding generated token, and
        # positions count valid tokens (holes are not counted).
        self.assertEqual(obj.bridge.masks[1].shape[-1],obj.bridge.masks[0].shape[-1]+1)
        expected=int(obj.bridge.masks[0][0,-1].sum())
        self.assertEqual(int(obj.bridge.positions[1][0,0]),expected)
    def test_attention_blocks_and_holes(self):
        fn=interop().functions['make_attn_mask']
        mask=fn(np.array([[True,False,True,True]]),np.array([[False,False,True,True]]))
        self.assertFalse(mask[0,0,2]);self.assertTrue(mask[0,3,2]);self.assertFalse(mask[0,:,1].any())
    def test_context_overflow_rejected(self):
        obj=interop()
        with self.assertRaisesRegex(ValueError,'overflow'):
            obj.generate(task_text='x'*250,rgb=np.zeros((224,224,3)),soft_tokens=torch.zeros(1,8,4))
    def test_source_tamper_rejected(self):
        bad=deepcopy(MANIFEST);bad['sha256']['src/openpi/models/pi0.py']='0'*64
        with self.assertRaisesRegex(ValueError,'hash_mismatch'):
            source_functions(UPSTREAM,bad,jax=None,jnp=None,np=None,model_module=None)
    def test_bounded_non_eos_is_not_success(self):
        result=interop().generate(task_text='Pick up OBJ_0.',rgb=np.zeros((224,224,3),np.float32),soft_tokens=torch.zeros(1,8,4),max_tokens=2)
        self.assertFalse(result['stopped_at_eos'])
    def test_upstream_training_masks_separate_target(self):
        obj=interop();tokens,valid,ar,loss=obj.functions['tokenize_high_low_prompt'](obj.tokenizer,'Pick up OBJ_0.','pick up obj 0')
        self.assertTrue((ar[loss]==1).all());self.assertFalse(loss[~valid].any())
        self.assertEqual(int(tokens[np.where(loss)[0][-1]]),1)

if __name__=='__main__':unittest.main()
