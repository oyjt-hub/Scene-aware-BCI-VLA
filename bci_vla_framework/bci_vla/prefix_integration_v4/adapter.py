"""Pi05 high-level two-stream API binding, pending real-checkpoint acceptance.

Boundary: existing native image+prompt prefix | 8 evidence vectors | causal answer.
No global monkey patch, action-expert call, optimizer, LoRA, or execution authority.
"""
from dataclasses import dataclass
import hashlib,math
import torch
from torch import nn
from torch.nn import functional as F
from bci_vla.interaction_contract_v4.validation import canonical_digest
from bci_vla.neural_interaction_v4.control import route_control
from bci_vla.high_level_v4.lora import high_level_scope

class DependencyGateError(RuntimeError):pass
class StaleGenerationError(RuntimeError):pass

def require_high_level_capability(audit):
 if audit.get('high_level_main_task_gate')!='PASSED' or audit.get('high_level_subtask_gate')!='PASSED':
  raise DependencyGateError('V4-03 high-level main-task/subtask capability not passed; engineering-only preparation cannot promote V4-07')

def allowed_attention(valid,ar):
 if valid.dtype!=torch.bool or ar.dtype!=torch.bool or valid.shape!=ar.shape or valid.ndim!=2:raise ValueError('mask shape/dtype')
 blocks=ar.cumsum(1)
 return (blocks[:,None,:]<=blocks[:,:,None])&valid[:,None,:]&valid[:,:,None]

@dataclass(frozen=True)
class PackedPrefix:
 embeddings:torch.Tensor
 valid:torch.Tensor
 ar:torch.Tensor
 enabled:bool
 native_length:int
 @property
 def positions(self):return self.valid.long().cumsum(1)-1
 @property
 def last_valid(self):
  indices=torch.arange(self.valid.shape[1],device=self.valid.device)[None,:].expand_as(self.valid)
  return indices.masked_fill(~self.valid,-1).max(1).values

def tensor_digest(tensor):
 t=tensor.detach().contiguous().cpu()
 return hashlib.sha256(str((tuple(t.shape),str(t.dtype))).encode()+t.view(torch.uint8).numpy().tobytes()).hexdigest()

class Pi05HighLevelPrefix(nn.Module):
 def __init__(self,base_model,capability_audit,*,engineering_only=False,checkpoint_tag,high_level_adapters=None,adapter_sha256=None):
  super().__init__()
  if not engineering_only:require_high_level_capability(capability_audit)
  if not checkpoint_tag:raise ValueError('explicit checkpoint binding required')
  expected=capability_audit.get('adapter')
  if expected and not engineering_only:
   if not high_level_adapters or adapter_sha256!=expected['sha256']:raise DependencyGateError('Required trained language adapter is not bound to this high-level runtime')
   if checkpoint_tag!=capability_audit['base_checkpoint']['sha256']:raise DependencyGateError('Base checkpoint hash binding mismatch')
   if sum(a.A.numel()+a.B.numel() for a in high_level_adapters.values())!=expected['parameters']:raise DependencyGateError('Language adapter size mismatch')
  self.high_level_adapters=high_level_adapters or {};self.adapter_sha256=adapter_sha256
  self.base=base_model;self.engineering_only=engineering_only;self.checkpoint_tag=checkpoint_tag
  config=base_model.paligemma_with_expert.paligemma.config.text_config
  self.hidden_size=config.hidden_size;self.context_limit=getattr(config,'max_position_embeddings',8192)
  self.base.requires_grad_(False).eval();self.reset_generation()
 def train(self,mode=True):
  super().train(mode);self.base.eval();return self
 def reset_generation(self):
  self._cache=None;self._key=None;self._valid=None;self._next_logits=None;self._generated=0
 def prepare(self,images,image_masks,prompt_ids,prompt_valid,*,soft_tokens=None,enabled=False):
  if prompt_ids.ndim!=2 or prompt_ids.dtype!=torch.long or prompt_valid.shape!=prompt_ids.shape or prompt_valid.dtype!=torch.bool:raise ValueError('prompt shape/dtype')
  if len(images)!=len(image_masks):raise ValueError('image masks')
  native,valid,ar=self.base.embed_prefix(images,image_masks,prompt_ids,prompt_valid)
  if native.ndim!=3 or native.shape[-1]!=self.hidden_size or valid.shape!=native.shape[:2] or ar.shape!=valid.shape or valid.dtype!=torch.bool or ar.dtype!=torch.bool:raise ValueError('native prefix contract')
  # This entry is prefix-LM context. Any unexpected causal native prompt fails closed.
  if ar.any():raise ValueError('expected native bidirectional image/prompt prefix')
  if not torch.isfinite(native).all():raise ValueError('nonfinite native prefix')
  native=native.to(self.base.paligemma_with_expert.paligemma.lm_head.weight.dtype)
  length=native.shape[1]
  if enabled:
   if soft_tokens is None or soft_tokens.shape!=(native.shape[0],8,self.hidden_size) or not soft_tokens.is_floating_point() or not torch.isfinite(soft_tokens).all():raise ValueError('eight finite soft tokens required')
   soft_tokens=soft_tokens.to(device=native.device,dtype=native.dtype)  # differentiable cast, no extra sqrt(D)
   native=torch.cat([native,soft_tokens],1)
   valid=torch.cat([valid,torch.ones(native.shape[0],8,dtype=torch.bool,device=valid.device)],1)
   ar=torch.cat([ar,torch.zeros(native.shape[0],8,dtype=torch.bool,device=ar.device)],1)
  if not valid.any(1).all() or native.shape[1]>self.context_limit:raise ValueError('empty/oversized context')
  return PackedPrefix(native,valid,ar,bool(enabled),length)
 def _mask(self,allowed,dtype):
  if dtype not in [torch.float32,torch.bfloat16,torch.float16]:raise ValueError('unsupported attention dtype')
  # Match original Pi05 helper; clamp only for FP16, where its sentinel overflows.
  return self.base._prepare_attention_masks_4d(allowed).clamp_min(torch.finfo(dtype).min).to(dtype)
 def _forward(self,embeddings,valid,ar,*,cache=None,position_ids=None,use_cache=False,allowed=None):
  allowed=allowed_attention(valid,ar) if allowed is None else allowed
  positions=valid.long().cumsum(1)-1 if position_ids is None else position_ids
  # The trained language branch is scoped to this high-level call. No action
  # entry is patched; callers must serialize shared-model ownership (Step11).
  with high_level_scope(self.high_level_adapters):
   (hidden,suffix),new_cache=self.base.paligemma_with_expert.forward(
    inputs_embeds=[embeddings,None],attention_mask=self._mask(allowed,embeddings.dtype),
    position_ids=positions,past_key_values=cache,use_cache=use_cache)
  if suffix is not None or hidden.shape!=embeddings.shape:raise ValueError('unexpected high-level two-stream result')
  return hidden,new_cache
 def logits(self,hidden):return self.base.paligemma_with_expert.paligemma.lm_head(hidden).float()
 def teacher_forcing(self,pack,target_ids,target_valid,*,eos_id):
  b,t=target_ids.shape
  if b!=pack.embeddings.shape[0] or t<1 or target_ids.dtype!=torch.long or target_valid.dtype!=torch.bool or target_valid.shape!=target_ids.shape or target_ids.device!=pack.embeddings.device:raise ValueError('target batch/shape/dtype/device')
  lengths=target_valid.sum(1)
  if (lengths<1).any() or not torch.equal(target_valid,torch.arange(t,device=target_valid.device)[None,:]<lengths[:,None]):raise ValueError('targets must be right-padded and nonempty')
  if not (target_ids.gather(1,(lengths-1)[:,None]).squeeze(1)==eos_id).all():raise ValueError('explicit terminal EOS required')
  if pack.embeddings.shape[1]+t-1>self.context_limit:raise ValueError('teacher forcing context overflow')
  vlm=self.base.paligemma_with_expert
  answers=vlm.embed_language_tokens(target_ids[:,:-1])*math.sqrt(self.hidden_size)
  answers=answers.to(pack.embeddings.dtype)
  emb=torch.cat([pack.embeddings,answers],1);valid=torch.cat([pack.valid,target_valid[:,1:]],1)
  ar=torch.cat([pack.ar,torch.ones_like(target_valid[:,1:])],1)
  hidden,_=self._forward(emb,valid,ar,use_cache=False)
  first=hidden[torch.arange(b,device=emb.device),pack.last_valid][:,None,:]
  predicted=torch.cat([first,hidden[:,pack.embeddings.shape[1]:]],1)
  logits=self.logits(predicted)
  loss=F.cross_entropy(logits[target_valid],target_ids[target_valid])
  return {'loss':loss,'logits':logits,'target_valid':target_valid,'prefix_length':pack.embeddings.shape[1],'execution_authorized':False,'checkpoint_verified':False}
 def _generation_key(self,public,pack):
  return (canonical_digest(public),self.checkpoint_tag,self.adapter_sha256,pack.enabled,tensor_digest(pack.embeddings),tensor_digest(pack.valid),tensor_digest(pack.ar))
 @torch.no_grad()
 def start_generation(self,public,pack):
  self.reset_generation()
  if not route_control(public)['call_neural_module']:raise DependencyGateError('STOP/CANCEL must bypass high-level generation through existing direct control path')
  key=self._generation_key(public,pack)
  hidden,cache=self._forward(pack.embeddings,pack.valid,pack.ar,use_cache=True)
  if cache is None:raise ValueError('backend did not return cache')
  self._next_logits=self.logits(hidden[torch.arange(hidden.shape[0],device=hidden.device),pack.last_valid])
  self._cache=cache;self._valid=pack.valid.clone();self._key=key
  return self._next_logits
 @torch.no_grad()
 def decode_step(self,public,pack,token_ids):
  if self._key is None or self._key!=self._generation_key(public,pack):
   self.reset_generation();raise StaleGenerationError('context, observation, goal, prompt, evidence, checkpoint or on/off changed; prefill again')
  b=self._valid.shape[0]
  if token_ids.shape!=(b,1) or token_ids.dtype!=torch.long or token_ids.device!=pack.embeddings.device:raise ValueError('decode exactly one token per live batch row')
  if self._valid.shape[1]+1>self.context_limit:self.reset_generation();raise ValueError('decode context overflow')
  token=self.base.paligemma_with_expert.embed_language_tokens(token_ids)*math.sqrt(self.hidden_size);token=token.to(pack.embeddings.dtype)
  positions=self._valid.long().sum(1)[:,None]
  valid=torch.cat([self._valid,torch.ones(b,1,dtype=torch.bool,device=token.device)],1)
  try:hidden,cache=self._forward(token,valid,torch.ones_like(valid),cache=self._cache,position_ids=positions,use_cache=True,allowed=valid[:,None,:])
  except Exception:
   self.reset_generation();raise
  self._cache=cache;self._valid=valid;self._generated+=1;self._next_logits=self.logits(hidden[:,0]);return self._next_logits
 @torch.no_grad()
 def generate(self,public,pack,*,eos_id,max_new_tokens=32):
  if pack.embeddings.shape[0]!=1 or not 1<=max_new_tokens<=128:raise ValueError('bounded single-session generation only')
  logits=self.start_generation(public,pack);tokens=[]
  try:
   for step in range(max_new_tokens):
    token=logits.argmax(-1);tokens.append(int(token[0]))
    if tokens[-1]==eos_id:break
    if step+1<max_new_tokens:logits=self.decode_step(public,pack,token[:,None])
   return {'token_ids':tokens,'stopped_at_eos':tokens[-1]==eos_id,'execution_authorized':False,'checkpoint_verified':False}
  finally:self.reset_generation()
