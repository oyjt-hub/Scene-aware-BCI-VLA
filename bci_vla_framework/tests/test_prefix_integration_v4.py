"""Local integration PREWORK only. Reference backend is not PaliGemma/checkpoint.

Actual LIBERO pixels/public features feed real Step06 connectors. A tiny random
attention reference implements the audited two-stream call signature to test our
packing/label/position/cache logic. Real Gemma kernels/tokenizer remain unverified.
"""
import ast,copy,json,math,unittest
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from bci_vla.prefix_integration_v4.adapter import Pi05HighLevelPrefix,DependencyGateError,StaleGenerationError,allowed_attention
from bci_vla.neural_interaction_v4.model import matched_models
from bci_vla.neural_interaction_v4.tensorizer import collate_public
ROOT=Path(__file__).resolve().parents[1]

class ReferenceVLM(nn.Module):
 def __init__(self,d=2048):
  super().__init__();self.paligemma=nn.Module();self.paligemma.config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=d,max_position_embeddings=512))
  self.paligemma.lm_head=nn.Linear(d,32,bias=False);self.embedding=nn.Embedding(32,d);self.image_projection=nn.Linear(3,2*d)
  self.q=nn.Linear(d,32);self.k=nn.Linear(d,32);self.v=nn.Linear(d,32);self.out=nn.Linear(32,d);self.pos=nn.Embedding(512,32);self.norm=nn.LayerNorm(d);self.calls=[]
 def embed_language_tokens(self,ids):return self.embedding(ids)
 def forward(self,attention_mask,position_ids,past_key_values,inputs_embeds,use_cache):
  self.calls.append({'mask':attention_mask.detach().clone(),'positions':position_ids.detach().clone(),'suffix':inputs_embeds[1],'use_cache':use_cache,'past_present':past_key_values is not None})
  assert inputs_embeds[1] is None
  emb=inputs_embeds[0];p=self.pos(position_ids.clamp_min(0));q=self.q(emb)+p;k=self.k(emb)+p;v=self.v(emb)
  if past_key_values is not None:k=torch.cat([past_key_values[0],k],1);v=torch.cat([past_key_values[1],v],1)
  scores=(q@k.transpose(1,2))/math.sqrt(32);weights=(scores+attention_mask[:,0]).softmax(-1)
  h=self.norm(emb+self.out(weights@v));return (h,None),(k,v) if use_cache else None

class ReferencePi05(nn.Module):
 def __init__(self):
  super().__init__();self.paligemma_with_expert=ReferenceVLM();self.action_expert=nn.Linear(4,4);self.action_calls=0
 def embed_prefix(self,images,image_masks,ids,mask):
  v=self.paligemma_with_expert;embs=[];valid=[]
  for image,im in zip(images,image_masks,strict=True):
   embs.append(v.image_projection(image.mean((-1,-2))).reshape(image.shape[0],2,2048));valid.append(im[:,None].expand(-1,2))
  embs.append(v.embed_language_tokens(ids)*math.sqrt(2048));valid.append(mask)
  emb=torch.cat(embs,1);pad=torch.cat(valid,1);return emb,pad,torch.zeros_like(pad)
 def _prepare_attention_masks_4d(self,mask):return torch.where(mask[:,None],0.,-2.3819763e38)
 def sample_actions(self,*a,**kw):self.action_calls+=1;raise AssertionError('Actions forbidden in reference integration tests')

class PrefixTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.manual_seed(71);torch.set_num_threads(2)
  cls.audit=json.loads((ROOT/'execution/v4/03_text_capability.json').read_text(encoding='utf-8'))
  cls.rows=json.loads((ROOT/'execution/v4/06_fixtures/public.json').read_text(encoding='utf-8'));cls.public=cls.rows[0]
  cls.base=ReferencePi05();cls.adapter=Pi05HighLevelPrefix(cls.base,cls.audit,engineering_only=True,checkpoint_tag='test-reference-random-not-pi05')
  path=ROOT/'data/v4'/cls.public['scene']['image_ref'];pixels=np.array(Image.open(path).convert('RGB').resize((32,32)),copy=True)
  cls.image=torch.tensor(pixels).permute(2,0,1)[None].float()/127.5-1.
  cls.ids=torch.tensor([[2,5,8,11,0,0]],dtype=torch.long);cls.valid=cls.ids.ne(0);cls.models=matched_models();cls.x,_=collate_public([cls.public])
  for m in cls.models.values():m.eval()
  with torch.no_grad():cls.soft=cls.models['Full'](cls.x)['prefix']
 def setUp(self):self.adapter.reset_generation()
 def pack(self,soft=None,enabled=True,ids=None,valid=None):
  return self.adapter.prepare([self.image],[torch.ones(1,dtype=torch.bool)],self.ids if ids is None else ids,self.valid if valid is None else valid,soft_tokens=self.soft if soft is None else soft,enabled=enabled)
 def eq(self,a,b):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
 def test_01_default_capability_gate_denies_real_use(self):
  with self.assertRaises(DependencyGateError):Pi05HighLevelPrefix(ReferencePi05(),self.audit,checkpoint_tag='unpassed-real-checkpoint')
  self.assertFalse(self.audit['external_llm_used']);self.assertFalse(self.audit['training_run'])
 def test_02_source_API_binding_and_mask_equivalence(self):
  src=(ROOT/'execution/v4/03_sources/04_pi0_pytorch.py').read_text(encoding='utf-8');tree=ast.parse(src)
  node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_att_2d_masks');scope={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),'audited_source','exec'),scope)
  v=torch.tensor([[True,True,False,True,True]]);ar=torch.tensor([[False,False,False,True,True]])
  self.assertTrue(torch.equal(allowed_attention(v,ar),scope['make_att_2d_masks'](v,ar)))
  source=(ROOT/'execution/v4/03_sources/05_gemma_pytorch.py').read_text(encoding='utf-8')
  method=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='forward')
  self.assertTrue({'inputs_embeds','attention_mask','position_ids','past_key_values','use_cache'}<=set(a.arg for a in method.args.args))
 def test_03_native_inputs_retained_and_off_has_no_extra_tokens(self):
  native,mask,ar=self.base.embed_prefix([self.image],[torch.ones(1,dtype=torch.bool)],self.ids,self.valid)
  off=self.pack(enabled=False,soft=torch.full_like(self.soft,float('nan')));on=self.pack()
  self.assertEqual(off.embeddings.shape[1],native.shape[1]);self.eq(off.embeddings,native);self.assertTrue(torch.equal(off.valid,mask));self.assertTrue(torch.equal(off.ar,ar))
  self.eq(on.embeddings[:,:native.shape[1]],native);self.eq(on.embeddings[:,-8:],self.soft)
  self.assertEqual(on.embeddings.shape[1],native.shape[1]+8)
  # Off bypasses the soft tensor entirely; it is not eight zero vectors left behind.
  baseline=(self.base.paligemma_with_expert.forward(inputs_embeds=[native,None],attention_mask=self.base._prepare_attention_masks_4d(allowed_attention(mask,ar)),position_ids=mask.long().cumsum(1)-1,past_key_values=None,use_cache=False))[0][0]
  expected=self.adapter.logits(baseline[:,mask[0].nonzero()[-1,0]])
  self.eq(self.adapter.start_generation(self.public,off),expected)
 def test_04_position_padding_and_causal_answer_masks(self):
  pack=self.pack();self.assertTrue(torch.equal(pack.positions,pack.valid.long().cumsum(1)-1))
  targets=torch.tensor([[7,9,1]]);valid=torch.ones_like(targets,dtype=torch.bool);self.adapter.teacher_forcing(pack,targets,valid,eos_id=1)
  call=self.base.paligemma_with_expert.calls[-1];mask=call['mask'][0,0];p=pack.embeddings.shape[1]
  self.assertLess(float(mask[p-1,p]),-1e20);self.assertLess(float(mask[p,p+1]),-1e20);self.assertEqual(float(mask[p+1,p]),0.)
  for j in (~pack.valid[0]).nonzero().flatten():self.assertTrue((mask[:,j]<-1e20).all())
 def test_05_teacher_forcing_and_KV_cache_logits(self):
  for mode in [False,True]:
   pack=self.pack(enabled=mode);target=torch.tensor([[7,9,1]]);y=self.adapter.teacher_forcing(pack,target,torch.ones_like(target,dtype=torch.bool),eos_id=1)
   logits=self.adapter.start_generation(self.public,pack);self.eq(logits,y['logits'][:,0])
   for j in range(2):logits=self.adapter.decode_step(self.public,pack,target[:,j:j+1]);self.eq(logits,y['logits'][:,j+1])
 def test_06_future_target_does_not_leak(self):
  pack=self.pack();a=torch.tensor([[7,9,1]]);b=torch.tensor([[7,18,1]]);valid=torch.ones_like(a,dtype=torch.bool)
  ya=self.adapter.teacher_forcing(pack,a,valid,eos_id=1);yb=self.adapter.teacher_forcing(pack,b,valid,eos_id=1)
  self.eq(ya['logits'][:,:2],yb['logits'][:,:2]);self.assertGreater(float((ya['logits'][:,2]-yb['logits'][:,2]).abs().max()),1e-5)
 def test_07_true_language_CE_gradient_into_each_connector(self):
  targets=torch.tensor([[7,9,1]]);valid=torch.ones_like(targets,dtype=torch.bool)
  before={k:p.detach().clone() for k,p in self.base.named_parameters()}
  for name,connector in self.models.items():
   connector.zero_grad(set_to_none=True);soft=connector(self.x)['prefix'];soft.retain_grad();pack=self.pack(soft)
   out=self.adapter.teacher_forcing(pack,targets,valid,eos_id=1);out['loss'].backward()
   self.assertGreater(float(soft.grad.abs().sum()),0.,name)
   for group in ['features','prefix_projection']:
    self.assertGreater(sum(float(p.grad.abs().sum()) for k,p in connector.named_parameters() if k.startswith(group) and p.grad is not None),0.,(name,group))
   self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.base.parameters()))
   self.assertTrue(all(torch.equal(before[k],p) for k,p in self.base.named_parameters()))
 def test_08_target_and_scene_changes_invalidate_cache(self):
  pack=self.pack()
  for update in ['scene','mapping','history','confirmed_task','image','soft','mode','prompt','checkpoint']:
   self.adapter.start_generation(self.public,pack);public=copy.deepcopy(self.public);other=pack;oldtag=self.adapter.checkpoint_tag
   if update=='scene':public['context']['scene_revision']+=1
   elif update=='mapping':public['context']['mapping_revision']+=1
   elif update=='history':public['as_of_s']+=.1
   elif update=='confirmed_task':public['public_state']['pending_suggestion']={'changed_goal':True}
   elif update=='image':other=copy.copy(pack);object.__setattr__(other,'embeddings',pack.embeddings+.01)
   elif update=='soft':other=self.pack(self.soft+.01)
   elif update=='mode':other=self.pack(enabled=False)
   elif update=='prompt':other=self.pack(ids=self.ids+1)
   else:self.adapter.checkpoint_tag='another-checkpoint'
   with self.assertRaises(StaleGenerationError,msg=update):self.adapter.decode_step(public,other,torch.tensor([[7]]))
   self.assertIsNone(self.adapter._cache);self.adapter.checkpoint_tag=oldtag
 def test_09_new_session_prefill_resets_old_cache(self):
  pack=self.pack();a=self.adapter.start_generation(self.public,pack);self.adapter.decode_step(self.public,pack,torch.tensor([[7]]))
  b=self.adapter.start_generation(self.public,pack);self.eq(a,b);self.assertEqual(self.adapter._valid.shape[1],pack.embeddings.shape[1]);self.assertEqual(self.adapter._generated,0)
 def test_10_batched_padding_matches_individual_rows(self):
  ids=torch.cat([self.ids,torch.tensor([[2,4,0,0,0,0]])],0);valid=ids.ne(0)
  pack=self.adapter.prepare([self.image.expand(2,-1,-1,-1)],[torch.ones(2,dtype=torch.bool)],ids,valid,soft_tokens=self.soft.expand(2,-1,-1),enabled=True)
  target=torch.tensor([[7,9,1],[3,1,0]]);tv=torch.tensor([[True,True,True],[True,True,False]])
  out=self.adapter.teacher_forcing(pack,target,tv,eos_id=1)
  for i in range(2):
   p=self.pack(ids=ids[i:i+1],valid=valid[i:i+1]);single=self.adapter.teacher_forcing(p,target[i:i+1],tv[i:i+1],eos_id=1)
   self.eq(out['logits'][i,tv[i]],single['logits'][0,tv[i]])
 def test_11_language_target_binding_and_input_validation(self):
  with self.assertRaises(ValueError):self.pack(soft=self.soft[:,:7])
  with self.assertRaises(ValueError):self.pack(soft=self.soft*float('nan'))
  with self.assertRaises(ValueError):self.adapter.teacher_forcing(self.pack(),torch.tensor([[7,9]]),torch.ones(1,2,dtype=torch.bool),eos_id=1)
  with self.assertRaises(ValueError):self.adapter.teacher_forcing(self.pack(),torch.tensor([[7,0,1]]),torch.tensor([[True,False,True]]),eos_id=1)
 def test_12_no_action_call_or_backbone_mutation(self):
  self.adapter.train();self.assertFalse(self.base.training);self.assertTrue(all(not p.requires_grad for p in self.base.parameters()))
  self.assertEqual(self.base.action_calls,0);self.assertTrue(all(c['suffix'] is None for c in self.base.paligemma_with_expert.calls))
 def test_13_bounded_generation_and_finally_cache_reset(self):
  for mode in [True,False]:
   out=self.adapter.generate(self.public,self.pack(enabled=mode),eos_id=1,max_new_tokens=3)
   self.assertLessEqual(len(out['token_ids']),3);self.assertFalse(out['execution_authorized']);self.assertFalse(out['checkpoint_verified']);self.assertIsNone(self.adapter._cache)
 def test_14_dtype_cast_keeps_prefix_gradient(self):
  # Device/dtype crossing is differentiable; real BF16 Gemma still needs separate acceptance.
  soft=self.soft.double().requires_grad_();pack=self.pack(soft);self.assertEqual(pack.embeddings.dtype,torch.float32)
  loss=self.adapter.teacher_forcing(pack,torch.tensor([[7,1]]),torch.ones(1,2,dtype=torch.bool),eos_id=1)['loss'];loss.backward()
  self.assertEqual(soft.grad.dtype,torch.float64);self.assertGreater(float(soft.grad.abs().sum()),0.)
 def test_15_joint_object_permutation_keeps_high_level_logits(self):
  p=copy.deepcopy(self.public);p['scene']['objects'].reverse();x,_=collate_public([p]);connector=self.models['Full']
  with torch.no_grad():soft=connector(x)['prefix']
  a=self.adapter.start_generation(self.public,self.pack());b=self.adapter.start_generation(p,self.pack(soft));self.eq(a,b)
 def test_16_stop_and_decode_failure_clear_cached_state(self):
  pack=self.pack();self.adapter.start_generation(self.public,pack)
  stop=json.loads((ROOT/'examples/v4/stop_clears_task.json').read_text(encoding='utf-8'));calls=len(self.base.paligemma_with_expert.calls)
  with self.assertRaises(DependencyGateError):self.adapter.start_generation(stop,pack)
  self.assertIsNone(self.adapter._cache);self.assertEqual(calls,len(self.base.paligemma_with_expert.calls))
  self.adapter.start_generation(self.public,pack);vlm=self.base.paligemma_with_expert;old=vlm.forward
  def fail(**kwargs):raise RuntimeError('reference simulated cache/backend failure')
  vlm.forward=fail
  try:
   with self.assertRaises(RuntimeError):self.adapter.decode_step(self.public,pack,torch.tensor([[7]]))
   self.assertIsNone(self.adapter._cache)
  finally:vlm.forward=old
 def test_17_mask_dtype_is_finite(self):
  pack=self.pack();allowed=allowed_attention(pack.valid,pack.ar)
  for dtype in [torch.float32,torch.bfloat16,torch.float16]:
   mask=self.adapter._mask(allowed,dtype);self.assertEqual(mask.dtype,dtype);self.assertTrue(torch.isfinite(mask).all());self.assertTrue((mask[~allowed[:,None]]<0).all())
 def test_18_passed_report_requires_trained_adapter_binding(self):
  audit={'high_level_main_task_gate':'PASSED','high_level_subtask_gate':'PASSED','adapter':{'sha256':'trained-language-sha','parameters':100},'base_checkpoint':{'sha256':'base-sha'}}
  with self.assertRaises(DependencyGateError):Pi05HighLevelPrefix(ReferencePi05(),audit,checkpoint_tag='base-sha')

if __name__=='__main__':unittest.main(verbosity=2)
