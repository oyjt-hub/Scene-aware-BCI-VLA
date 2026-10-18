"""Actual frozen Pi05 + trained high-level LoRA + untrained interaction prefixes.

Engineering acceptance only. Uses already-present public LIBERO fixture/captures;
no local labels uploaded, optimizer, new simulation dynamics, or action dispatch.
"""
import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TORCHDYNAMO_DISABLE']='1'
import sys,json,hashlib,time,math,copy,traceback,re
from pathlib import Path
JOB=Path.cwd();sys.path.insert(0,str(JOB));report={'scope':'V4-07 actual high-level token integration engineering only','training':False,'optimizer_steps':0,'action_sampling':False,'new_dynamics':False,'formal_test_accessed':False,'checks':[]}
def check(name,condition,**data):
 row={'name':name,'passed':bool(condition),**data};report['checks'].append(row);print('CHECK='+json.dumps(row),flush=True)
 if not condition:raise AssertionError(name)
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while chunk:=f.read(8*1024**2):h.update(chunk)
 return h.hexdigest()
try:
 import torch,numpy as np,sentencepiece as spm
 from PIL import Image
 from safetensors import safe_open
 from safetensors.torch import load_model,load_file
 from transformers.modeling_utils import no_init_weights
 from openpi.models.pi0_config import Pi0Config
 from openpi.models_pytorch.pi0_pytorch import PI0Pytorch,make_att_2d_masks
 from bci_vla.high_level_v4.lora import install_high_level_lora,high_level_scope
 from bci_vla.neural_interaction_v4.model import matched_models
 from bci_vla.neural_interaction_v4.tensorizer import collate_public
 from bci_vla.prefix_integration_v4.adapter import Pi05HighLevelPrefix,StaleGenerationError
 torch.set_num_threads(4);torch.manual_seed(71);device=torch.device('cuda:0')
 check('available_GPU_budget',torch.cuda.mem_get_info()[0]>24*1024**3)
 upload=json.loads((JOB/'code_manifest.json').read_text())
 check('code_hashes',all(digest(JOB/r['path'])==r['sha256'] for r in upload['files']),files=len(upload['files']))
 language=Path('/home/oyjt/Scene-aware-BCI-VLA-runs/v4_step03R_language_20261017_v1')
 evidence=json.loads((language/'reload_audit.json').read_text());states=json.loads((language/'state_precision_audit.json').read_text())
 check('V4_03R_restricted_dependency',all(v['exact']==v['count'] for v in list(evidence['metrics'].values())+list(states['metrics'].values())) and states['fp32_math_control']['max_abs_cached_vs_teacher_logit']<1e-4,scope='canonical public-state language protocol, not general planning')
 checkpoint='/home/oyjt/workspace/RLinf-RECAP/models/baseline_sft/RLinf-Pi05-LIBERO-SFT/model.safetensors';base_sha='4616e4c1966a5fb063b442ea5d69857f45e873222d839b794634d87d252ab9e7';adapter_sha='c1dbc8bd23e8e2590ac2670839a068d82423aa0c4620e734d2d99618afaec380'
 check('base_and_language_adapter_hash',digest(checkpoint)==base_sha and digest(language/'high_level_lora_step240.safetensors')==adapter_sha)
 with no_init_weights():base=PI0Pytorch(Pi0Config(pi05=True,dtype='bfloat16'))
 with safe_open(checkpoint,framework='pt',device='cpu') as f:meta=f.metadata()
 alias='paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight';source='paligemma_with_expert.paligemma.lm_head.weight';assert meta[alias]==source
 base.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight=base.paligemma_with_expert.paligemma.lm_head.weight
 load_model(base,checkpoint,strict=True,device='cpu');base.eval().requires_grad_(False).to(device)
 adapters=install_high_level_lora(base,rank=16,alpha=32);weights=load_file(str(language/'high_level_lora_step240.safetensors'))
 with torch.no_grad():
  for name,a in adapters.items():
   for side in ['A','B']:getattr(a,side).copy_(weights[name+'.'+side].to(device))
 cap={'high_level_main_task_gate':'PASSED','high_level_subtask_gate':'PASSED','adapter':{'sha256':adapter_sha,'parameters':3022848},'base_checkpoint':{'sha256':base_sha}}
 bridge=Pi05HighLevelPrefix(base,cap,checkpoint_tag=base_sha,high_level_adapters=adapters,adapter_sha256=adapter_sha)
 check('native_hidden_dimension',bridge.hidden_size==2048,dimension=bridge.hidden_size)
 original_versions={id(p):p._version for p in base.parameters()};expert_shapes={k:list(v.shape) for k,v in base.paligemma_with_expert.gemma_expert.named_parameters()};sampler=base.sample_actions
 public_path=Path('/home/oyjt/Scene-aware-BCI-VLA-runs/v4_step06_cpu_20261017_v2/execution/v4/06_fixtures/public.json')
 publics=json.loads(public_path.read_text());public=next(p for p in publics if p['public_state']['diagnostics']['status']=='SUFFICIENT' and not p['public_state']['pending_question'])
 check('existing_LIBERO_public_input',public['provenance']['bci']=='simulated_channel' and public['provenance']['vision']=='privileged_simulator',source=str(public_path),public_only=True,raw_ROI_dimension=public['scene']['roi_dim'],semantic_valid=False)
 capture_root=Path('/home/oyjt/Scene-aware-BCI-VLA-runs/v4_step04_libero_data_20261017_v1');image_path=capture_root/public['scene']['image_ref']
 rgb=np.array(Image.open(image_path).convert('RGB').resize((224,224)),copy=True);image=torch.tensor(rgb,device=device).permute(2,0,1)[None].float()/127.5-1.;image_masks=[torch.ones(1,dtype=torch.bool,device=device)]
 episode=image_path.stem.rsplit('_',1)[0];data=json.loads((language/'language_data.json').read_text());row=next(v for v in data['train'] if v['episode']==episode and v['family']=='main')
 sp=spm.SentencePieceProcessor(model_file='/home/oyjt/.cache/openpi/big_vision/paligemma_tokenizer.model');ids=torch.tensor([sp.encode(row['prompt'],add_bos=True)],device=device);valid=torch.ones_like(ids,dtype=torch.bool);target=torch.tensor([sp.encode(row['answer'],add_eos=True)],device=device);tv=torch.ones_like(target,dtype=torch.bool)
 x,metadata=collate_public([public]);x={k:v.to(device) for k,v in x.items()};models=matched_models()
 for model in models.values():model.to(device).eval()
 def pack(soft=None,enabled=False,pids=None,pvalid=None):return bridge.prepare([image],image_masks,ids if pids is None else pids,valid if pvalid is None else pvalid,soft_tokens=soft,enabled=enabled)
 def raw_native_logits():
  em,mask,ar=base.embed_prefix([image],image_masks,ids,valid);em=em.to(torch.bfloat16)
  with high_level_scope(adapters):
   (h,_),_=base.paligemma_with_expert.forward(inputs_embeds=[em,None],attention_mask=base._prepare_attention_masks_4d(make_att_2d_masks(mask,ar)).to(em.dtype),position_ids=mask.long().cumsum(1)-1,past_key_values=None,use_cache=False)
  return base.paligemma_with_expert.paligemma.lm_head(h[:,-1]).float(),em
 with torch.no_grad():
  native_logits,native_emb=raw_native_logits();off=pack(torch.full((1,8,2048),float('nan'),device=device),False);off_logits=bridge.start_generation(public,off)
  delta=float((native_logits-off_logits).abs().max())
  check('off_exact_native_regression',delta==0 and off.embeddings.shape[1]==native_emb.shape[1],max_abs_logits=delta,native_length=native_emb.shape[1],off_length=off.embeddings.shape[1])
  soft=models['Full'](x)['prefix'];on=pack(soft,True);check('fixed_boundary_and_unscaled_tokens',on.embeddings.shape[1]==native_emb.shape[1]+8 and torch.equal(on.embeddings[:,:native_emb.shape[1]],native_emb) and torch.equal(on.embeddings[:,-8:],soft.to(torch.bfloat16)),prefix_shape=list(soft.shape),dtype=str(on.embeddings.dtype))
  zero=pack(torch.zeros_like(soft),True);zero_logits=bridge.start_generation(public,zero)
  check('zero_tokens_are_not_off',float((zero_logits-off_logits).abs().max())>1e-5,max_abs_logits=float((zero_logits-off_logits).abs().max()))
  for mode in [False,True]:
   p=pack(soft,mode);y=bridge.teacher_forcing(p,target,tv,eos_id=sp.eos_id());z=bridge.start_generation(public,p);argmax=[];differences=[]
   for j in range(target.shape[1]):
    a=y['logits'][:,j];argmax.append(bool(torch.equal(a.argmax(-1),z.argmax(-1))));differences.append(float((a-z).abs().max()))
    if j<target.shape[1]-1:z=bridge.decode_step(public,p,target[:,j:j+1])
   check('teacher_KV_argmax_'+str(mode),all(argmax),max_abs_logits=max(differences),strict_BF16_numeric_equivalence=False)
  for mode in [False,True]:
   p=pack(soft,mode);generated=bridge.generate(public,p,eos_id=sp.eos_id(),max_new_tokens=24);text=sp.decode(generated['token_ids']).strip()
   check('native_language_generation_'+str(mode),text==row['answer'] and generated['stopped_at_eos'],generated=text,expected=row['answer'],scope='explicit selected-object engineering fixture; not prefix utility evidence')
 # Real CE backward through frozen VLM/high-level LoRA into every interaction architecture.
 for name,connector in models.items():
  connector.zero_grad(set_to_none=True);soft=connector(x)['prefix'];soft.retain_grad();output=bridge.teacher_forcing(pack(soft,True),target,tv,eos_id=sp.eos_id());output['loss'].backward()
  groups={}
  for key,p in connector.named_parameters():
   if p.grad is not None:
    assert torch.isfinite(p.grad).all(),(name,key);groups[key.split('.')[0]]=groups.get(key.split('.')[0],0.)+float(p.grad.abs().sum())
  check('real_language_gradient_'+name,float(soft.grad.abs().sum())>0 and groups.get('features',0)>0 and groups.get('prefix_projection',0)>0 and all(p.grad is None and not p.requires_grad for p in base.parameters()),CE_loss=float(output['loss'].detach()),prefix_gradient_L1=float(soft.grad.abs().sum()),component_gradient_L1=groups)
  del soft,output;connector.zero_grad(set_to_none=True)
 # Row permutation: semantic object dictionary stays the same, tensor rows/IDs reorder together.
 with torch.no_grad():
  original=models['Full'](x)['prefix'];other=copy.deepcopy(public);other['scene']['objects'].reverse();px,_=collate_public([other]);px={k:v.to(device) for k,v in px.items()};reordered=models['Full'](px)['prefix'];err=float((original-reordered).abs().max())
  check('object_row_equivariance_real_prefix',err<1e-5,max_abs_prefix=err)
  one=bridge.generate(public,pack(original,True),eos_id=sp.eos_id(),max_new_tokens=24);two=bridge.generate(other,pack(reordered,True),eos_id=sp.eos_id(),max_new_tokens=24)
  check('reordered_instance_generation',one['token_ids']==two['token_ids'],generated=sp.decode(two['token_ids']))
  aliases=[v['alias'] for v in public['scene']['objects']];rename={v:aliases[(i+1)%len(aliases)] for i,v in enumerate(aliases)};sub=lambda s:re.sub(r'OBJ_\d+',lambda m:rename[m[0]],s)
  aliased=copy.deepcopy(public)
  for obj in aliased['scene']['objects']:obj['alias']=rename[obj['alias']]
  ax,_=collate_public([aliased]);ax={k:v.to(device) for k,v in ax.items()};asoft=models['Full'](ax)['prefix'];aids=torch.tensor([sp.encode(sub(row['prompt']),add_bos=True)],device=device);apack=pack(asoft,True,aids,torch.ones_like(aids,dtype=torch.bool));answer=sp.decode(bridge.generate(aliased,apack,eos_id=sp.eos_id(),max_new_tokens=24)['token_ids']).strip()
  check('joint_alias_binding_real_generation',answer==sub(row['answer']),generated=answer,expected=sub(row['answer']),source='native alias dictionary and neural tensor binding jointly consistent; no claim of learned pointer utility')
  p=pack(original,True)
  for mutation in ['mapping','task','on_off','soft']:
   bridge.start_generation(public,p);changed=copy.deepcopy(public);newpack=p
   if mutation=='mapping':changed['context']['mapping_revision']+=1
   elif mutation=='task':changed['public_state']['pending_suggestion']={'new_request':True}
   elif mutation=='on_off':newpack=pack(None,False)
   else:newpack=pack(original+.1,True)
   rejected=False
   try:bridge.decode_step(changed,newpack,target[:,:1])
   except StaleGenerationError:rejected=bridge._cache is None
   check('actual_KV_invalidation_'+mutation,rejected)
  padded_ids=torch.cat([ids,torch.zeros(1,3,dtype=torch.long,device=device)],1);padded_valid=torch.cat([valid,torch.zeros(1,3,dtype=torch.bool,device=device)],1)
  pp=pack(original,True,padded_ids,padded_valid);padded=bridge.generate(public,pp,eos_id=sp.eos_id(),max_new_tokens=24)
  check('real_padding_and_positions',sp.decode(padded['token_ids']).strip()==row['answer'] and torch.equal(pp.positions,pp.valid.long().cumsum(1)-1),generated=sp.decode(padded['token_ids']))
 check('base_and_action_expert_immutable',all(id(p) in original_versions and p._version==original_versions[id(p)] and p.grad is None and not p.requires_grad for p in base.parameters()) and expert_shapes=={k:list(v.shape) for k,v in base.paligemma_with_expert.gemma_expert.named_parameters()} and base.sample_actions is sampler,action_expert_parameters=sum(p.numel() for p in base.paligemma_with_expert.gemma_expert.parameters()),action_sampling_run=False)
 check('language_adapter_scoped_off_after_calls',all(not a.enabled for a in adapters.values()))
 report.update(status='PASS_REAL_PREFIX_ENGINEERING',checks_passed=len(report['checks']),high_level_language_scope='V4-03R restricted public-state protocol only',checkpoint_loaded=True,interactions_trained=False,action_expert_structure_changed=False,concurrent_mixed_high_low_calls_verified=False,full_closed_loop=False,peak_GPU_bytes=torch.cuda.max_memory_allocated(),input_public_sha256=metadata[0]['input_sha256'],image_sha256=digest(image_path))
except Exception as exc:report.update(status='FAIL',error=str(exc),traceback=traceback.format_exc())
(JOB/'integration_result.json').write_text(json.dumps(report,indent=2));print('INTEGRATION_RESULT='+json.dumps({k:v for k,v in report.items() if k not in ['checks']}),flush=True)
