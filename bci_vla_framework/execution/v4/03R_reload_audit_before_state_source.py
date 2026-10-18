"""Independent reload/contrast audit; no optimizer or new data collection."""
import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TORCHDYNAMO_DISABLE']='1'
import json,math,time,ast,hashlib,traceback,sys,re
from pathlib import Path
JOB=Path.cwd();report={'scope':'development reload and contrast audit of restricted language pilot','training':False,'formal_test_accessed':False,'actual_after_grasp_images':False,'stages':[]}
def stage(name,**kw):
 v={'name':name,**kw};report['stages'].append(v);print('AUDIT_STAGE='+json.dumps(v),flush=True)
try:
 import torch,numpy as np,sentencepiece as spm
 from PIL import Image
 from safetensors import safe_open
 from safetensors.torch import load_model,load_file
 from transformers.modeling_utils import no_init_weights
 from openpi.models.pi0_config import Pi0Config
 from openpi.models_pytorch.pi0_pytorch import PI0Pytorch,make_att_2d_masks
 from high_level_lora import install_high_level_lora,high_level_scope,adapter_state
 torch.set_num_threads(4);torch.manual_seed(13);device=torch.device('cuda:0');assert torch.cuda.mem_get_info()[0]>24*1024**3
 checkpoint='/home/oyjt/workspace/RLinf-RECAP/models/baseline_sft/RLinf-Pi05-LIBERO-SFT/model.safetensors'
 with no_init_weights():model=PI0Pytorch(Pi0Config(pi05=True,dtype='bfloat16'))
 with safe_open(checkpoint,framework='pt',device='cpu') as f:meta=f.metadata()
 assert meta['paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight']=='paligemma_with_expert.paligemma.lm_head.weight'
 model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight=model.paligemma_with_expert.paligemma.lm_head.weight
 load_model(model,checkpoint,strict=True,device='cpu');model.eval().requires_grad_(False).to(device)
 vlm=model.paligemma_with_expert;head=vlm.paligemma.lm_head;sp=spm.SentencePieceProcessor(model_file='/home/oyjt/.cache/openpi/big_vision/paligemma_tokenizer.model')
 data=json.loads((JOB/'language_data.json').read_text());sources=[s for s in data['sources'] if s['split']=='validation'];rows=data['validation'];images={}
 root=Path('/home/oyjt/Scene-aware-BCI-VLA-runs/v4_step04_libero_data_20261017_v1')
 for source in sources:
  path=root/'captures/images'/f"{source['episode']}_0.png";assert hashlib.sha256(path.read_bytes()).hexdigest()==source['image_sha256']
  rgb=np.array(Image.open(path).convert('RGB').resize((224,224)),copy=True);image=torch.tensor(rgb,device=device).permute(2,0,1)[None].float()/127.5-1.
  with torch.no_grad():images[source['episode']]=vlm.embed_image(image).detach().to(torch.bfloat16)
 # Reuse the exact frozen pilot text forward/decoder source, without executing its training body.
 code=(JOB/'pilot.py').read_text();tree=ast.parse(code)
 for node in ast.walk(tree):
  if isinstance(node,ast.FunctionDef) and node.name in ['prefix','forward','decode']:
   module=ast.Module(body=[node],type_ignores=[]);exec(compile(module,'frozen_pilot_functions','exec'),globals())
 sentinel=rows[0]
 with torch.no_grad():p=prefix(sentinel);h,_=forward(p,torch.zeros(p.shape[:2],dtype=torch.bool,device=device));off_before=head(h[:,-1:]).detach().clone()
 adapters=install_high_level_lora(model,rank=16,alpha=32);weights=load_file(str(JOB/'high_level_lora_step240.safetensors'),device='cpu')
 assert set(weights)==set(adapter_state(adapters))
 with torch.no_grad():
  for name,a in adapters.items():
   for side in ['A','B']:getattr(a,side).copy_(weights[name+'.'+side].to(device))
 model.requires_grad_(False)
 stage('adaptation_reloaded',parameters=sum(v.numel() for v in weights.values()),weights_sha256=hashlib.sha256((JOB/'high_level_lora_step240.safetensors').read_bytes()).hexdigest())
 cases=[]
 for row in rows:cases.append({**row,'group':'all_validation_original'})
 for family in sorted({v['family'] for v in rows}):
  for row in [v for v in rows if v['family']==family][:8]:
   # A consistent alias permutation tests instance binding independently of row order.
   ids=sorted(set(re.findall(r'OBJ_\d+',row['prompt'])),key=lambda v:int(v.split('_')[1]));mapping={v:ids[(i+1)%len(ids)] for i,v in enumerate(ids)}
   rename=lambda text:re.sub(r'OBJ_\d+',lambda m:mapping[m[0]],text)
   cases.append({**row,'prompt':rename(row['prompt']),'answer':rename(row['answer']),'group':'joint_alias_permutation'})
   prompt=row['prompt'].replace('State the main task.','Write the requested main instruction.').replace('State the next subtask.','Give the next step.').replace('Current gripper state:','Observed gripper state:').replace('Selected object:','Chosen object:')
   cases.append({**row,'prompt':prompt,'group':'unseen_wording'})
 results=[]
 with high_level_scope(adapters),torch.inference_mode():
  for i,row in enumerate(cases):
   generated=decode(row);results.append({'group':row['group'],'family':row['family'],'episode':row['episode'],'expected':row['answer'],'generated':generated,'exact':generated==row['answer'],'visual':row['visual']})
   if (i+1)%40==0:stage('evaluation_progress',completed=i+1,total=len(cases))
  # Teacher forcing / cached decoding comparison, only native prefix (no interaction tokens).
  p=prefix(sentinel);target=sp.encode(sentinel['answer'],add_eos=True);answer=vlm.embed_language_tokens(torch.tensor([target[:-1]],device=device))*math.sqrt(2048)
  whole=torch.cat([p,answer.to(torch.bfloat16)],1);ar=torch.zeros(whole.shape[:2],dtype=torch.bool,device=device);ar[:,p.shape[1]:]=True
  h,_=forward(whole,ar);teacher=head(h[:,p.shape[1]-1:]).float();h,cache=forward(p,torch.zeros(p.shape[:2],dtype=torch.bool,device=device),use_cache=True)
  diffs=[];argmax=[]
  for i in range(len(target)):
   cached=head(h[:,-1:]).float();diffs.append(float((cached-teacher[:,i:i+1]).abs().max()));argmax.append(bool(torch.equal(cached.argmax(-1),teacher[:,i:i+1].argmax(-1))))
   if i<len(target)-1:
    emb=vlm.embed_language_tokens(torch.tensor([[target[i]]],device=device))*math.sqrt(2048);h,cache=forward(emb,torch.ones(1,1,dtype=torch.bool,device=device),cache,offset=p.shape[1]+i,use_cache=True)
 with torch.no_grad():p=prefix(sentinel);h,_=forward(p,torch.zeros(p.shape[:2],dtype=torch.bool,device=device));off_after=head(h[:,-1:]);offdiff=float((off_before.float()-off_after.float()).abs().max())
 metrics={g:{'exact':sum(v['exact'] for v in results if v['group']==g),'count':sum(v['group']==g for v in results)} for g in sorted({v['group'] for v in results})}
 report.update(status='AUDIT_COMPLETE',metrics=metrics,cases=results,cache_max_abs_logit_difference=max(diffs),cache_argmax_all_agree=all(argmax),off_regression_max_abs=offdiff,adapters_disabled=all(not a.enabled for a in adapters.values()),checkpoint_loaded=True,high_level_gate_passed=False,optimizer_steps=0)
except Exception as exc:report.update(status='FAILED',error=str(exc),traceback=traceback.format_exc())
(JOB/'reload_audit.json').write_text(json.dumps(report,indent=2));print('AUDIT_RESULT='+json.dumps({k:v for k,v in report.items() if k not in ['cases','stages']}),flush=True)
