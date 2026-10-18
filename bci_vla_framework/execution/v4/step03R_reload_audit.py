"""Independent reload/contrast audit; no optimizer or new data collection."""
import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TORCHDYNAMO_DISABLE']='1'
import json,math,time,ast,hashlib,traceback,sys,re
from pathlib import Path
JOB=Path.cwd();report={'scope':'development reload and contrast audit of restricted language pilot','training':False,'formal_test_accessed':False,'actual_after_grasp_images':False,'stages':[]}
state_precision=os.environ.get('V4_AUDIT_MODE')=='state_precision'
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
 for row in ([] if state_precision else rows):cases.append({**row,'group':'all_validation_original'})
 for family in ([] if state_precision else sorted({v['family'] for v in rows})):
  for row in [v for v in rows if v['family']==family][:8]:
   # A consistent alias permutation tests instance binding independently of row order.
   ids=sorted(set(re.findall(r'OBJ_\d+',row['prompt'])),key=lambda v:int(v.split('_')[1]));mapping={v:ids[(i+1)%len(ids)] for i,v in enumerate(ids)}
   rename=lambda text:re.sub(r'OBJ_\d+',lambda m:mapping[m[0]],text)
   cases.append({**row,'prompt':rename(row['prompt']),'answer':rename(row['answer']),'group':'joint_alias_permutation'})
   prompt=row['prompt'].replace('State the main task.','Write the requested main instruction.').replace('State the next subtask.','Give the next step.').replace('Current gripper state:','Observed gripper state:').replace('Selected object:','Chosen object:')
   cases.append({**row,'prompt':prompt,'group':'unseen_wording'})
 if state_precision:
  legacy=Path('/home/oyjt/Scene-aware-BCI-VLA-runs/step15C_full_20261016_v1/results')
  # Only calibration init0 records, never legacy holdout/test episodes.
  labels=['calibration-init0-target0','calibration-init0-target1'];events=[]
  for line in (legacy/'simulator_events.jsonl').open():
   if any('"label": "'+label+'"' in line for label in labels):events.append(json.loads(line))
  for i,label in enumerate(labels):
   steps=[e for e in events if e['label']==label and e['kind']=='actual_simulation_step'];endpoint=steps[-1]
   physical=['cream_cheese_1','wine_bottle_1'][i];assert all(e['predicates'][physical]['held'] for e in steps[-3:])
   assert endpoint['predicates'][physical]['grasp_contact'] and endpoint['predicates'][physical]['lift_m']>.03
   path=legacy/(label+'_last.png');rgb=np.array(Image.open(path).convert('RGB').resize((224,224)),copy=True);image=torch.tensor(rgb,device=device).permute(2,0,1)[None].float()/127.5-1.
   with torch.no_grad():images[label]=vlm.embed_image(image).detach().to(torch.bfloat16)
   stage('actual_held_image',label=label,image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),predicates=endpoint['predicates'],trace_step=endpoint['step'],source='existing calibration rollout endpoint, no new action sampling')
   dictionary='OBJ_0 cream cheese; OBJ_1 wine bottle; OBJ_2 black bowl; OBJ_3 plate'
   for family,goal,expected in [('actual_grasp_pick_done',f'pick up OBJ_{i}','Done.'),('actual_grasp_next_place',f'put OBJ_{i} into OBJ_2',f'Place OBJ_{i} into OBJ_2.')]:
    prompt=f'answer en Objects: {dictionary}. Mode: NEXT_SUBTASK. Confirmed task: {goal}. Current gripper state: holds OBJ_{i}. State the next subtask. Output only the answer using the OBJ aliases.\n'
    row={'episode':label,'family':family,'prompt':prompt,'answer':expected,'visual':True,'group':'actual_held_phase'};cases.append(row)
    mapping={'OBJ_0':'OBJ_1','OBJ_1':'OBJ_0','OBJ_2':'OBJ_3','OBJ_3':'OBJ_2'};rename=lambda text:re.sub(r'OBJ_\d+',lambda m:mapping[m[0]],text)
    cases.append({**row,'prompt':rename(prompt),'answer':rename(expected),'group':'actual_held_joint_alias'})
  report['actual_after_grasp_images']=True
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
 if state_precision:
  # Diagnostic FP32 math control only; this process never serves the low-level policy.
  vlm.paligemma.language_model.float();head.float();torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='prefix');source=ast.get_source_segment(code,node).replace('.to(torch.bfloat16)','.to(head.weight.dtype)');exec(source,globals())
  with high_level_scope(adapters),torch.inference_mode():
   p=prefix(sentinel);target=sp.encode(sentinel['answer'],add_eos=True);answer=vlm.embed_language_tokens(torch.tensor([target[:-1]],device=device))*math.sqrt(2048)
   whole=torch.cat([p,answer.to(head.weight.dtype)],1);ar=torch.zeros(whole.shape[:2],dtype=torch.bool,device=device);ar[:,p.shape[1]:]=True
   fullh,_=forward(whole,ar);teacher32=torch.cat([head(fullh[:,p.shape[1]-1+j:p.shape[1]+j]).float() for j in range(len(target))],1)
   changed=whole.clone();changed[:,-1,:]=0;changedh,_=forward(changed,ar);masked_diff=float((head(changedh[:,p.shape[1]-1:p.shape[1]])-teacher32[:,:1]).abs().max())
   negativeh,_=forward(changed,torch.zeros_like(ar));negative_diff=float((head(negativeh[:,p.shape[1]-1:p.shape[1]])-teacher32[:,:1]).abs().max())
   h,cache=forward(p,torch.zeros(p.shape[:2],dtype=torch.bool,device=device),use_cache=True);fp32diffs=[];fp32argmax=[]
   for i in range(len(target)):
    cached=head(h[:,-1:]).float();fp32diffs.append(float((cached-teacher32[:,i:i+1]).abs().max()));fp32argmax.append(bool(torch.equal(cached.argmax(-1),teacher32[:,i:i+1].argmax(-1))))
    if i<len(target)-1:
     emb=vlm.embed_language_tokens(torch.tensor([[target[i]]],device=device))*math.sqrt(2048);h,cache=forward(emb,torch.ones(1,1,dtype=torch.bool,device=device),cache,offset=p.shape[1]+i,use_cache=True)
  report['fp32_math_control']={'max_abs_cached_vs_teacher_logit':max(fp32diffs),'all_argmax_agree':all(fp32argmax),'masked_future_difference':masked_diff,'unmasked_negative_control_difference':negative_diff,'TF32_disabled':True,'scope':'isolated diagnostic math check only, not change to low-level precision/configuration'}
 report.update(status='AUDIT_COMPLETE',metrics=metrics,cases=results,cache_max_abs_logit_difference=max(diffs),cache_argmax_all_agree=all(argmax),off_regression_max_abs=offdiff,adapters_disabled=all(not a.enabled for a in adapters.values()),checkpoint_loaded=True,high_level_gate_passed=False,optimizer_steps=0)
except Exception as exc:report.update(status='FAILED',error=str(exc),traceback=traceback.format_exc())
(JOB/('state_precision_audit.json' if state_precision else 'reload_audit.json')).write_text(json.dumps(report,indent=2));print('AUDIT_RESULT='+json.dumps({k:v for k,v in report.items() if k not in ['cases','stages']}),flush=True)
