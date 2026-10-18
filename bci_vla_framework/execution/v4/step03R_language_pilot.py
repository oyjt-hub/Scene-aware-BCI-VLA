"""Bounded high-level language LoRA pilot on existing server LIBERO dev assets.

Only public object dictionaries/explicit selections/declared symbolic states form
targets. No hidden goals, future probe labels, real EEG, robot, or formal test.
After-grasp/place examples are symbolic TEXT-ONLY interventions, not fake images
or successful demonstrations. This distinction is retained in every report.
"""
import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TORCHDYNAMO_DISABLE']='1'
import json,time,math,hashlib,traceback,random,sys
from pathlib import Path
JOB=Path.cwd();report={'scope':'restricted high-level language remediation pilot','checkpoint':'/home/oyjt/workspace/RLinf-RECAP/models/baseline_sft/RLinf-Pi05-LIBERO-SFT/model.safetensors','training_run':True,'optimizer_steps':0,'base_checkpoint_modified':False,'formal_test_accessed':False,'hidden_labels_used':False,'action_sampling':False,'stages':[]}
def stage(name,**kw):
 row={'name':name,**kw};report['stages'].append(row);print('PILOT_STAGE='+json.dumps(row),flush=True)
try:
 import torch,numpy as np,sentencepiece as spm
 from PIL import Image
 from safetensors import safe_open
 from safetensors.torch import load_model,save_file
 from transformers.modeling_utils import no_init_weights
 from openpi.models.pi0_config import Pi0Config
 from openpi.models_pytorch.pi0_pytorch import PI0Pytorch,make_att_2d_masks
 from high_level_lora import install_high_level_lora,high_level_scope,adapter_state
 torch.set_num_threads(4);torch.manual_seed(13);rng=random.Random(13)
 assert torch.cuda.mem_get_info()[0]>24*1024**3,'Insufficient free memory; do not preempt other jobs'
 device=torch.device('cuda:0');started=time.monotonic()
 with no_init_weights():model=PI0Pytorch(Pi0Config(pi05=True,dtype='bfloat16'))
 with safe_open(report['checkpoint'],framework='pt',device='cpu') as f:meta=f.metadata()
 alias='paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight';source='paligemma_with_expert.paligemma.lm_head.weight';assert meta[alias]==source
 model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight=model.paligemma_with_expert.paligemma.lm_head.weight
 load_model(model,report['checkpoint'],strict=True,device='cpu');model.eval().requires_grad_(False).to(device)
 vlm=model.paligemma_with_expert;head=vlm.paligemma.lm_head;sp=spm.SentencePieceProcessor(model_file='/home/oyjt/.cache/openpi/big_vision/paligemma_tokenizer.model')
 stage('strict_checkpoint_loaded',seconds=time.monotonic()-started)
 root=Path('/home/oyjt/Scene-aware-BCI-VLA-runs/v4_step04_libero_data_20261017_v1');manifest=json.loads((root/'captures/capture_manifest.json').read_text())
 records=[];images={};source_rows=[]
 # One existing observation per episode, disjoint scene groups inherited from 04.
 for capture in manifest['episodes']:
  if capture['partition'] not in ['train','validation']:continue
  frame=capture['frames'][0];path=root/frame['image_path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==frame['image_sha256']
  key=capture['base_episode_id'];objects=[o for o in frame['objects'] if o['object_valid']]
  dictionary='; '.join(o['alias']+' '+o['label']+' center '+str([round((o['bbox_xyxy'][0]+o['bbox_xyxy'][2])/2,2),round((o['bbox_xyxy'][1]+o['bbox_xyxy'][3])/2,2)]) for o in objects)
  rgb=np.array(Image.open(path).convert('RGB').resize((224,224)),copy=True);image=torch.tensor(rgb,device=device).permute(2,0,1)[None].float()/127.5-1.
  with torch.no_grad():images[key]=vlm.embed_image(image).detach().to(torch.bfloat16)
  source_rows.append({'episode':key,'group':capture['group_id'],'split':capture['partition'],'image_sha256':frame['image_sha256']})
  for i,obj in enumerate(objects):
   source_id=obj['alias'];destination=objects[(i+1)%len(objects)]['alias']
   for family,question,answer,visual in [
    ('main',f'Mode: MAIN_TASK. Available action: pick up selected object. Selected object: {source_id}. State the main task.',f'Pick up {source_id}.',True),
    ('subtask_before',f'Mode: NEXT_SUBTASK. Confirmed task: put {source_id} into {destination}. Current gripper state: empty. State the next subtask.',f'Pick up {source_id}.',True),
    ('subtask_after_symbolic',f'Mode: NEXT_SUBTASK. Confirmed task: put {source_id} into {destination}. Current gripper state: holds {source_id}. State the next subtask.',f'Place {source_id} into {destination}.',False),
    ('subtask_pick_done_symbolic',f'Mode: NEXT_SUBTASK. Confirmed task: pick up {source_id}. Current gripper state: holds {source_id}. State the next subtask.','Done.',False),
    ('clarify',f'Mode: MAIN_TASK. Selected object: {source_id}. No desired action is specified. Ask for the missing action.','Which task do you want?',True),
   ]:
    prompt='answer en Objects: '+dictionary+'. '+question+' Output only the answer using the OBJ aliases.\n'
    records.append({'episode':key,'split':capture['partition'],'family':family,'prompt':prompt,'answer':answer,'visual':visual})
 assert not ({r['group'] for r in source_rows if r['split']=='train'} & {r['group'] for r in source_rows if r['split']=='validation'})
 train=[r for r in records if r['split']=='train'];validation=[r for r in records if r['split']=='validation']
 # Evaluation is development only; select a fixed balanced subset before training.
 eval_rows=[]
 for family in sorted({r['family'] for r in validation}):eval_rows.extend([r for r in validation if r['family']==family][::max(1,len([r for r in validation if r['family']==family])//8)][:8])
 (JOB/'language_data.json').write_text(json.dumps({'source':'existing public LIBERO development assets; explicit symbolic counterfactuals','sources':source_rows,'train':train,'validation':validation,'evaluation':eval_rows},indent=2))
 stage('data_frozen',train=len(train),validation=len(validation),evaluation=len(eval_rows),families=sorted({r['family'] for r in records}),actual_after_grasp_images=False)
 def prefix(row):
  ids=sp.encode(row['prompt'],add_bos=True);lang=vlm.embed_language_tokens(torch.tensor([ids],device=device))*math.sqrt(2048)
  parts=([images[row['episode']]] if row['visual'] else [])+[lang.to(torch.bfloat16)]
  return torch.cat(parts,1)
 def forward(emb,ar,cache=None,offset=0,use_cache=False):
  n=emb.shape[1]
  if cache is None:
   valid=torch.ones(1,n,dtype=torch.bool,device=device);mask=model._prepare_attention_masks_4d(make_att_2d_masks(valid,ar)).to(emb.dtype)
  else:mask=torch.zeros(1,1,n,offset+n,dtype=emb.dtype,device=device)
  (h,suffix),cache=vlm.forward(inputs_embeds=[emb,None],attention_mask=mask,position_ids=torch.arange(offset,offset+n,device=device)[None],past_key_values=cache,use_cache=use_cache)
  assert suffix is None;return h,cache
 def loss_on(row):
  p=prefix(row);target=sp.encode(row['answer'],add_eos=True);answer=vlm.embed_language_tokens(torch.tensor([target[:-1]],device=device))*math.sqrt(2048)
  emb=torch.cat([p,answer.to(torch.bfloat16)],1);ar=torch.zeros(emb.shape[:2],dtype=torch.bool,device=device);ar[:,p.shape[1]:]=True
  h,_=forward(emb,ar);logits=head(h[:,p.shape[1]-1:]).float()
  return torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),torch.tensor(target,device=device))
 @torch.no_grad()
 def decode(row):
  p=prefix(row);h,cache=forward(p,torch.zeros(p.shape[:2],dtype=torch.bool,device=device),use_cache=True);ids=[]
  for j in range(24):
   token=int(head(h[:,-1:]).float().argmax(-1));ids.append(token)
   if token==sp.eos_id():break
   emb=vlm.embed_language_tokens(torch.tensor([[token]],device=device))*math.sqrt(2048);h,cache=forward(emb,torch.ones(1,1,dtype=torch.bool,device=device),cache,offset=p.shape[1]+j,use_cache=True)
  return sp.decode(ids).strip()
 @torch.no_grad()
 def evaluate(label):
  rows=[]
  for row in eval_rows:
   generated=decode(row);rows.append({'episode':row['episode'],'family':row['family'],'expected':row['answer'],'generated':generated,'exact':generated==row['answer'],'visual':row['visual']})
  metrics={f:sum(v['exact'] for v in rows if v['family']==f)/sum(v['family']==f for v in rows) for f in sorted({v['family'] for v in rows})}
  result={'label':label,'metrics':metrics,'exact':sum(v['exact'] for v in rows),'count':len(rows),'cases':rows};(JOB/(label+'.json')).write_text(json.dumps(result,indent=2));stage('development_eval',label=label,metrics=metrics,exact=result['exact'],count=len(rows));return result
 baseline=evaluate('before')
 sentinel=train[0]
 with torch.no_grad():p=prefix(sentinel);h,_=forward(p,torch.zeros(p.shape[:2],device=device,dtype=torch.bool));off_before=head(h[:,-1:]).detach().clone();del p,h
 base_versions={id(p):p._version for p in model.parameters()}
 adapters=install_high_level_lora(model,rank=16,alpha=32);parameters=[p for a in adapters.values() for p in [a.A,a.B]]
 stage('lora_installed',count=len(adapters),trainable_parameters=sum(p.numel() for p in parameters),base_frozen=all(not p.requires_grad for name,p in model.named_parameters() if not name.endswith(('.A','.B'))),initially_disabled=all(not a.enabled for a in adapters.values()))
 optimizer=torch.optim.AdamW(parameters,lr=1e-3,weight_decay=0.);maximum=int(os.environ.get('V4_PILOT_STEPS','240'));assert maximum<=480
 order=list(range(len(train)));rng.shuffle(order);history=[]
 with high_level_scope(adapters):
  for step in range(1,maximum+1):
   optimizer.zero_grad(set_to_none=True);loss=loss_on(train[order[(step-1)%len(order)]])
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite language loss')
   loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();report['optimizer_steps']=step;history.append(float(loss.detach()))
   if step%20==0:stage('training',step=step,last20_mean_loss=sum(history[-20:])/20,grad_norm=float(norm),elapsed_s=time.monotonic()-started)
   if step in [120,maximum]:
    evaluation=evaluate('step'+str(step));save_file(adapter_state(adapters),str(JOB/f'high_level_lora_step{step}.safetensors'),metadata={'source_checkpoint':report['checkpoint'],'scope':'HIGH_LEVEL_ONLY; low-level requires adapters disabled','rank':'16','alpha':'32'})
    report['latest_evaluation']=evaluation
    (JOB/'progress.json').write_text(json.dumps({'optimizer_steps':step,'metrics':evaluation['metrics'],'training_loss_last20':sum(history[-20:])/20},indent=2))
 with torch.no_grad():p=prefix(sentinel);h,_=forward(p,torch.zeros(p.shape[:2],device=device,dtype=torch.bool));off_after=head(h[:,-1:]);offdiff=float((off_before.float()-off_after.float()).abs().max())
 # All original tensors must retain their version counter; wrappers only change names.
 original=[p for name,p in model.named_parameters() if not name.endswith(('.A','.B'))]
 assert all(id(p) in base_versions and p._version==base_versions[id(p)] for p in original),'Original parameter changed'
 assert offdiff==0.,'Disabled high-level adapter must recover exact original sentinel logits'
 report.update(status='PILOT_COMPLETE_NOT_GENERAL_CAPABILITY_ACCEPTANCE',off_regression_max_abs=offdiff,all_base_parameters_frozen=all(not p.requires_grad for p in original),adapters_disabled_after_scope=all(not a.enabled for a in adapters.values()),language_loss_last20=sum(history[-20:])/min(len(history),20),trainable_parameters=sum(p.numel() for p in parameters),source_episodes=len(source_rows),development_only=True,actual_after_grasp_images=False,high_level_gate_passed=False)
except Exception as exc:report.update(status='FAILED',error=str(exc),traceback=traceback.format_exc())
(JOB/'pilot_result.json').write_text(json.dumps(report,indent=2));print('PILOT_RESULT='+json.dumps({k:v for k,v in report.items() if k not in ['stages','latest_evaluation']}),flush=True)
