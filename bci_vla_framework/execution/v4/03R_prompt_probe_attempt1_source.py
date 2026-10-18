"""Existing checkpoints + native prompts on an existing LIBERO training capture.
No training, hidden labels, project-data upload, action sampling, or new dynamics.
"""
import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TORCHDYNAMO_DISABLE']='1'
import gc,json,time,math,hashlib,traceback
from pathlib import Path
report={'scope':'V4-03 remediation diagnosis only, not gate promotion','training':False,'action_sampling':False,'formal_test_accessed':False,'stages':[]}
def stage(name,**kw):
 row={'name':name,**kw};report['stages'].append(row);print('STAGE='+json.dumps(row),flush=True)
try:
 import torch,numpy as np,sentencepiece as spm
 from PIL import Image
 from safetensors import safe_open
 from safetensors.torch import load_model
 from transformers.modeling_utils import no_init_weights
 from openpi.models.pi0_config import Pi0Config
 from openpi.models_pytorch.pi0_pytorch import PI0Pytorch,make_att_2d_masks
 torch.set_num_threads(4);torch.manual_seed(3)
 assert torch.cuda.mem_get_info()[0]>24*1024**3,'Need >=24GiB free; preserve other jobs'
 device=torch.device('cuda:0')
 root=Path('/home/oyjt/Scene-aware-BCI-VLA-runs/v4_step04_libero_data_20261017_v1')
 manifest=json.loads((root/'captures/capture_manifest.json').read_text());capture=next(v for v in manifest['episodes'] if v['partition']=='train');frame=capture['frames'][0];image_path=root/frame['image_path']
 assert hashlib.sha256(image_path.read_bytes()).hexdigest()==frame['image_sha256']
 rgb=np.array(Image.open(image_path).convert('RGB').resize((224,224)),copy=True);image=torch.tensor(rgb,device=device).permute(2,0,1)[None].float()/127.5-1.
 sp=spm.SentencePieceProcessor(model_file='/home/oyjt/.cache/openpi/big_vision/paligemma_tokenizer.model')
 stage('input_provenance',platform='existing actual LIBERO train capture',image_sha256=frame['image_sha256'],tokenizer_vocab=sp.vocab_size(),tokenizer_newline_ids=sp.encode('\n'),private_labels_used=False)
 dictionary='; '.join(f"{o['alias']} {o['label']} bbox {o['bbox_xyxy']}" for o in frame['objects']);dest=frame['objects'][-1]['alias']
 cases=[
  ('native_caption','caption en\n',None,True,False),
  ('native_question','answer en What objects are on the table?\n',None,True,False),
  ('native_main',f'answer en Objects: {dictionary}. Available task: pick up a selected object. Selected object: OBJ_0. State the main task in one sentence.\n','Pick up OBJ_0.',True,False),
  ('native_subtask',f'answer en Objects: {dictionary}. Confirmed task: put OBJ_0 into {dest}. The gripper is empty. State the next subtask.\n','Pick up OBJ_0.',True,False),
  ('text_subtask_after_grasp',f'Confirmed task: put OBJ_0 into {dest}. The gripper holds OBJ_0. Next subtask:\n',f'Place OBJ_0 into {dest}.',False,False),
  ('causal_text_main','Objects: OBJ_0 bowl, OBJ_1 cup. Allowed action: pick up. Selected object: OBJ_1. Main task:\n','Pick up OBJ_1.',False,True),
  ('fewshot_main','Selected OBJ_2. Main task: Pick up OBJ_2.\nSelected OBJ_3. Main task: Pick up OBJ_3.\nSelected OBJ_1. Main task:','Pick up OBJ_1.',False,False),
  ('causal_text_sanity','The capital of France is',None,False,True),
 ]
 for tag,checkpoint in [('base','/home/oyjt/workspace/RLinf-RECAP/models/openpi_pytorch/pi05_base/model.safetensors'),('libero_sft','/home/oyjt/workspace/RLinf-RECAP/models/baseline_sft/RLinf-Pi05-LIBERO-SFT/model.safetensors')]:
  started=time.monotonic()
  with no_init_weights():model=PI0Pytorch(Pi0Config(pi05=True,dtype='bfloat16'))
  with safe_open(checkpoint,framework='pt',device='cpu') as f:meta=f.metadata()
  alias='paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight';source='paligemma_with_expert.paligemma.lm_head.weight';assert meta[alias]==source
  model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight=model.paligemma_with_expert.paligemma.lm_head.weight
  load_model(model,checkpoint,strict=True,device='cpu');model.eval().requires_grad_(False).to(device)
  stage('checkpoint_loaded',tag=tag,path=checkpoint,strict=True,seconds=time.monotonic()-started,hidden=model.paligemma_with_expert.paligemma.config.text_config.hidden_size)
  vlm=model.paligemma_with_expert;head=vlm.paligemma.lm_head
  def forward(emb,ar,cache=None,offset=0):
   n=emb.shape[1]
   if cache is None:
    valid=torch.ones((1,n),device=device,dtype=torch.bool);mask=model._prepare_attention_masks_4d(make_att_2d_masks(valid,ar)).to(emb.dtype)
   else:mask=torch.zeros(1,1,n,offset+n,device=device,dtype=emb.dtype)
   (h,suffix),cache=vlm.forward(inputs_embeds=[emb,None],attention_mask=mask,position_ids=torch.arange(offset,offset+n,device=device)[None],past_key_values=cache,use_cache=True)
   return h,cache
  with torch.inference_mode():
   for name,prompt,expected,with_image,causal in cases:
    ids=torch.tensor([sp.encode(prompt,add_bos=True)],device=device);prefix,valid,ar=model.embed_prefix([image] if with_image else [],[torch.ones(1,dtype=torch.bool,device=device)] if with_image else [],ids,torch.ones_like(ids,dtype=torch.bool));prefix=prefix.to(torch.bfloat16)
    if causal:ar=torch.ones_like(ar)
    start=time.monotonic();h,cache=forward(prefix,ar);generated=[]
    for i in range(32):
     token=int(head(h[:,-1:]).float().argmax(-1));generated.append(token)
     if token==sp.eos_id():break
     emb=vlm.embed_language_tokens(torch.tensor([[token]],device=device))*math.sqrt(2048)
     h,cache=forward(emb,torch.ones(1,1,dtype=torch.bool,device=device),cache,offset=prefix.shape[1]+i)
    text=sp.decode(generated);stage('prompt_case',checkpoint=tag,case=name,prompt=prompt,expected_fixture=expected,generated=text,ids=generated,with_actual_image=with_image,causal_context=causal,exact_match=expected is not None and text.strip()==expected,seconds=time.monotonic()-start)
    del h,cache,prefix
  del model,vlm,head;gc.collect();torch.cuda.empty_cache()
 report['status']='COMPLETE_DIAGNOSTIC'
except Exception as exc:report.update(status='FAILED',error=str(exc),traceback=traceback.format_exc())
print('V4_03R_RESULT='+json.dumps(report),flush=True)
