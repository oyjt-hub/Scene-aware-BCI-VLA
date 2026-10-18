"""Isolated server-side development capability probe; no training/actions/services."""
import os
os.environ['JAX_PLATFORMS']='cpu'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import hashlib,json,math,sys,time,traceback
from pathlib import Path

result={'scope':'development capability smoke, synthetic images and explicit text fixtures; not held-out performance','checkpoint':'/home/oyjt/workspace/RLinf-RECAP/models/baseline_sft/RLinf-Pi05-LIBERO-SFT/model.safetensors','training':False,'action_sampling':False,'real_eeg':False,'physical_robot':False,'stages':[]}
def stage(name,**data):
    result['stages'].append(dict(name=name,**data))
    print('V4_STAGE_JSON='+json.dumps(result['stages'][-1]),flush=True)
try:
    import torch
    import sentencepiece as spm
    from safetensors.torch import load_model
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch,make_att_2d_masks
    from transformers.models.gemma import modeling_gemma
    runtime_path=Path(modeling_gemma.__file__)
    stage('runtime_source',gemma_path=str(runtime_path),gemma_sha256=hashlib.sha256(runtime_path.read_bytes()).hexdigest())
    device=torch.device('cuda:0')  # launcher restricts visibility to physical GPU4
    assert torch.cuda.mem_get_info(0)[0]>24*1024**3,'Insufficient free GPU memory'
    torch.manual_seed(0)
    torch.set_num_threads(4)
    start=time.perf_counter()
    if os.environ.get('V4_SKIP_RANDOM_INIT')=='1':
        from transformers.modeling_utils import no_init_weights
        with no_init_weights():model=PI0Pytorch(Pi0Config(pi05=True,dtype='bfloat16'))
    else:
        model=PI0Pytorch(Pi0Config(pi05=True,dtype='bfloat16'))
    stage('skeleton_constructed',seconds=time.perf_counter()-start,random_init_skipped=os.environ.get('V4_SKIP_RANDOM_INIT')=='1')
    if os.environ.get('V4_RESTORE_DECLARED_TIE')=='1':
        from safetensors import safe_open
        with safe_open(result['checkpoint'],framework='pt',device='cpu') as f:metadata=f.metadata()
        alias='paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight'
        source='paligemma_with_expert.paligemma.lm_head.weight'
        assert metadata[alias]==source,'Tie must come from checkpoint metadata'
        model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight=model.paligemma_with_expert.paligemma.lm_head.weight
        stage('checkpoint_declared_weight_tie',alias=alias,source=source,metadata_verified=True)
    if os.environ.get('V4_LOAD_TRACE')=='1':
        import faulthandler
        faulthandler.dump_traceback_later(20,repeat=True)
    load_model(model,result['checkpoint'],strict=True,device='cpu')
    if os.environ.get('V4_LOAD_TRACE')=='1':faulthandler.cancel_dump_traceback_later()
    model.eval().requires_grad_(False).to(device)
    stage('strict_checkpoint_load',seconds=time.perf_counter()-start,torch_version=torch.__version__,device_name=torch.cuda.get_device_name(),vlm_dim=model.paligemma_with_expert.paligemma.config.text_config.hidden_size)
    tokpath=Path('/home/oyjt/.cache/openpi/big_vision/paligemma_tokenizer.model')
    sp=spm.SentencePieceProcessor(model_file=str(tokpath))
    result['tokenizer']={'path':str(tokpath),'sha256':hashlib.sha256(tokpath.read_bytes()).hexdigest(),'bos':sp.bos_id(),'eos':sp.eos_id(),'pad':sp.pad_id(),'vocab_size':sp.vocab_size()}
    vlm=model.paligemma_with_expert
    def embed(ids):
        return vlm.embed_language_tokens(torch.tensor([ids],device=device))*math.sqrt(2048)
    def forward(emb,ar=None,cache=None,offset=0):
        n=emb.shape[1]
        if cache is None:
            valid=torch.ones((1,n),device=device,dtype=torch.bool)
            ar=torch.zeros_like(valid) if ar is None else ar
            mask=model._prepare_attention_masks_4d(make_att_2d_masks(valid,ar))
        else:
            mask=torch.zeros((1,1,n,offset+n),device=device,dtype=torch.float32)
        if os.environ.get('V4_MATCH_MASK_DTYPE')=='1':mask=mask.to(emb.dtype)
        pos=torch.arange(offset,offset+n,device=device)[None,:]
        (h,_),cache=vlm.forward(attention_mask=mask,position_ids=pos,past_key_values=cache,inputs_embeds=[emb,None],use_cache=True)
        return h,cache
    head=vlm.paligemma.lm_head
    cases=[
        ('main_task','Objects: OBJ_0 blue cup; OBJ_1 red cup; OBJ_2 tray. User selected OBJ_0. Propose a main task using the selected object.\nAnswer:','Pick up OBJ_0.'),
        ('next_subtask','Confirmed task: put OBJ_0 into OBJ_2. Current state: OBJ_0 is on the table and the gripper is empty. What is the next subtask?\nAnswer:','Pick up OBJ_0.'),
        ('next_after_grasp','Confirmed task: put OBJ_0 into OBJ_2. Current state: the gripper holds OBJ_0. What is the next subtask?\nAnswer:','Place OBJ_0 into OBJ_2.'),
        ('same_class_instance','Objects: OBJ_0 blue cup on the left; OBJ_1 red cup on the right. User selected OBJ_1. Which object should be picked up?\nAnswer:','OBJ_1.'),
    ]
    with torch.inference_mode():
        image=torch.zeros((1,3,224,224),device=device,dtype=torch.float32)
        image_mask=torch.ones(1,device=device,dtype=torch.bool)
        img0=vlm.embed_image(image)
        img1=vlm.embed_image(torch.ones_like(image))
        stage('synthetic_image_path',shape=list(img0.shape),finite=bool(torch.isfinite(img0).all()),black_white_max_difference=float((img0-img1).abs().max()),semantic_image_evidence=False)
        parity_done=False
        for name,prompt,answer in cases:
            ids=sp.encode(prompt,add_bos=True)
            ids_t=torch.tensor([ids],device=device)
            prefix,_,_=model.embed_prefix([image],[image_mask],ids_t,torch.ones_like(ids_t,dtype=torch.bool))
            prefix=prefix.to(torch.bfloat16)
            target=sp.encode(answer,add_eos=True)
            whole=torch.cat([prefix,embed(target[:-1])],dim=1)
            ar=torch.cat([torch.zeros((1,prefix.shape[1]),device=device,dtype=torch.bool),torch.ones((1,len(target)-1),device=device,dtype=torch.bool)],dim=1)
            h,_=forward(whole,ar)
            logits=head(h[:,prefix.shape[1]-1:]).float()
            loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),torch.tensor(target,device=device))
            if not parity_done:
                altered=whole.clone();altered[:,-1,:]=0
                alt,_=forward(altered,ar)
                causal_diff=float((head(alt[:,prefix.shape[1]-1]).float()-logits[:,0]).abs().max())
                if os.environ.get('V4_PRECISION_CHECK')=='1':
                    repeated,_=forward(whole,ar)
                    repeat_diff=float((head(repeated[:,prefix.shape[1]-1]).float()-logits[:,0]).abs().max())
                    unmasked,_=forward(altered,torch.zeros_like(ar))
                    negative_diff=float((head(unmasked[:,prefix.shape[1]-1]).float()-logits[:,0]).abs().max())
                    stage('causal_precision_controls',identical_input_repeat_max_diff=repeat_diff,masked_future_change_max_diff=causal_diff,unmasked_future_negative_control_max_diff=negative_diff)
            start=time.perf_counter();h,cache=forward(prefix)
            decoded=[];steps=48;first_cached=None
            for i in range(steps):
                next_id=int(head(h[:,-1:]).float().argmax(-1).item())
                decoded.append(next_id)
                if next_id==sp.eos_id():break
                h,cache=forward(embed([next_id]),cache=cache,offset=prefix.shape[1]+i)
                if not parity_done and i==0:first_cached=head(h[:,-1:]).float().clone()
            seconds=time.perf_counter()-start
            row=dict(prompt=prompt,target_fixture=answer,target_includes_eos=target[-1]==sp.eos_id(),target_tokens=target,teacher_forcing_loss=float(loss),loss_finite=bool(torch.isfinite(loss)),generated_ids=decoded,generated_text=sp.decode(decoded),stopped_at_eos=decoded[-1]==sp.eos_id(),max_new_tokens=steps,seconds=seconds,tokens_per_second=len(decoded)/seconds)
            stage(name,**row)
            if not parity_done and first_cached is not None:
                full=torch.cat([prefix,embed([decoded[0]])],dim=1)
                fullar=torch.zeros((1,full.shape[1]),device=device,dtype=torch.bool);fullar[:,-1]=True
                fullh,_=forward(full,fullar)
                delta=float((first_cached-head(fullh[:,-1:]).float()).abs().max())
                stage('cache_position_and_causal_mask',max_abs_logit_difference=delta,tolerance=0.25,within_tolerance=delta<=0.25,future_answer_first_logit_difference=causal_diff,label_shift='logits at prefix last position predict first target token; each answer position predicts next; EOS included')
                if os.environ.get('V4_PRECISION_CHECK')=='1':
                    full_logits=head(fullh[:,-1:]).float()
                    stage('cache_argmax_control',cached_argmax=int(first_cached.argmax(-1).item()),full_argmax=int(full_logits.argmax(-1).item()),argmax_agrees=bool(torch.equal(first_cached.argmax(-1),full_logits.argmax(-1))))
                parity_done=True
        stage('memory',peak_allocated_bytes=torch.cuda.max_memory_allocated(),allocated_bytes=torch.cuda.memory_allocated())
    result['execution_status']='completed'
except Exception as exc:
    result.update(execution_status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
print('V4_STEP03_JSON='+json.dumps(result,ensure_ascii=True),flush=True)
