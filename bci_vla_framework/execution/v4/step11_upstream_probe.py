"""Real frozen developer VLM + requested upstream subtask code; no actions."""
import os
os.environ.update(JAX_PLATFORMS='cpu',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TORCHDYNAMO_DISABLE='1')
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
PREVIOUS=Path('/home/oyjt/Scene-aware-BCI-VLA-runs/v4_step09_binding_20261018_v3')
CAPTURE=Path('/home/oyjt/Scene-aware-BCI-VLA-runs/v4_step04_libero_data_20261017_v1')
BASE=Path('/home/oyjt/workspace/RLinf-RECAP/models/baseline_sft/RLinf-Pi05-LIBERO-SFT/model.safetensors')
BASE_SHA='4616e4c1966a5fb063b442ea5d69857f45e873222d839b794634d87d252ab9e7'
CONNECTOR_SHA='427fba987ad3b8a6b0b4e3dafb133f86980345706698b8b8ba48a41969f72464'
LORA_SHA='28c18b8f262e234b49953dda81e732191df3f1700064e61d89813aba4bbf7c3b'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(8*1024**2):h.update(b)
    return h.hexdigest()

def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def confirmed_public(public, main):
    """Public exact-proposal receipt from the trusted simulated confirmation API."""
    p=deepcopy(public); now=main['confirmed_at_s']
    task={k:deepcopy(main[k]) for k in ('suggestion_id','context','action','source_id','destination_id')}
    task.update(task_id='step11-'+main['suggestion_id'],task_revision=1,
                valid_until_s=main['expires_at_s'],capability_verified=False)
    options=[{'option_id':key,'symbol':'s'+str(i),'kind':'DECISION','reference':key} for i,key in enumerate(('accept','reject'))]
    event={'event_id':main['confirmation_receipt_id'],'sequence':len(p['history'])+1,'role':'CONFIRM','original_role':None,
        'question':{'question_id':'q-'+main['confirmation_receipt_id'],'role':'CONFIRM','context':deepcopy(p['context']),
            'layout_id':'step11-confirm','issued_at_s':now-.1,'expires_at_s':main['expires_at_s'],
            'presentation_ack':True,'options':options,'diagnostic_option_id':None,'proposal':task},
        'started_at_s':now-.05,'ended_at_s':now,'valid':True,'invalid_reason':None,
        'evidence':{'score_kind':'HARD_LABEL','values':[{'option_id':'accept','value':1},{'option_id':'reject','value':0}],
                    'selected_option_id':'accept','calibration':None}}
    p['history'].append(event);p['current_event_id']=event['event_id'];p['as_of_s']=now
    p['public_state']['confirmed_task']={'task':task,'confirmation_receipt':deepcopy(event)}
    from bci_vla.data_v4.contracts import validate_public
    validate_public(p)
    return p

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists();args.output.mkdir()
    report={'status':'STARTED','formal_test_accessed':False,'training':False,'action_sampling':False,
            'physical_actions':False,'real_EEG':False,'scope':'declared pick development, upstream subtask integration',
            'cases':[],'language_rechecks':[]}
    def stage(name,**kw):
        print('STEP11_STAGE='+json.dumps(dict(stage=name,**kw)),flush=True)
        dump(args.output/'result.json',report)
    try:
        import numpy as np
        import torch
        import sentencepiece as spm
        from PIL import Image
        from safetensors import safe_open
        from safetensors.torch import load_model,load_file
        from transformers.modeling_utils import no_init_weights
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from bci_vla.high_level_v4.lora import install_high_level_lora
        from bci_vla.neural_interaction_v4.model import matched_models
        from bci_vla.neural_interaction_v4.tensorizer import collate_public
        from bci_vla.prefix_integration_v4.adapter import Pi05HighLevelPrefix
        from bci_vla.prefix_integration_v4.binding import ObjectReferenceBinding,build_reference_bank
        from bci_vla.training_v4.data import load_partition,language_prompt,PROTOCOL
        from bci_vla.high_level_v4.upstream_subtask import UpstreamSubtaskInterop
        from bci_vla.high_level_v4.authorization import HighLevelSession,AuthorizationError
        from bci_vla.interaction_contract_v4.validation import canonical_digest
        torch.set_num_threads(4);torch.manual_seed(0);device=torch.device('cuda:0')
        assert torch.cuda.mem_get_info()[0]>26*1024**3
        report['code_manifest_verified']=all(sha(ROOT/r['path'])==r['sha256'] for r in json.loads((ROOT/'code_manifest.json').read_text())['files'])
        assert report['code_manifest_verified']
        selected=PREVIOUS/'results/Full_seed0/step2400'
        assert sha(BASE)==BASE_SHA and sha(selected/'connector.safetensors')==CONNECTOR_SHA and sha(selected/'language_lora.safetensors')==LORA_SHA
        report['model_binding']={'base':BASE_SHA,'connector':CONNECTOR_SHA,'language_lora':LORA_SHA}
        stage('weights_verified')
        with no_init_weights():base=PI0Pytorch(Pi0Config(pi05=True,dtype='bfloat16'))
        with safe_open(BASE,framework='pt',device='cpu') as f:meta=f.metadata()
        alias='paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight'
        assert meta[alias]=='paligemma_with_expert.paligemma.lm_head.weight'
        base.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight=base.paligemma_with_expert.paligemma.lm_head.weight
        load_model(base,str(BASE),strict=True,device='cpu');base.requires_grad_(False).eval().to(device)
        adapters=install_high_level_lora(base,rank=16,alpha=32)
        lora={f'{name}.{side}':getattr(module,side) for name,module in adapters.items() for side in ('A','B')}
        weights=load_file(str(selected/'language_lora.safetensors'));assert set(lora)==set(weights)
        with torch.no_grad():
            for k,v in lora.items():v.copy_(weights[k].to(device));v.requires_grad_(False)
        # Do not promote historical 03R capability to a different adapter hash.
        bridge=Pi05HighLevelPrefix(base,{},engineering_only=True,checkpoint_tag=BASE_SHA,
                                  high_level_adapters=adapters,adapter_sha256=LORA_SHA)
        models=matched_models();connector=models['Full'];del models
        connector.load_state_dict(load_file(str(selected/'connector.safetensors')));connector.to(device).eval().requires_grad_(False)
        sp=spm.SentencePieceProcessor(model_file='/home/oyjt/.cache/openpi/big_vision/paligemma_tokenizer.model')
        upstream=UpstreamSubtaskInterop(bridge,sp,ROOT/'third_party/openpi_with_subtask',json.loads((ROOT/'configs/v4/upstream_subtask_source.json').read_text()))
        binder=ObjectReferenceBinding(mode='reference',residual_scale=.1,reference_scale=1.)
        frozen_versions={k:p._version for k,p in base.named_parameters()}

        class Backend:
            model_binding=deepcopy(report['model_binding'])
            calls=0
            def reset_generation(self):bridge.reset_generation()
            @torch.no_grad()
            def generate(self,public,mode,gripper,*,confirmed_task=None):
                self.calls+=1
                p=confirmed_public(public,confirmed_task) if mode=='next_subtask' else public
                imagepath=CAPTURE/p['scene']['image_ref']
                rgb=np.array(Image.open(imagepath).convert('RGB').resize((224,224)),copy=True).astype(np.float32)/127.5-1.
                x,metadata=collate_public([p]);x={k:v.to(device) for k,v in x.items()}
                output=connector(x)
                refs=build_reference_bank([p],lambda t:sp.encode(t,add_bos=False),base.paligemma_with_expert.embed_language_tokens,
                                         hidden_size=2048,device=device,dtype=torch.float32)
                soft=binder(output,refs,x['candidate_mask'])
                if mode=='next_subtask':
                    source=next(o['alias'] for o in p['scene']['objects'] if o['instance_id']==confirmed_task['source_id'])
                    result=upstream.generate(task_text='Pick up '+source+'.',rgb=rgb,soft_tokens=soft,max_tokens=32)
                else:
                    prompt=language_prompt(p,PROTOCOL,mode,gripper)
                    ids=torch.tensor([sp.encode(prompt,add_bos=True)],dtype=torch.long,device=device)
                    image=torch.from_numpy(rgb).to(device).permute(2,0,1)[None]
                    pack=bridge.prepare([image],[torch.tensor([True],device=device)],ids,torch.ones_like(ids,dtype=torch.bool),soft_tokens=soft,enabled=True)
                    result=bridge.generate(p,pack,eos_id=sp.eos_id(),max_new_tokens=32)
                    result['raw_text']=sp.decode(result['token_ids'])
                    result['implementation']='existing_main_task_branch'
                result.update(candidate_probabilities=dict(zip(metadata[0]['instance_ids'],output['candidate_probabilities'][0].tolist())),
                              outside_probability=float(output['outside_probability'][0]),model_binding=deepcopy(self.model_binding),
                              public_input_sha256=canonical_digest(p),image_sha256=sha(imagepath))
                return result

        backend=Backend();stage('runtime_loaded',upstream=upstream.functions['generate_subtask'].__code__.co_filename)
        # Developer validation only. No private labels enter the backend/session.
        rows=load_partition(PREVIOUS/'training_data','validation')
        chosen=[];seen=set()
        for row in rows:
            if row['supervision']['family']!='declared_pick':continue
            key=(row['public']['context']['scene_id'],row['supervision']['binding']['source_position_audit_only']) if 'binding' in row['supervision'] else (row['public']['context']['scene_id'],row['public']['selection_event_id'])
            if key not in seen:seen.add(key);chosen.append(row)
            if len(chosen)>=4:break
        for i,row in enumerate(chosen):
            p=row['public'];session=HighLevelSession(backend);now=p['as_of_s'];case={'sample_id':p['sample_id']}
            try:
                proposal=session.propose_main(p,now=now)
                session.confirm(p,proposal_sha256=proposal['sha256'],receipt_id=f'main-{i}',accepted=True,now=now+.2)
                subtask=session.propose_subtask(p,now=now+.3,gripper_state='empty')
                session.confirm(p,proposal_sha256=subtask['sha256'],receipt_id=f'subtask-{i}',accepted=True,now=now+.4)
                case['virtual_dispatch']=session.consume_virtual(p,now=now+.5);case['closed_loop_passed']=True
            except AuthorizationError as e:
                case.update(closed_loop_passed=False,rejected_reason=str(e))
            case['journal']=session.journal;report['cases'].append(case);stage('case_completed',i=i,passed=case['closed_loop_passed'])
        # Existing eight declaration-conflict failures remain reported as raw
        # language failures. Integration must reject them BEFORE any model call.
        conflicts=[r for r in rows if r['supervision']['family']=='conflicting_declarations']
        controls=[]
        for row in conflicts:
            session=HighLevelSession(backend);before=backend.calls
            try:session.propose_main(row['public'],now=row['public']['as_of_s']);rejected=False
            except AuthorizationError:rejected=True
            controls.append({'sample_id':row['public']['sample_id'],'rejected':rejected,'model_calls':backend.calls-before})
        report['conflict_guards']=controls
        report['frozen_weight_versions_unchanged']=all(p._version==frozen_versions[k] for k,p in base.named_parameters())
        report['base_and_selected_files_unchanged']=sha(BASE)==BASE_SHA and sha(selected/'connector.safetensors')==CONNECTOR_SHA and sha(selected/'language_lora.safetensors')==LORA_SHA
        report['summary']={'closed_loops':sum(c['closed_loop_passed'] for c in report['cases']),'attempts':len(report['cases']),
                           'conflicts_rejected':sum(c['rejected'] for c in controls),'conflicts':len(controls)}
        report['status']='PASS_RESTRICTED_UPSTREAM_INTEGRATION' if report['summary']['closed_loops']==len(chosen) and chosen else 'INTEGRATION_READY_LANGUAGE_GATE_NOT_PASSED'
        stage('complete',status=report['status'],summary=report['summary'])
    except Exception:
        report['status']='PROCESS_ERROR';report['error']=traceback.format_exc();print(report['error'],flush=True)
    finally:dump(args.output/'result.json',report)

if __name__=='__main__':main()
