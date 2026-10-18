"""Record only local completed work; real-weight gate remains pending."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'execution/v4'
sys.path[:0]=[str(ROOT.parent/'.runtime/step06_cpu'),str(ROOT),str(ROOT/'tests')]
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    from test_step11_authorization_v4 import public,StubBackend
    from bci_vla.high_level_v4.authorization import HighLevelSession
    tests=load(OUT/'11_local_tests.json');assert tests['successful'] and tests['tests']==28
    assert sha(OUT/'11_local_tests.log')==tests['log_sha256']
    source=load(ROOT/'configs/v4/upstream_subtask_source.json')
    assert all(sha(ROOT/'third_party/openpi_with_subtask'/p)==h for p,h in source['sha256'].items())
    frozen=load(OUT/'10_acceptance.json')
    manifest10=load(OUT/'10_source_manifest.json')
    assert all(sha(ROOT/p)==h for p,h in manifest10['source_sha256'].items())
    # Explicit fake backend provenance is kept adjacent to every local trace.
    p=public();backend=StubBackend();s=HighLevelSession(backend);now=p['as_of_s']
    mainp=s.propose_main(p,now=now);s.confirm(p,proposal_sha256=mainp['sha256'],receipt_id='local-main-sim',accepted=True,now=now+.2)
    spec=importlib.util.spec_from_file_location('step11_probe',OUT/'step11_upstream_probe.py');probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
    generated_public=probe.confirmed_public(p,s.main)
    sub=s.propose_subtask(p,now=now+.3,gripper_state='empty')
    s.confirm(p,proposal_sha256=sub['sha256'],receipt_id='local-sub-sim',accepted=True,now=now+.4)
    dispatch=s.consume_virtual(p,now=now+.5)
    dump(OUT/'11_local_virtual_trace.json',{'scope':'StubBackend scripted text; no real model inference',
         'journal':s.journal,'result':dispatch,'confirmed_public_schema_passed':True,
         'confirmed_model_input':generated_public,'VLA_called':False})
    blocked={'action':'upload current project source/schema/config plus pinned upstream methods and run isolated development probe',
             'destination':load(OUT/'11_code_manifest1.json')['remote_job'],
             'decision':'REJECTED_BY_AUTOMATIC_APPROVAL_REVIEW',
             'reason':'用户未明确授权将这批私有源代码外传到该具体目的地',
             'upload_performed':False,'probe_executed':False,
             'pending_human_authorization':True,'source_files':35,'bundle_bytes':(OUT/'11_code_bundle1.zip').stat().st_size,
             'bundle_sha256':sha(OUT/'11_code_bundle1.zip'),
             'excluded_uploads':['data','images','EEG','weights','credentials']}
    dump(OUT/'11_remote_approval_status.json',blocked)
    acceptance={'step':'V4-11','status':'PARTIAL_LOCAL_MECHANISMS_PASSED_REMOTE_REAL_WEIGHTS_PENDING',
        'upstream':{'url':source['url'],'commit':source['commit'],'source_unchanged':True,
                    'method_reuse':'exact AST method bodies with PyTorch backend interop',
                    'native_JAX_checkpoint_run':False,'upstream_joint_training_reproduced':False},
        'local_tests':tests,'prior_step10_sources_unchanged':True,'step11_overall_complete':False,
        'actual_VLM_inference':False,'same_VLM_real_language_gate':'NOT_RUN',
        'formal_test_accessed':False,'action_sampling':False,'real_EEG':False,'physical_actions':False,
        'server_upload_status':blocked,'next_command_after_authorization':'python execution/v4/run_step11_upstream.py 1 --execute-prepared',
        'known_gaps':['real JAX/Torch interop execution','current LoRA transfer to upstream prompt format',
                      'Step08 full training and ablation coverage','placement/multistep/dynamic action grounding']}
    dump(OUT/'11_acceptance.json',acceptance)
    progress_path=ROOT/'execution/v4_plan_progress.json';progress=load(progress_path)
    before=deepcopy(progress)
    backup=OUT/'11_progress_before.json'
    if not backup.exists():dump(backup,before)
    step=next(s for s in progress['steps'] if s['step_id']=='V4-11')
    step.update(status='部分完成：本地机制通过，指定上游实权重接入复验待上传授权',
                actual_artifacts=['bci_vla/high_level_v4/authorization.py','bci_vla/high_level_v4/upstream_subtask.py',
                    'configs/v4/upstream_subtask_source.json','execution/v4/11_acceptance.json','execution/v4/11_local_tests.json',
                    'execution/v4/11_local_virtual_trace.json','execution/v4/11_code_manifest1.json','docs/v4/11_authorization.txt'],
                verification={'status':acceptance['status'],'local_tests':28,'actual_VLM_inference':False,
                              'report':'execution/v4/11_acceptance.json'})
    progress.update(current_stage='V4-11_local_mechanisms_passed_pending_remote_upload_authorization',next_step='V4-11_real_weight_probe',
                    latest_handoff='execution/v4/handoff_step11.txt',
                    authorization_this_turn='步骤11；子任务改用用户指定 openpi_with_subtask；远程上传被自动审批拒绝，等待明确上传授权')
    assert [s for s in before['steps'] if s['step_id']!='V4-11']==[s for s in progress['steps'] if s['step_id']!='V4-11']
    dump(progress_path,progress)
    handoff=('V4-11部分完成：按用户要求采用LisavilaLee/openpi_with_subtask，固定99c738a。\n'
        '复用上游方法体+当前PyTorch同VLM互操作；没有把上游能力算作自有创新。\n'
        '本地28项授权/上游接口机制测试通过，均为显式替身，不是真实VLM能力。\n'
        '主任务确认仅授权规划，子任务需单独确认；STOP/CANCEL/版本变更撤销并清队列缓存。\n'
        '35文件约103KB源码包已准备：execution/v4/11_code_manifest1.json；无本地数据/权重/凭据。\n'
        '自动审批拒绝向现有服务器新隔离目录上传源码；未上传、未运行实权重探针。\n'
        '取得明确上传授权后执行 python execution/v4/run_step11_upstream.py 1 --execute-prepared。\n'
        '当前LoRA旧提示格式不代表已通过上游Task/Subtask格式，真实复验可能需要适配训练。\n'
        '步骤11整体未完成，旧08缺口/09冻结/10证据/正式test封存/EEG和实体排除保持。\n')
    (OUT/'handoff_step11.txt').write_text(handoff,encoding='utf-8')
    (OUT/'handoff.txt').write_text(handoff,encoding='utf-8')
    for name in ('README.txt','execution/handoff.txt','execution/worklog.txt'):
        with (ROOT/name).open('a',encoding='utf-8') as f:f.write('\n'+handoff)
    files=[ROOT/r['path'] for r in load(OUT/'11_code_manifest1.json')['files']]
    files += [ROOT/'bci_vla/high_level_v4/authorization.py',ROOT/'bci_vla/high_level_v4/upstream_subtask.py',
              ROOT/'tests/test_step11_authorization_v4.py',ROOT/'tests/test_upstream_subtask_v4.py',
              ROOT/'docs/v4/11_authorization.txt',OUT/'11_local_tests.json',OUT/'11_local_tests.log',
              OUT/'11_local_virtual_trace.json',OUT/'11_acceptance.json',OUT/'11_remote_approval_status.json',
              OUT/'run_step11_upstream.py',OUT/'verify_step11_local.py',OUT/'finalize_step11_partial.py']
    dump(OUT/'11_evidence_manifest.json',{'scope':'local mechanism evidence; real inference not performed',
         'created_at_utc':datetime.now(timezone.utc).isoformat(),
         'files':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in sorted(set(files))]})
    print(json.dumps({'status':acceptance['status'],'tests':28,'upload_performed':False}))

if __name__=='__main__':main()
