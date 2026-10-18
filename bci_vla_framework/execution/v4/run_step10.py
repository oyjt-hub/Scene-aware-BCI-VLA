"""Frozen local development acceptance. Never loads a VLA or a formal test."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT.parent/'.runtime/step06_cpu'),str(ROOT)]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


class AlwaysWait:
    name='AlwaysWait'
    def decide(self,public):return {'action':'WAIT','reason':'ALWAYS_WAIT_NEGATIVE_CONTROL'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,required=True);args=parser.parse_args()
    out=ROOT/f'runs/step10_development/v{args.attempt}'
    if out.exists():raise FileExistsError('Append-only development acceptance')
    out.mkdir(parents=True)
    from bci_vla.calibration_v4 import policy_hash
    from bci_vla.calibration_v4.policy import _validate_config
    from bci_vla.evaluation_v4.development import score_trace,audit_trace,aggregate,paired_scene_interval,sample_size_draft,numerically_equal
    from bci_vla.evaluation_v4.runner import run_episode,submission_request
    from bci_vla.interaction_contract_v4.validation import canonical_digest,ContractError
    from execution.v4.step09_calibrate_and_policy import ConnectorPredictor,CalibratedPolicy,SCENARIOS
    from execution.v4.build_step04_data import generate
    activation=load(ROOT/'configs/v4/policy_activation.json')
    previous=load(ROOT/activation['report'])
    assert activation['enabled'] and activation['simulated_interaction_only'] and not activation['VLA_execution_enabled']
    assert previous['policy_operating_constraints_passed'] and not previous['formal_test_accessed']
    selected=activation['selected_policy_id'];plain=selected.rsplit('_',1)[0]+'_plain'
    configs={};artifacts={}
    for name,pid in [('Full_frozen',selected),('Full_no_recheck',plain)]:
        cfg=load(ROOT/f'execution/v4/09_calibration_evidence7/results/artifacts/{pid}/policy.json')
        _validate_config(cfg);artifact=previous['calibrations'][pid]
        assert policy_hash(cfg)==artifact['binding']['policy_sha256']
        configs[name]=cfg;artifacts[name]=artifact
    assert canonical_digest({k:v for k,v in configs['Full_frozen'].items() if k!='confirmation_gate'})==canonical_digest({k:v for k,v in configs['Full_no_recheck'].items() if k!='confirmation_gate'})
    data_root=ROOT/'data/v4';manifest_path=data_root/'captures/capture_manifest.json'
    assert sha(manifest_path)==previous['data_binding']['capture_manifest_sha256']
    captures=[c for c in load(manifest_path)['episodes'] if c['partition']=='validation']
    assert len(captures)==8 and len({c['group_id'] for c in captures})==2
    # The capture catalogue contains development only; do not open any formal
    # test source, future holdout IDs, or calibration rows for fitting.
    assert all(c['partition']=='validation' for c in captures)
    evidence=load(ROOT/'execution/v4/09_checkpoint_manifest3.json')
    weights=next(f for f in evidence['files'] if f['name']=='connector.safetensors')
    assert sha(weights['local_path'])==weights['sha256']
    source_files=['bci_vla/evaluation_v4/development.py','execution/v4/run_step10.py']+list(configs['Full_frozen']['implementation_binding'])
    source_before={p:sha(ROOT/p) for p in source_files}
    preflight={'scope':'development validation only; frozen LIBERO captures and simulated BCI; public fixed pick action',
        'registered_before_replay':datetime.now(timezone.utc).isoformat(),'source_sha256':source_before,
        'checkpoint_sha256':weights['sha256'],'capture_manifest_sha256':sha(manifest_path),
        'config_sha256':{k:policy_hash(v) for k,v in configs.items()},
        'binding':{k:v['binding'] for k,v in artifacts.items()},
        'units':{'scene_groups':2,'base_episodes':8,'world_conditions':4},
        'metrics':'scene macro of within-base mean resolution; scene macro of base-any-error; all conditions including waits in denominator',
        'thresholds_refit':False,'training':False,'formal_test_accessed':False,'VLM_loaded':False,'VLA_called':False}
    preflight['replay_comparison']={'floating_abs_tolerance':1e-12,'all_non_float_fields':'exact','changed_raw_publics':'must also have exactly equal tensorized model inputs'}
    dump(out/'preflight.json',preflight)
    import torch
    from safetensors.torch import load_file
    from bci_vla.neural_interaction_v4.model import matched_models
    torch.set_num_threads(4);torch.manual_seed(0)
    model=matched_models()['Full'].eval().requires_grad_(False)
    model.load_state_dict(load_file(weights['local_path']),strict=True)
    predictor=ConnectorPredictor(model)
    def policy(name):
        if name=='AlwaysWait':return AlwaysWait()
        a=artifacts[name]
        return CalibratedPolicy(predictor,configs[name],a['binding'],a['temperatures']['intent'],a['temperatures']['channel'],name)
    results={name:[] for name in ['Full_frozen','Full_no_recheck','AlwaysWait']}
    audits=Counter();trace_hashes=[];reference_matches=[];bitwise_matches=[];tensor_pairs_checked=0;sample_for_negative=None
    previous_zip=Path(previous['local_evidence_archive'])
    assert sha(previous_zip)==previous['local_evidence_archive_sha256']
    def signature(trace):
        return canonical_digest({'journal':trace['journal'],'submissions':trace['submissions'],'scoring':trace['private_scoring_only']})
    with zipfile.ZipFile(previous_zip) as z:
        for capture in captures:
            for scenario in SCENARIOS:
                _,world=generate(deepcopy(capture),scenario)
                for name in results:
                    trace=run_episode(capture,world,policy(name),data_root,horizon=24,declared_action='pick')
                    audit=audit_trace(trace,configs.get(name));audits.update(audit)
                    summary=score_trace(trace,world,capture['group_id']);summary['scenario']=scenario
                    results[name].append(summary)
                    path=out/'episodes'/f"{name}_{capture['base_episode_id']}_{scenario}.json"
                    dump(path,{'summary':summary,'trace':trace,'audit':audit});trace_hashes.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
                    if name=='Full_frozen':
                        old=json.loads(z.read(f"results/artifacts/{selected}/validation/{capture['base_episode_id']}_{scenario}.json"))
                        old_trace=old['trace']
                        bitwise_matches.append(signature(trace)==signature(old_trace))
                        reference_matches.append(all(numerically_equal(trace[k],old_trace[k]) for k in ('journal','submissions','private_scoring_only')))
                        from bci_vla.neural_interaction_v4.tensorizer import collate_public
                        for current_j,old_j in zip(trace['journal'],old_trace['journal']):
                            if current_j.get('input') and canonical_digest(current_j['input'])!=canonical_digest(old_j['input']):
                                inputs,_=collate_public([current_j['input'],old_j['input']])
                                assert all(torch.equal(v[0],v[1]) for v in inputs.values()),'Cross-platform raw scores changed model tensors'
                                tensor_pairs_checked+=1
                        if sample_for_negative is None:
                            sample_for_negative=next((deepcopy(j) for j in trace['journal'] if j['kind']=='INTERACTION' and j['simulated_submission']),None)
            print(json.dumps({'event':'development_capture_complete','episode':capture['base_episode_id']},ensure_ascii=False),flush=True)
    assert all(reference_matches),'Frozen discrete trajectory differs from Step09 validation replay'
    summaries={name:aggregate(rows) for name,rows in results.items()}
    assert summaries['AlwaysWait']['resolution']==0 and summaries['AlwaysWait']['wrong_episode_rate']==0
    paired=paired_scene_interval(results['Full_frozen'],results['Full_no_recheck'])
    # Perturbation and control runs are separate engineering cases; they do not
    # increase the independent sample count of the primary 8 base episodes.
    robustness=[]
    for capture in captures:
        _,world=generate(deepcopy(capture),'stable')
        for case,kwargs in [('STOP_immediate',{'control_at':(0.,'STOP')}),('STOP_mid',{'control_at':(12.,'STOP')}),
                            ('CANCEL_late',{'control_at':(20.,'CANCEL')}),('mapping_change',{'mapping_at':12.})]:
            trace=run_episode(capture,world,policy('Full_frozen'),data_root,horizon=24,declared_action='pick',**kwargs)
            audit_trace(trace,configs['Full_frozen'])
            if 'control_at' in kwargs:
                at,role=kwargs['control_at'];assert trace['journal'][-1]['kind']=='CONTROL'
                assert not any(s['submitted_at_s']>=at for s in trace['submissions'])
            else:
                for j in trace['journal']:
                    if j.get('input') and j['at_s']>=12:
                        assert j['input']['context']['mapping_revision']==2
                        assert all(e['question']['context']==j['input']['context'] for e in j['input']['public_state']['diagnostics']['receipts'])
            scored=score_trace(trace,world,capture['group_id']);scored['case']=case;robustness.append(scored)
            dump(out/'robustness'/f"{capture['base_episode_id']}_{case}.json",{'summary':scored,'trace':trace})
    negative=[]
    assert sample_for_negative is not None
    j=sample_for_negative;p=j['public_after'];e=j['event'];task=e['question']['proposal']
    submission_request(deepcopy(p),deepcopy(task),deepcopy(e),set())
    for case in ['stop_latched','wrong_confirmation','wrong_suggestion','stale_mapping','duplicate_receipt','expired_task']:
        pp,ee,tt=deepcopy(p),deepcopy(e),deepcopy(task);seen=set()
        if case=='stop_latched':pp['public_state']['stop_latched']=True
        elif case=='wrong_confirmation':ee['role']='REJECT'
        elif case=='wrong_suggestion':tt['suggestion_id']='another-proposal'
        elif case=='stale_mapping':tt['context']['mapping_revision']+=1
        elif case=='duplicate_receipt':submission_request(pp,tt,ee,seen)
        elif case=='expired_task':pp['as_of_s']=tt['valid_until_s']
        try:submission_request(pp,tt,ee,seen)
        except ContractError as err:negative.append({'case':case,'rejected':True,'reason':str(err)})
        else:raise AssertionError('Invalid submission was accepted: '+case)
    ambiguity=[];outside=[]
    def declare_pick(public):
        # Explicit shared action menu; never select a source from private truth.
        public['task_options']=[{'task_id':'dev-pick-'+o['instance_id'],'task_revision':1,
            'suggestion_id':'dev-pick-suggestion-'+o['instance_id'],'context':deepcopy(public['context']),
            'action':'pick','source_id':o['instance_id'],'destination_id':None,
            'valid_until_s':public['as_of_s']+120.,'capability_verified':False}
            for o in public['scene']['objects'] if o['object_valid']]
        return public
    for capture in captures:
        records,_=generate(deepcopy(capture),'ambiguous')
        for public,private,_future in records:
            # Distinct possible hidden goals are scorer annotations, never a
            # predictor/policy argument. Equal public inputs give equal decisions.
            if public['public_state']['stop_latched']:continue
            declare_pick(public)
            a=policy('Full_frozen').decide(deepcopy(public));b=policy('Full_frozen').decide(deepcopy(public))
            assert a==b
            ambiguity.append({'input_sha256':canonical_digest(public),'decision':a,'hidden_outcomes_not_forced_POINT':True})
        records,_=generate(deepcopy(capture),'candidate_miss')
        controller=policy('Full_frozen')
        for public,_private,_future in records:
            declare_pick(public)
            decision=controller.decide(public)
            assert decision['action']!='CONFIRM','No-response outside-candidate fixture must not invent a target'
            outside.append({'input_sha256':canonical_digest(public),'action':decision['action']})
    assert source_before=={p:sha(ROOT/p) for p in source_files}
    assert sha(weights['local_path'])==weights['sha256']
    changed=[r for r in results['Full_frozen'] if r['goal_changed']]
    updated=[r['total_update_latency_s'] for r in changed if r['total_update_latency_s'] is not None]
    latency={'changed_conditions':len(changed),'updates_observed':len(updated),'censored':sum(r['update_censored'] for r in changed),
             'mean_latency_among_observed_s':sum(updated)/len(updated) if updated else None,
             'restricted_mean_latency_including_censor_s':sum(r['total_update_latency_s'] if r['total_update_latency_s'] is not None else r['censor_after_change_s'] for r in changed)/len(changed),
             'identified_public_anchor_coverage':0.,'identifiable_response_latency':None,
             'reason':'No trusted noiseless goal declaration exists; noisy consistency is a rule certificate, not mathematical identifiability',
             'old_target_submissions_after_change':sum(r['old_target_submissions_after_change'] for r in changed)}
    result={'status':'PASS_RESTRICTED_DEVELOPMENT_ENGINEERING','scope':preflight['scope'],'preflight':preflight,
            'summary':summaries,'paired_rule_comparison':paired,'goal_update_latency':latency,
            'integrity_audit':dict(audits),'step09_discrete_replays_matched':sum(reference_matches),
            'step09_bitwise_replays_matched':sum(bitwise_matches),'cross_platform_tensor_pairs_checked':tensor_pairs_checked,
            'robustness_runs':len(robustness),'robustness_summaries':robustness,'negative_submissions':negative,
            'ambiguous_public_cases':len(ambiguity),'outside_no_response_cases':len(outside),
            'source_and_weights_unchanged':True,'calibration_refit':False,'training':False,
            'neural_architecture_gain_established':False,'full_step08_complete':False,
            'known_language_failures':{'conflicting_declarations':8,'all_language_exact':'112/120'},
            'formal_test_accessed':False,'VLA_called':False,'VLM_loaded':False,'real_dynamics_advanced':False,
            'trace_manifest':trace_hashes,'future_step11_scope':'adapter/authorization development only; no unrestricted VLM or robot execution'}
    dump(out/'ambiguity_checks.json',{'cases':ambiguity,'outside_cases':outside})
    dump(out/'result.json',result)
    protocol={'version':'V4-10-formal-protocol-draft/0.1','status':'DRAFT_NOT_FINAL_LOCK_NOT_TEST_AUTHORIZATION',
        'formal_test_accessed':False,'planned_final_lock':'V4-13 after all applicable training and calibration',
        'primary_endpoint':'Correct final confirmed goal/task within error-risk and interaction budgets; frozen-capture simulated submission is distinct from robot completion',
        'primary_comparison':'Full vs B4_aux shared-query with matched data/LoRA/binding/parameter budget and shared fixed inquiry contract',
        'additional_comparisons':['B0 recent choice','B1 binary filter','B2 same-information text','B3_aux matched MLP'],
        'current_comparison_scope':'Full same-checkpoint confirmation-gate engineering ablation only; not architecture comparison',
        'risk':{'threshold':.1,'development_basis':'scene-macro episode-any-error point estimate only',
                'formal_basis':'one-sided 95% upper bound on scene-any-error as conservative gate; also report episode-any-error and submission-conditional error',
                'assumptions':'independent representative scene/task clusters; equal locked initialization/scenario composition; unavailable capacity is a blocker'},
        'budgets':configs['Full_frozen']['budgets'],'horizon_s':24,'minimum_effective_gain':.1,
        'gain_status':'provisional design target, not tuned to formal outcomes; finalize effect and sample assumptions using development only',
        'statistics':{'unit':'paired scene/task clusters; keep all initializations and scenarios together',
                      'effect_interval':'paired scene-cluster bootstrap after adequate independent cluster count',
                      'primary_alpha':.05,'multiplicity':'one prespecified primary comparison; secondary/ablation results descriptive or Holm-corrected within declared family',
                      'all_failures_in_denominator':True,'always_wait_is_unresolved':True,'training_seeds_not_independent_test_units':True},
        'sample_size':sample_size_draft(),
        'latency':{'total':'from private scorer change time to first correct updated confirmation; unresolved censored at horizon',
                   'identifiable_anchor':'trusted explicit public goal declaration only; no such anchor in present raw-noise captures',
                   'conditional_latency':'report only where anchor exists; publish coverage and censoring, never relabel noisy agreement as identified truth'},
        'freeze_requirements':['all baseline/ablation training on train/validation','same new binding mechanism for B3/B4/Full',
            'method-specific checkpoint/schema/policy calibration','shared fixed inquiry for structure comparison',
            'adequate new independent heldout scene capacity','full VLM and LIBERO-action interface gates when claiming execution',
            'exact parsing/timeout/no-response failure accounting','write immutable V4-13 lock before one-time formal test opening'],
        'known_blockers':['full Step08 hidden-intent and placement supervision incomplete','8 conflicting-declaration language failures',
                          'only two development scene groups; architecture effect and formal sample feasibility unverified'],
        'current_frozen_development_binding':activation['binding'],'no_test_result_tuning':True}
    dump(ROOT/'execution/v4/10_evaluation_protocol_draft.json',protocol)
    dump(ROOT/'execution/v4/10_development_result.json',result)
    print(json.dumps({'status':result['status'],'Full':summaries['Full_frozen'],'latency':latency},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
