"""Immutable companion records without changing the frozen public protocol."""
from dataclasses import dataclass
import hashlib
import json
import math

from ..protocols import (Event, JointBelief, TaskContract, event_clock, fields, freeze,
                         integer, interval, json_object, number, require, text, thaw, timestamp)
from ..adapters.bci.replay import BIND_FIELDS
from ..models.consequence import ObservedTrajectory
from ..scene.state import GroundedScene
from ..scene.tasks import goal_id
from ..scene._legacy.rendering import render_instruction


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ExecutionContract:
    data: object

    def __post_init__(self):
        v=json_object(self.data)
        fields(v,{'proposal','goal_id','instruction','backend_id','backend_spec_sha256','scene_fingerprint',
                  'completion','preconditions','abort_policy'})
        p=TaskContract(v['proposal']).to_dict()
        require(p['status']=='proposed','proposed_contract_required')
        require(v['goal_id']==goal_id(p['intent']),'contract_goal_mismatch')
        for k in ('instruction','backend_id','backend_spec_sha256','scene_fingerprint'):text(v[k])
        require(len(v['backend_spec_sha256'])==64,'backend_fingerprint_required')
        i=p['intent']; c=v['completion']
        fields(c,{'predicate_version','name','subject_id','object_id'})
        text(c['predicate_version'])
        require((c['name'],c['subject_id'],c['object_id'])==
                ({'pick':'holding','place':'inside','fold':'folded'}[i['action']],i['source_id'],i['destination_id']),
                'completion_condition_mismatch')
        require(v['preconditions']=={'visible_instance_ids':[x for x in [i['source_id'],i['destination_id']] if x is not None],
                                    'scene_version_must_match':True},'precondition_mismatch')
        require(v['abort_policy']=={'on_scene_or_task_change':'invalidate_and_request_stop',
                                   'on_unknown_submission':'pause_no_automatic_resubmit',
                                   'on_stop_request':'latch_until_feedback_no_resume'},'abort_policy_mismatch')
        object.__setattr__(self,'data',freeze(v))

    @classmethod
    def from_scene(cls,proposal,scene,*,backend_id,backend_spec_sha256,predicate_version='endpoint-v1'):
        require(isinstance(proposal,TaskContract) and isinstance(scene,GroundedScene),'validated_contract_and_scene_required')
        p=proposal.to_dict(); i=p['intent']
        require((p['scene_id'],p['scene_revision'])==(scene.data['scene_id'],scene.data['revision']),'contract_scene_mismatch')
        visible={o['instance_id'] for o in scene.data['objects'] if o['visible']}
        require(all(x in visible for x in [i['source_id'],i['destination_id']] if x is not None),'contract_instance_not_visible')
        instruction=render_instruction(scene.legacy_observation(),i)
        return cls({'proposal':p,'goal_id':goal_id(i),'instruction':instruction,'backend_id':backend_id,
            'backend_spec_sha256':backend_spec_sha256,'scene_fingerprint':scene.relevant_fingerprint(i),
            'completion':{'predicate_version':predicate_version,'name':{'pick':'holding','place':'inside','fold':'folded'}[i['action']],
                          'subject_id':i['source_id'],'object_id':i['destination_id']},
            'preconditions':{'visible_instance_ids':[x for x in [i['source_id'],i['destination_id']] if x is not None],
                             'scene_version_must_match':True},
            'abort_policy':{'on_scene_or_task_change':'invalidate_and_request_stop',
                            'on_unknown_submission':'pause_no_automatic_resubmit','on_stop_request':'latch_until_feedback_no_resume'}})

    def to_dict(self):return thaw(self.data)


@dataclass(frozen=True)
class AuthorizationRecord:
    data: object

    def __post_init__(self):
        v=json_object(self.data)
        fields(v,{'authorization_id','scope','contract_sha256','contract_id','task_id','task_revision','scene_id',
            'scene_revision','request','response','model_refs','issued_at','valid_until','belief_at_confirmation',
            'belief_at_authorization','authorization_prediction_gap_s',
            'goal_probability','confirmation_is_noisy','frozen_until_dispatch','consume_once','physical_execution_allowed'})
        require(v['scope']=='offline_fake_only' and v['physical_execution_allowed'] is False,'live_authorization_unavailable')
        require(v['confirmation_is_noisy'] is True and v['consume_once'] is True and v['frozen_until_dispatch'] is True,'authorization_freeze_required')
        for k in ('authorization_id','contract_sha256','contract_id','task_id','scene_id'):text(v[k])
        issued,valid=timestamp(v['issued_at']),timestamp(v['valid_until'])
        require(valid>issued,'authorization_already_expired')
        q,r=Event.parse(v['request']).data,Event.parse(v['response']).data
        require(q['kind']=='REQUEST' and r['kind']=='RESPONSE' and q['mode']==r['mode']=='CONFIRM' and
                r['response_to']==q['event_id'],'actual_confirmation_evidence_required')
        require(all(q[k]==r[k] for k in BIND_FIELDS),'response_binding_mismatch')
        p=q['payload']['contract']
        require((v['contract_id'],v['task_id'],v['task_revision'],v['scene_id'],v['scene_revision'])==
                (p['contract_id'],p['task_id'],p['revision'],p['scene_id'],p['scene_revision']) and
                (v['task_id'],v['task_revision'],v['scene_id'],v['scene_revision'])==
                (q['task_id'],q['task_revision'],q['scene_id'],q['scene_revision']),
                'authorization_contract_mismatch')
        require(r['payload']['status']=='OBSERVATION','observed_confirmation_required')
        for k in ('task_revision','scene_revision'):integer(v[k],1)
        accepted=[o['symbol'] for o in q['payload']['options'] if o['semantic_ref']=='confirmation:accept']
        require(len(accepted)==1 and r['payload']['symbol']==accepted[0], 'actual_accept_selection_required')
        window=r['payload']['sample_window']
        require(window is not None and window['source_kind']=='offline_replay','offline_confirmation_required')
        require(r['clock_uncertainty_ms']==0,'authorization_requires_exact_reference_window')
        def response_time(key):
            return interval({'host_id':r['host_id'],'at':window[key],
                             'offset_ms':r['clock_offset_ms'],'uncertainty_ms':r['clock_uncertainty_ms']})
        start,end=response_time('started_at')[0],response_time('ended_at')[1]
        expiry=min(event_clock(q,'expires_at')[0],event_clock(r,'expires_at')[0],
                   interval({'host_id':q['host_id'],'at':p['valid_until'],
                             'offset_ms':q['clock_offset_ms'],'uncertainty_ms':q['clock_uncertainty_ms']})[0])
        require(start>=event_clock(q)[1] and end<=event_clock(r)[0] and end<expiry and
                event_clock(r)[1]<=issued<expiry and valid<=expiry,'authorization_time_outside_confirmation')
        number(v['authorization_prediction_gap_s'],0)
        require(math.isclose(v['authorization_prediction_gap_s'],(issued-end).total_seconds(),
                             abs_tol=1e-9,rel_tol=0),'authorization_prediction_gap_mismatch')
        confirmed=JointBelief(v['belief_at_confirmation']).data
        authorized=JointBelief(v['belief_at_authorization']).data
        require(confirmed['as_of_event_id']==r['event_id'],'confirmation_belief_evidence_mismatch')
        require(confirmed['goal_ids']==authorized['goal_ids'] and confirmed['mode_ids']==authorized['mode_ids'],
                'authorization_belief_axes_mismatch')
        canonical_goal=goal_id(thaw(p['intent']))
        require(canonical_goal in confirmed['goal_ids'] and canonical_goal in authorized['goal_ids'],
                'authorization_goal_not_in_belief')
        number(v['goal_probability'],0,1)
        marginal=math.fsum(row['probability'] for row in authorized['mass'] if row['goal_id']==canonical_goal)
        require(math.isclose(v['goal_probability'],marginal,abs_tol=1e-9,rel_tol=0),
                'authorization_goal_probability_mismatch')
        refs=v['model_refs']
        hash_fields={'channel_sha256','transition_sha256','confirmation_sha256','policy_sha256','backend_spec_sha256'}
        fields(refs,hash_fields|{'channel_profile'})
        require(refs['channel_profile']==q['channel_profile_id'],'authorization_channel_profile_mismatch')
        for key in hash_fields|{'contract_sha256'}:
            value=v[key] if key=='contract_sha256' else refs[key]
            require(isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value),
                    'model_fingerprint_required')
        object.__setattr__(self,'data',freeze(v))

    def to_dict(self):return thaw(self.data)


@dataclass(frozen=True)
class EndpointEvidence:
    """Bound offline annotation; a completed transport event is insufficient.

    Session ownership and causal timing are checked by the lifecycle, not by this
    standalone immutable record. A calibration trajectory cannot validate itself.
    """
    data: object

    def __post_init__(self):
        value=json_object(self.data)
        fields(value,{'evidence_id','session_id','submission_id','contract_id','contract_revision',
                      'scene_id','scene_revision','observed_at','source','trajectory'})
        for key in ('evidence_id','session_id','submission_id','contract_id','scene_id'):
            text(value[key])
        for key in ('contract_revision','scene_revision'):
            integer(value[key],1)
        require(value['source']=='simulation_fixture_annotation','offline_endpoint_annotation_required')
        interval(value['observed_at'])
        trajectory=ObservedTrajectory(value['trajectory'])
        require(trajectory.data['provenance']=='simulation_fixture','simulation_endpoint_trajectory_required')
        require(trajectory.data['split'] in {'test','calibration_holdout'},'endpoint_validation_split_required')
        object.__setattr__(self,'data',freeze(value))

    def to_dict(self):return thaw(self.data)
