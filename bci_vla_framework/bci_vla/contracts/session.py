"""Offline fake-only lifecycle. Evidence, authorization and execution are distinct."""
from datetime import datetime, timezone, timedelta

from ..protocols import Event, TaskContract, JointBelief, ProtocolError, freeze, thaw, require, number, text, timestamp, event_clock, interval, json_object
from ..adapters.bci.replay import ReplayBCIAdapter, context
from ..adapters.vla import FakeVLAAdapter, DryRunVLAAdapter
from ..core.joint_belief import JointBeliefFilter, TaskQuestionBinding
from ..core.diagnostic_policy import PlanningState
from ..models.consequence import ObservedTrajectory, evaluate_trajectory
from ..monitoring.journal import EventJournal
from ..scene.state import GroundedScene, IdentityLedger
from ..scene.tasks import goal_id
from ..scene._legacy.planning import candidate_key, generate_candidates
from .records import ExecutionContract, AuthorizationRecord, EndpointEvidence, digest

STATES={'COLLECTING','CLARIFYING','PROBING','AWAITING_CONFIRMATION','AWAITING_AUTHORIZATION','AUTHORIZED',
        'SUBMITTED','SUBMISSION_UNKNOWN','EXECUTING','VERIFYING','PAUSED','SUCCEEDED','FAILED','CANCELLED','EXPIRED'}
ACTIVE={'SUBMITTED','SUBMISSION_UNKNOWN','EXECUTING','VERIFYING'}


def at(value):
    if isinstance(value,datetime):
        require(value.tzinfo is not None and value.utcoffset() is not None,'timezone_required')
        return value.astimezone(timezone.utc)
    return timestamp(value)


class InteractionSession:
    def __init__(self,*,current_context,scene,belief_filter,policy,backend,config,started_at):
        require(type(backend) in {FakeVLAAdapter,DryRunVLAAdapter},'only_fake_or_dry_run_backend_allowed')
        require(isinstance(scene,GroundedScene) and isinstance(belief_filter,JointBeliefFilter),'validated_scene_filter_required')
        require(config['scope']=='offline_fake_only' and config['physical_execution_allowed'] is False,'offline_session_only')
        self.context=dict(context(current_context)); self.scene=scene; self.filter=belief_filter; self.policy=policy
        self.backend=backend; self.config=freeze(config)
        require(dict(belief_filter._context)==self.context,'filter_context_mismatch')
        require(thaw(policy.config['context'])==self.context,'policy_context_mismatch')
        require((scene.data['scene_id'],scene.data['revision'])==(self.context['scene_id'],self.context['scene_revision']),'scene_context_mismatch')
        require(scene.data['identity']['session_id']==self.context['session_id'],'scene_identity_session_mismatch')
        require(belief_filter.channel_model.data==policy.channel.data and belief_filter.transition_model.data==policy.transition.data,'filter_policy_model_mismatch')
        number(config['minimum_goal_probability'],0,1); number(config['max_observation_age_s'],0)
        number(config['max_confirmation_age_s'],0)
        self.binding=TaskQuestionBinding(config['task_binding'],belief_filter.vocabulary)
        require(config['accept_symbol']==policy.config['actions']['accept_symbol'],'confirmation_mapping_mismatch')
        self.adapter=ReplayBCIAdapter(self.context,response_hosts=set(config['response_hosts']),control_hosts=set(config['control_hosts']))
        self.journal=EventJournal(self.context['session_id'])
        self.state='COLLECTING'; self.reason='session_started'; self.contract=None; self.authorization=None
        self.confirmation=None; self.submission=None; self.submission_contract=None
        self.authorization_consumed=False; self.stop_latched=False; self.stop_feedback='NOT_REQUESTED'
        self.submission_at=None; self.completion_feedback_at=None; self._started_at=at(started_at)
        self._generation_valid=True; self._request_count=0; self._proposal_ids=set(); self._confirmation_contexts=set()
        self._consumed_evidence=set(); self._feedback_ids=set(); self._trajectory_ids=set(); self._submission_digest=None
        self.model_refs=self._active_model_refs()
        initial=self.filter.snapshot()
        self._record('SESSION_STARTED',{'scope':'offline_fake_only','replay_bundle':{
            'current_context':dict(self.context),'scene':scene.to_dict(),
            'filter_config':{'vocabulary':thaw(self.filter.vocabulary.data),'initial_joint':initial['belief'],
                'initial_at':initial['as_of_reference_time'],'layout_id':self.filter.layout_id,
                'outlier_weight':self.filter.outlier_weight,'current_context':dict(self.context)},
            'channel_config':thaw(self.filter.channel_model.data),'transition_config':thaw(self.filter.transition_model.data),
            'policy_config':thaw(self.policy.config),'outcome_config':thaw(self.policy.outcomes.data),
            'session_config':thaw(self.config),'backend_spec':thaw(backend.spec.data)}},self._now(started_at))

    def _active_model_refs(self):
        return {'channel_profile':self.context['channel_profile_id'],
            'channel_sha256':digest(thaw(self.filter.channel_model.data)),
            'transition_sha256':digest(thaw(self.filter.transition_model.data)),
            'confirmation_sha256':digest(thaw(self.filter.channel_model.data['matrices']['CONFIRM'])),
            'policy_sha256':digest(thaw(self.policy.config)),'backend_spec_sha256':digest(thaw(self.backend.spec.data))}

    def _now(self,value):
        now=at(value)
        require(not self.journal.records or now>=at(self.journal.records[-1]['at']),'session_clock_reversed')
        return now

    def snapshot(self):
        return {'schema_version':'interaction-session/1.0','scope':'offline_fake_only','context':dict(self.context),
            'state':self.state,'reason':self.reason,'contract':None if self.contract is None else self.contract.to_dict(),
            'authorization':None if self.authorization is None else self.authorization.to_dict(),
            'authorization_consumed':self.authorization_consumed,'submission':None if self.submission is None else dict(self.submission),
            'submission_at':self.submission_at,'completion_feedback_at':self.completion_feedback_at,
            'submission_contract':None if self.submission_contract is None else self.submission_contract.to_dict(),
            'stop_latched':self.stop_latched,'stop_feedback':self.stop_feedback,'generation_valid':self._generation_valid,
            'model_refs':dict(self.model_refs),'filter':self.filter.snapshot(),
            'execution_allowed':False,'physical_stop_verified':False}

    def _record(self,kind,event,now):
        state=self.snapshot()
        self.journal.append(kind,{'event':event,'state':state,'state_sha256':digest(state)},at=now)

    def _revoke(self):
        self.authorization=None; self.confirmation=None

    def _current_contract(self,now):
        require(self.contract is not None,'proposal_required')
        p=self.contract.data['proposal']
        require((p['task_id'],p['revision'],p['scene_id'],p['scene_revision'])==
                (self.context['task_id'],self.context['task_revision'],self.context['scene_id'],self.context['scene_revision']),'stale_contract_context')
        require(now<timestamp(p['valid_until']),'contract_expired')
        require(self.contract.data['scene_fingerprint']==self.scene.relevant_fingerprint(thaw(p['intent'])),'scene_fingerprint_changed')
        require(self.contract.data['backend_spec_sha256']==digest(thaw(self.backend.spec.data)),'backend_spec_changed')
        require(self._generation_valid,'models_or_context_require_explicit_rebuild')
        require(self._active_model_refs()==self.model_refs,'model_changed_requires_explicit_rebuild')
        return p

    def _prediction_copy(self,snapshot,now):
        snap=json_object(snapshot); gap=(now-at(snap['as_of_reference_time'])).total_seconds()
        require(gap>=0,'belief_time_regression')
        cells={(r['goal_id'],r['mode_id']):r['probability'] for r in snap['belief']['mass']}
        known=[(g,z) for g in self.filter.vocabulary.known_goal_ids for z in self.filter.channel_model.data['mode_ids']]
        matrix=self.filter.transition_model.matrix(gap); predicted=dict(cells)
        for j,key in enumerate(known):predicted[key]=sum(cells.get(old,0)*matrix[i][j] for i,old in enumerate(known))
        snap['belief']=JointBelief(dict(snap['belief'],mass=[{'goal_id':g,'mode_id':z,'probability':p} for (g,z),p in predicted.items()])).to_dict()
        snap['as_of_reference_time']=now.isoformat()
        snap['goal_marginals']={g:sum(p for (h,z),p in predicted.items() if h==g) for g in self.filter.goal_ids}
        snap['channel_marginals']={z:sum(p for (g,w),p in predicted.items() if w==z) for z in self.filter.mode_ids}
        return snap,gap

    def propose(self,proposal,*,now):
        now=self._now(now)
        require(not self.stop_latched and self.state not in ACTIVE,'proposal_while_execution_unresolved')
        require(self.adapter.pending_request is None,'interaction_pending')
        p=TaskContract(proposal)
        require(p.data['contract_id'] not in self._proposal_ids,'contract_id_reused')
        require((p.data['task_id'],p.data['revision'])==(self.context['task_id'],self.context['task_revision']),'proposal_task_version_mismatch')
        require(now<timestamp(p.data['valid_until']),'contract_expired')
        require(goal_id(thaw(p.data['intent'])) in self.filter.vocabulary.known_goal_ids,'proposal_outside_vocabulary')
        require(candidate_key(thaw(p.data['intent'])) in {candidate_key(i) for i in generate_candidates(self.scene.legacy_observation())},'intent_not_in_current_scene_grammar')
        require(p.data['intent']['action'] in self.config['simulation_supported_actions'],'backend_action_capability_unverified')
        c=ExecutionContract.from_scene(p,self.scene,backend_id=self.config['backend_id'],
            backend_spec_sha256=self.model_refs['backend_spec_sha256'],predicate_version=self.config['predicate_version'])
        self.contract=c; self._proposal_ids.add(p.data['contract_id']); self._revoke()
        self.state='CLARIFYING'; self.reason='proposal_requires_actual_confirmation'
        self._record('CONTRACT_PROPOSED',c.to_dict(),now)
        return c

    def begin_interaction(self,mode,*,now,monotonic_now,expires_in_s=10,prompt_symbol=None,presentation_ack=False,presented_at=None):
        now=self._now(now); number(expires_in_s,0); require(expires_in_s>0,'invalid_expiry')
        require(mode in {'TASK_QUERY','CHANNEL_PROBE','CONFIRM'},'unsupported_session_interaction')
        require(not self.stop_latched and self.state not in ACTIVE,'interaction_while_execution_unresolved')
        require(self._generation_valid,'models_or_context_require_explicit_rebuild')
        require(self.adapter.pending_request is None,'interaction_pending')
        require(self.filter.snapshot()['point_inference_ready'],'filter_not_ready')
        expiry=now+timedelta(seconds=expires_in_s); contract=None; prompt=None
        if mode=='TASK_QUERY':
            question=self.binding.data['question']; options=thaw(self.binding.data['options'])
        elif mode=='CHANNEL_PROBE':
            require(prompt_symbol in self.policy.channel.data['scope']['symbols'],'unknown_prompt_symbol')
            require(type(presentation_ack) is bool,'invalid_ack')
            require(not presentation_ack or presented_at is not None,'actual_presentation_ack_required')
            prompt={'symbol':prompt_symbol,'source':'display_controller','presented_at':presented_at,'presentation_ack':presentation_ack}
            question='Diagnostic only: attend the highlighted marker; no robot task.'
            options=[{'option_id':'probe-'+s,'symbol':s,'semantic_ref':'probe:'+s} for s in self.policy.channel.data['scope']['symbols']]
        else:
            p=self._current_contract(now); contract=thaw(p)
            key=tuple(self.context[k] for k in sorted(self.context))
            require(key not in self._confirmation_contexts,'confirmation_already_attempted_for_version')
            expiry=min(expiry,timestamp(p['valid_until']))
            question='Confirm the exact proposed task and current scene version?'
            accept=self.config['accept_symbol']; reject=next(s for s in self.policy.channel.data['scope']['symbols'] if s!=accept)
            options=[{'option_id':'accept','symbol':accept,'semantic_ref':'confirmation:accept'},
                     {'option_id':'reject','symbol':reject,'semantic_ref':'confirmation:reject'}]
        index=self._request_count+1; token=self.context['session_id']+'-'+str(index)
        q=dict(self.context,schema_version='bci-vla/1.0',event_id='request-'+token,host_id=self.config['request_host'],sequence=index,
            kind='REQUEST',mode=mode,interaction_id='interaction-'+token,response_to=None,
            query_id=None if mode=='CHANNEL_PROBE' else 'query-'+token,probe_id='probe-'+token if mode=='CHANNEL_PROBE' else None,
            sent_at=now.isoformat(),expires_at=expiry.isoformat(),clock_offset_ms=0,clock_uncertainty_ms=0,
            payload={'question':question,'options':options,'probe_prompt':prompt,'contract':contract})
        request=self.adapter.register_request(q,now=now,monotonic_now=monotonic_now)
        self._request_count=index; self._revoke()
        if mode=='CONFIRM':self._confirmation_contexts.add(key)
        self.state={'TASK_QUERY':'CLARIFYING','CHANNEL_PROBE':'PROBING','CONFIRM':'AWAITING_CONFIRMATION'}[mode]
        self.reason='awaiting_bound_response'; self._record('INTERACTION_REQUESTED',{'request':request.to_dict(),'monotonic_now':monotonic_now},now)
        return request

    def ingest(self,value,*,now,monotonic_now):
        now=self._now(now)
        if self.stop_latched and Event.parse(value).data['mode'] not in {'STOP','CANCEL'}:
            from ..adapters.bci.replay import IngestResult
            result=IngestResult(False,'session_stop_latched')
        else:result=self.adapter.ingest(value,now=now,monotonic_now=monotonic_now)
        event={'accepted':result.accepted,'reason':result.reason,'monotonic_now':monotonic_now,
               'request':None if result.request is None else result.request.to_dict(),
               'response':None if result.response is None else result.response.to_dict()}
        if not result.accepted:
            self._record('INPUT_REJECTED',event,now); return result
        r=None if result.response is None else result.response.data
        if r is not None and r['mode'] in {'STOP','CANCEL'}:
            self._revoke()
            self._stop('stop_requested' if r['mode']=='STOP' else 'cancel_requested',now)
            self._record('CONTROL_RECEIVED',event,now); return result
        q=result.request.data
        binding=self.binding if q['mode']=='TASK_QUERY' else None
        update=self.filter.update(result,binding=binding,
            compliance_id=self.config['probe_compliance_id'] if q['mode']=='CHANNEL_PROBE' else None,
            assumption='state_constant_within_window',finalized=True)
        event['filter_update']=update
        self._revoke()
        if not update['updated'] or r['payload']['status']!='OBSERVATION':
            self.state='PAUSED'; self.reason=update['reason'] if not update['updated'] else 'nonresponse_requires_pause'
        elif q['mode']=='CONFIRM':
            require(self.contract is not None and thaw(q['payload']['contract'])==thaw(self.contract.data['proposal']),'confirmation_proposal_mismatch')
            accepted=[o['symbol'] for o in q['payload']['options'] if o['semantic_ref']=='confirmation:accept']
            if len(accepted)==1 and r['payload']['symbol']==accepted[0]:
                self.confirmation={'request':result.request.to_dict(),'response':result.response.to_dict(),
                                   'snapshot':update['snapshot']}
                self.state='AWAITING_AUTHORIZATION'; self.reason='noisy_accept_requires_separate_authorization'
            else:self.state='PAUSED'; self.reason='confirmation_rejected_no_reasking'
        else:self.state='COLLECTING'; self.reason='evidence_updated_no_execution_authority'
        self._record('INPUT_RECEIVED',event,now)
        return result

    def authorize(self,*,now):
        now=self._now(now)
        require(not self.stop_latched and self.state=='AWAITING_AUTHORIZATION','actual_confirmation_not_pending_authorization')
        require(self.adapter.pending_request is None and self.confirmation is not None,'actual_confirmation_evidence_required')
        p=self._current_contract(now); proof=self.confirmation
        require(now<event_clock(proof['response'],'expires_at')[0],'confirmation_evidence_expired')
        require((now-at(proof['snapshot']['as_of_reference_time'])).total_seconds()<=self.config['max_confirmation_age_s'],'confirmation_too_old')
        require(self.filter.snapshot()==proof['snapshot'] and proof['snapshot']['point_inference_ready'],'evidence_changed_after_confirmation')
        refs=self.model_refs
        require(refs['backend_spec_sha256']==digest(thaw(self.backend.spec.data)),'backend_spec_changed')
        event_id=proof['response']['event_id']; require(event_id not in self._consumed_evidence,'confirmation_already_consumed')
        authorization_view,gap=self._prediction_copy(proof['snapshot'],now)
        probability=authorization_view['goal_marginals'][self.contract.data['goal_id']]
        require(probability>=self.config['minimum_goal_probability'],'confirmation_posterior_below_declared_policy')
        record=AuthorizationRecord({'authorization_id':'auth-'+event_id,'scope':'offline_fake_only',
            'contract_sha256':digest(self.contract.to_dict()),'contract_id':p['contract_id'],'task_id':p['task_id'],'task_revision':p['revision'],
            'scene_id':p['scene_id'],'scene_revision':p['scene_revision'],'request':proof['request'],'response':proof['response'],
            'model_refs':dict(refs),'issued_at':now.isoformat(),'valid_until':min(timestamp(p['valid_until']),event_clock(proof['response'],'expires_at')[0]).isoformat(),
            'belief_at_confirmation':proof['snapshot']['belief'],'belief_at_authorization':authorization_view['belief'],
            'authorization_prediction_gap_s':gap,'goal_probability':probability,'confirmation_is_noisy':True,
            'frozen_until_dispatch':True,'consume_once':True,'physical_execution_allowed':False})
        self.authorization=record; self.authorization_consumed=False; self._consumed_evidence.add(event_id)
        self.state='AUTHORIZED'; self.reason='offline_fake_authorization_only'
        self._record('AUTHORIZATION_GRANTED',record.to_dict(),now); return record

    def dispatch(self,submission_id,observation,*,now,execution_mode='offline'):
        now=self._now(now); require(execution_mode=='offline','live_execution_unverified'); text(submission_id)
        fingerprint=digest({'submission_id':submission_id,'observation':observation})
        if self.submission is not None:
            require(self.submission['submission_id']==submission_id and self._submission_digest==fingerprint,'unresolved_or_reused_submission_no_resubmit')
            self._record('DUPLICATE_DISPATCH_IGNORED',{'submission_id':submission_id},now)
            return dict(self.submission)
        require(not self.stop_latched and self.state=='AUTHORIZED' and not self.authorization_consumed,'valid_unused_authorization_required')
        require(self.adapter.pending_request is None,'probe_or_interaction_freezes_dispatch')
        p=self._current_contract(now); require(self.authorization is not None,'authorization_required')
        auth=self.authorization.data
        require(now<timestamp(auth['valid_until']) and auth['contract_sha256']==digest(self.contract.to_dict()),'authorization_expired_or_contract_changed')
        require(thaw(auth['model_refs'])==self.model_refs,'authorization_model_changed')
        require(self.confirmation is not None and self.filter.snapshot()==self.confirmation['snapshot'],'evidence_changed_after_confirmation')
        require((now-at(self.confirmation['snapshot']['as_of_reference_time'])).total_seconds()<=self.config['max_confirmation_age_s'],'confirmation_too_old')
        dispatch_view,dispatch_gap=self._prediction_copy(self.confirmation['snapshot'],now)
        require(dispatch_view['goal_marginals'][self.contract.data['goal_id']]>=self.config['minimum_goal_probability'],'dispatch_posterior_below_declared_policy')
        observation=self.backend.validate_observation(observation)
        require((observation['scene_id'],observation['scene_revision'])==(self.context['scene_id'],self.context['scene_revision']),'observation_scene_mismatch')
        age=(now-timestamp(observation['captured_at'])).total_seconds()
        require(0<=age<=self.config['max_observation_age_s'],'observation_stale_or_future')
        self.authorization_consumed=True; self.submission_contract=self.contract; self._submission_digest=fingerprint
        self.submission_at=now.isoformat()
        self.submission={'submission_id':submission_id,'contract_id':p['contract_id'],'revision':p['revision'],
                         'status':'UNKNOWN','execution_status':'UNKNOWN','actual_execution':False,'last_feedback_id':None}
        self.state='SUBMISSION_UNKNOWN'; self.reason='submission_intent_recorded_before_call'
        self._record('SUBMISSION_INTENT',{'submission_id':submission_id,'observation':observation,'observation_sha256':digest(observation),
            'dispatch_prediction_gap_s':dispatch_gap,'dispatch_goal_probability':dispatch_view['goal_marginals'][self.contract.data['goal_id']]},now)
        try:
            reply=self.backend.submit(submission_id,p['contract_id'],p['revision'],self.contract.data['instruction'],observation)
            self._validate_reply(reply,submission_id)
        except Exception:
            self.reason='submit_exception_unknown_no_retry'; self._record('SUBMISSION_UNKNOWN',{'submission_id':submission_id},now)
            return dict(self.submission)
        self.submission=reply
        self.state={'ACCEPTED':'SUBMITTED','UNKNOWN':'SUBMISSION_UNKNOWN','REJECTED':'FAILED'}[reply['status']]
        self.reason='transport_acceptance_is_not_execution' if reply['status']=='ACCEPTED' else reply['reason']
        self._record('SUBMISSION_RETURNED',reply,now); return dict(reply)

    def _validate_reply(self,reply,submission_id):
        require(isinstance(reply,dict) and reply.get('submission_id')==submission_id,'backend_submission_binding_mismatch')
        require(reply.get('actual_execution') is False,'fake_feedback_required')
        require(reply.get('status') in {'ACCEPTED','UNKNOWN','REJECTED'} and
                reply.get('execution_status') in {'NOT_STARTED','UNKNOWN','STARTED','COMPLETED','FAILED','STOPPED'},'invalid_backend_status')
        require('last_feedback_id' in reply,'feedback_reference_required')
        if reply['contract_id'] is not None:
            p=self.submission_contract.data['proposal']
            require((reply['contract_id'],reply['revision'])==(p['contract_id'],p['revision']),'feedback_contract_mismatch')

    def refresh_submission(self,*,now):
        now=self._now(now); require(self.submission is not None,'no_submission')
        try:
            reply=self.backend.query(self.submission['submission_id']); self._validate_reply(reply,self.submission['submission_id'])
        except Exception:
            self.state='PAUSED' if self.stop_latched else 'SUBMISSION_UNKNOWN';self.reason='query_exception_unknown_no_resubmit'
            self._record('SUBMISSION_QUERY_FAILED',{'submission_id':self.submission['submission_id']},now)
            return dict(self.submission,status='UNKNOWN',execution_status='UNKNOWN')
        expected=self.submission_contract.data['proposal']
        if reply['contract_id'] is not None:
            require((reply['contract_id'],reply['revision'])==(expected['contract_id'],expected['revision']),'feedback_contract_mismatch')
        self.submission=reply; feedback_id=reply['last_feedback_id']
        if reply['status']=='UNKNOWN' or reply['execution_status']=='UNKNOWN':
            self.state='PAUSED' if self.stop_latched else 'SUBMISSION_UNKNOWN'; self.reason='query_unknown_no_resubmit'
            self._record('SUBMISSION_QUERY_UNKNOWN',reply,now); return dict(reply)
        if feedback_id is None or feedback_id in self._feedback_ids:
            self._record('SUBMISSION_QUERIED_NO_NEW_FEEDBACK',reply,now); return dict(reply)
        self._feedback_ids.add(feedback_id); status=reply['execution_status']
        if status=='COMPLETED':self.completion_feedback_at=now.isoformat()
        if status=='STOPPED':
            self.stop_feedback='SIMULATED_STOPPED'; self.state='CANCELLED'; self.reason='fake_stop_feedback_received'
        elif self.stop_latched:
            self.state='PAUSED'; self.reason='stop_latched_waiting_for_stop_feedback'
        elif not self._generation_valid:
            self.state='PAUSED'; self.reason='old_context_feedback_no_new_authority'
        else:
            self.state={'STARTED':'EXECUTING','COMPLETED':'VERIFYING','FAILED':'FAILED','UNKNOWN':'SUBMISSION_UNKNOWN'}[status]
            self.reason={'STARTED':'fake_execution_started','COMPLETED':'endpoint_verification_required','FAILED':'execution_failed_not_decoder_label','UNKNOWN':'execution_unknown_no_resubmit'}[status]
        self._record('SIMULATED_EXECUTION_FEEDBACK',reply,now); return dict(reply)

    def verify_endpoint(self,evidence,*,now):
        now=self._now(now); require(self.state=='VERIFYING' and not self.stop_latched,'execution_completion_required_before_verification')
        require(isinstance(evidence,EndpointEvidence),'submission_bound_endpoint_evidence_required')
        e=evidence.data; c=self.submission_contract.data; p=c['proposal']
        require((e['session_id'],e['submission_id'],e['contract_id'],e['contract_revision'],e['scene_id'],e['scene_revision'])==
            (self.context['session_id'],self.submission['submission_id'],p['contract_id'],p['revision'],p['scene_id'],p['scene_revision']),
            'endpoint_submission_context_mismatch')
        require(e['observed_at']['host_id'] in self.config['endpoint_hosts'],'unexpected_endpoint_source')
        low,high=interval(thaw(e['observed_at']))
        require(self.completion_feedback_at is not None and low>=at(self.completion_feedback_at) and high<=now,'endpoint_observation_outside_current_completion')
        trajectory=ObservedTrajectory(thaw(e['trajectory']))
        v=trajectory.data; c=self.submission_contract.data; p=c['proposal']
        require(v['trajectory_id'] not in self._trajectory_ids,'duplicate_endpoint_evidence')
        require(v['provenance']=='simulation_fixture' and v['executor_id']==c['backend_id'] and v['checkpoint_ref'] is None,'fake_independent_endpoint_source_required')
        require(v['predicate_version']==c['completion']['predicate_version'],'endpoint_predicate_version_mismatch')
        require(v['command_dispatched'] and thaw(v['commanded_intent'])==thaw(p['intent']),'trajectory_command_mismatch')
        result=evaluate_trajectory(trajectory,[{'goal_id':c['goal_id'],'kind':'task','intent':thaw(p['intent'])}])[c['goal_id']]
        match=result['endpoint_matches_goal']; self._trajectory_ids.add(v['trajectory_id'])
        self.state='SUCCEEDED' if match is True else 'FAILED' if match is False else 'PAUSED'
        self.reason='independent_endpoint_matches' if match is True else 'independent_endpoint_fails' if match is False else 'endpoint_unknown'
        self._record('ENDPOINT_JUDGED',{'evidence':evidence.to_dict(),'result':result},now); return result

    def _stop(self,reason,now):
        self.stop_latched=True; self._revoke(); self.state='PAUSED'; self.reason=reason; self.stop_feedback='REQUESTED'
        if self.submission is not None:
            try:self.backend.stop(self.submission['submission_id'])
            except Exception:self.reason='stop_request_delivery_unknown'
        elif reason=='cancel_requested':self.state='CANCELLED'
        self._record('STOP_REQUESTED',{'reason':reason,'delivery_reason':self.reason,'physical_stop_verified':False},now)

    def request_stop(self,*,now):
        now=self._now(now); self._stop('stop_requested',now)

    def cancel(self,*,now):
        now=self._now(now); self._stop('cancel_requested',now)

    def expire(self,*,now,monotonic_now):
        now=self._now(now); timeout=self.adapter.expire_pending(monotonic_now=monotonic_now)
        expired=self.contract is not None and now>=timestamp(self.contract.data['proposal']['valid_until'])
        if timeout.accepted or expired:
            self._revoke()
            if self.state in ACTIVE:self._stop('deadline_expired_stop_requested',now)
            else:self.state='EXPIRED' if expired else 'PAUSED'; self.reason='contract_expired' if expired else 'interaction_timeout'
            self._record('EXPIRED_OR_TIMED_OUT',{'expired':expired,'interaction_timeout':timeout.accepted,'monotonic_now':monotonic_now},now)
        return self.snapshot()

    def revise_context(self,new_context,new_scene,*,now):
        now=self._now(now); require(isinstance(new_scene,GroundedScene),'validated_scene_required')
        fresh=dict(context(new_context)); require(fresh['session_id']==self.context['session_id'],'new_session_required')
        require((fresh['scene_id'],fresh['scene_revision'])==(new_scene.data['scene_id'],new_scene.data['revision']),'scene_context_mismatch')
        require(new_scene.data['identity']['session_id']==fresh['session_id'],'scene_identity_session_mismatch')
        if new_scene.data['scene_id']==self.scene.data['scene_id']:
            if new_scene.data['revision']==self.scene.data['revision']:
                require(new_scene.to_dict()==self.scene.to_dict(),'scene_version_reused')
            else:
                ledger=IdentityLedger(); ledger.observe(self.scene); ledger.observe(new_scene)
        self.adapter.update_context(fresh)
        if self.state in ACTIVE:self._stop('context_changed_stop_requested',now)
        self.context=fresh; self.scene=new_scene; self._revoke(); self._generation_valid=False
        self.state='PAUSED'; self.reason='context_changed_explicit_filter_policy_rebuild_required'
        self._record('CONTEXT_INVALIDATED',{'context':fresh,'scene':new_scene.to_dict()},now)

    def invalidate_models(self,*,now):
        now=self._now(now)
        if self.state in ACTIVE:self._stop('models_changed_stop_requested',now)
        self._revoke(); self._generation_valid=False; self.state='PAUSED'; self.reason='models_require_explicit_rebuild'
        self._record('MODELS_INVALIDATED',{},now)

    def request_policy(self,*,remaining_s,scene_valid_until_s,now,depth=1):
        now=self._now(now); require(not self.stop_latched,'stop_latched')
        require(self._generation_valid,'models_or_context_require_explicit_rebuild')
        # Predict an explicitly logged planning COPY through receive/compute delay.
        # The evidence filter remains at its last window end, avoiding double T.
        snap,gap=self._prediction_copy(self.filter.snapshot(),now)
        contracts={} if self.contract is None else {self.contract.data['goal_id']:thaw(self.contract.data['proposal'])}
        state=PlanningState.from_filter_snapshot(snap,scene_id=self.context['scene_id'],scene_revision=self.context['scene_revision'],
            task_id=self.context['task_id'],task_revision=self.context['task_revision'],now_s=now.timestamp(),
            used_s=(now-self._started_at).total_seconds(),remaining_s=remaining_s,scene_valid_until_s=scene_valid_until_s,contracts=contracts,
            pending=self.adapter.pending_request is not None)
        result=self.policy.plan(state,depth=depth)
        self._record('POLICY_RETURNED',{'planning_prediction_gap_s':gap,'decision':result,
            'inputs':{'remaining_s':remaining_s,'scene_valid_until_s':scene_valid_until_s,'depth':depth,'used_s':(now-self._started_at).total_seconds()}},now); return result


def replay_session_state(journal):
    from .replay import replay_session_state as event_replay
    return event_replay(journal)
