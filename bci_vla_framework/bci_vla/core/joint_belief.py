"""A correlated finite-state Bayes filter with fail-closed evidence accounting."""
from __future__ import annotations

from datetime import datetime, timezone
import math

from ..protocols import JointBelief, ProtocolError, fields, freeze, thaw, json_object, number, require, text, event_clock, interval, timestamp
from ..adapters.bci.replay import IngestResult, CONTEXT_FIELDS, BIND_FIELDS, context
from ..models.evidence import FinalPrefixLedger
from ..scene.tasks import goal_id
from ..scene._legacy.planning import candidate_key

UNKNOWN_CHANNEL = "unknown_channel"


class GoalVocabulary:
    """Explicit scene-bound identities; old/new/index aliases are not inferred."""
    def __init__(self, config):
        v=json_object(config)
        fields(v,{"schema_version","scene_id","scene_revision","goals"})
        require(v["schema_version"]=="goal-vocabulary/1.0","unsupported_goal_schema")
        text(v["scene_id"])
        from ..protocols import integer
        from ..models.consequence import validate_goal
        integer(v["scene_revision"])
        require(isinstance(v["goals"],list),"invalid_goals")
        ids=set(); known=[]; specials=set()
        for g in v["goals"]:
            validate_goal(g); require(g["goal_id"] not in ids,"duplicate_goal");ids.add(g["goal_id"])
            if g["kind"]=="task":
                require(g["goal_id"]==goal_id(g["intent"]),"goal_alias_not_canonical")
                known.append(g["goal_id"])
            else:
                require(g["goal_id"]==g["kind"],"reserved_goal_id");specials.add(g["kind"])
        require(len(known)==2 and specials=={"unknown","none"},"binary_vocabulary_with_reserved_states_required")
        self.data=freeze(v);self.known_goal_ids=tuple(known)
        self.goal_ids=tuple(g["goal_id"] for g in v["goals"])
        self._goals={g["goal_id"]:freeze(g) for g in v["goals"]}

    def confirmation_symbols(self,request):
        contract=request["payload"]["contract"]
        symbols={o["semantic_ref"]:o["symbol"] for o in request["payload"]["options"]}
        require(set(symbols)=={"confirmation:accept","confirmation:reject"},"confirmation_mapping_ambiguous")
        matching=[g for g in self.known_goal_ids if candidate_key(thaw(self._goals[g]["intent"]))==candidate_key(thaw(contract["intent"]))]
        require(len(matching)==1,"contract_outside_goal_vocabulary")
        return {g:symbols["confirmation:accept" if g==matching[0] else "confirmation:reject"] for g in self.known_goal_ids}


class TaskQuestionBinding:
    def __init__(self,config,vocabulary):
        v=json_object(config)
        fields(v,{"question","mapping_id","stimulus_version","options"})
        for k in ["question","mapping_id","stimulus_version"]:text(v[k])
        require(isinstance(v["options"],list) and len(v["options"])==2,"binary_question_required")
        ids=set();symbols=set();goals=set();mapping={}
        for option in v["options"]:
            fields(option,{"option_id","symbol","semantic_ref"})
            for value in option.values():text(value)
            prefix="goal:"
            require(option["semantic_ref"].startswith(prefix),"explicit_goal_reference_required")
            g=option["semantic_ref"][len(prefix):]
            require(g in vocabulary.known_goal_ids and g not in goals,"question_goal_mismatch")
            require(option["symbol"] not in symbols and option["option_id"] not in ids,"duplicate_option")
            ids.add(option["option_id"]);symbols.add(option["symbol"]);goals.add(g);mapping[g]=option["symbol"]
        self.data=freeze(v);self.goal_symbols=freeze(mapping)

    def validate_request(self,q):
        require(q["mode"]=="TASK_QUERY","not_task_query")
        require(q["mapping_id"]==self.data["mapping_id"] and q["stimulus_version"]==self.data["stimulus_version"],"question_mapping_mismatch")
        require(q["payload"]["question"]==self.data["question"],"question_text_mismatch")
        actual={o["option_id"]:dict(o) for o in q["payload"]["options"]}
        expected={o["option_id"]:dict(o) for o in self.data["options"]}
        require(actual==expected,"question_option_mismatch")


class JointBeliefFilter:
    """Offline, two modeled task goals plus reserved unresolved states.

    Unknown/none/channel-unknown likelihoods are NOT invented. Positive mass on
    such states pauses point inference until an explicit calibrated model exists.
    The model cannot establish that real observations are within its model class.
    """
    def __init__(self,initial,*,vocabulary,channel_model,transition_model,current_context,layout_id,initial_at,outlier_weight=0.0):
        from ..models.channel import ChannelModel, JointTransitionModel
        require(isinstance(initial,JointBelief),"validated_joint_belief_required")
        require(isinstance(vocabulary,GoalVocabulary) and isinstance(channel_model,ChannelModel) and isinstance(transition_model,JointTransitionModel),"validated_models_required")
        self.vocabulary=vocabulary;self.channel_model=channel_model;self.transition_model=transition_model
        self._context=context(current_context);text(layout_id);self.layout_id=layout_id
        require(self._context["scene_id"]==vocabulary.data["scene_id"] and self._context["scene_revision"]==vocabulary.data["scene_revision"],"vocabulary_scene_mismatch")
        require(self._context["channel_profile_id"]==channel_model.data["profile_id"],"profile_mismatch")
        require(tuple(transition_model.data["goal_ids"])==vocabulary.known_goal_ids,"transition_goal_ids_mismatch")
        require(tuple(transition_model.data["mode_ids"])==tuple(channel_model.data["mode_ids"]),"transition_mode_ids_mismatch")
        self.goal_ids=vocabulary.goal_ids;self.mode_ids=tuple(channel_model.data["mode_ids"])+(UNKNOWN_CHANNEL,)
        require(UNKNOWN_CHANNEL not in channel_model.data["mode_ids"],"reserved_channel_id")
        require(tuple(initial.data["goal_ids"])==self.goal_ids and tuple(initial.data["mode_ids"])==self.mode_ids,"belief_axes_mismatch")
        require(isinstance(initial_at,datetime) and initial_at.tzinfo is not None,"timezone_required")
        number(outlier_weight,0,1)
        self.outlier_weight=outlier_weight
        self._mass={(g,z):0.0 for g in self.goal_ids for z in self.mode_ids}
        for row in initial.data["mass"]:self._mass[(row["goal_id"],row["mode_id"])]=row["probability"]
        self._last_at=initial_at.astimezone(timezone.utc);self._as_of=initial.data["as_of_event_id"]
        self._ledger=FinalPrefixLedger();self._events=set();self._request_ids=set();self._interactions=set();self._last_sequence={}
        self._audit=[];self._status="ready";self._reason=None

    @property
    def audit_log(self):return tuple(self._audit)

    def _view(self,mass,as_of):
        return {"goal_ids":list(self.goal_ids),"mode_ids":list(self.mode_ids),"mass":[{"goal_id":g,"mode_id":z,"probability":mass[(g,z)]} for g in self.goal_ids for z in self.mode_ids],"as_of_event_id":as_of}

    def snapshot(self):
        unmodeled=math.fsum(p for (g,z),p in self._mass.items() if g not in self.vocabulary.known_goal_ids or z==UNKNOWN_CHANNEL)
        return {"status":self._status,"reason":self._reason,"model_validity":"conditional_on_finite_simulation_models","unmodeled_support_mass":unmodeled,"point_inference_ready":self._status=="ready" and unmodeled==0,"execution_allowed":False,"as_of_reference_time":self._last_at.isoformat(),"belief":self._view(self._mass,self._as_of),"goal_marginals":{g:math.fsum(self._mass[(g,z)] for z in self.mode_ids) for g in self.goal_ids},"channel_marginals":{z:math.fsum(self._mass[(g,z)] for g in self.goal_ids) for z in self.mode_ids}}

    def _predict(self,elapsed_s):
        matrix=self.transition_model.matrix(elapsed_s)
        states=[(g,z) for g in self.vocabulary.known_goal_ids for z in self.channel_model.data["mode_ids"]]
        predicted=dict(self._mass)
        for j,state in enumerate(states):predicted[state]=math.fsum(self._mass[old]*matrix[i][j] for i,old in enumerate(states))
        # Reserved states are not silently mapped into supported goals/channels.
        return predicted

    def update(self,result,*,binding=None,compliance_id=None,assumption=None,finalized=False,nonresponse_window=None):
        before=self.snapshot();predicted=None;event_id=None;q=None;response=None;likelihoods=None;gap=None;duration=None
        if self._status!="ready":
            if isinstance(result,IngestResult) and result.accepted and result.response is not None and result.response.data["mode"]=="STOP" and result.response.data["session_id"]==self._context["session_id"]:
                self._status="stopped";self._reason="stop"
                self._audit.append(freeze({"before":before,"after":self.snapshot(),"evidence_id":result.response.data["event_id"],"mode":"STOP","reason":"stop","updated":False}))
                return {"updated":False,"reason":"stop","snapshot":self.snapshot()}
            return {"updated":False,"reason":"filter_"+self._status,"snapshot":self.snapshot()}
        try:
            require(isinstance(result,IngestResult) and result.accepted and result.response is not None,"accepted_result_required")
            response=result.response.data;event_id=response["event_id"]
            require(response["session_id"]==self._context["session_id"],"wrong_session")
            if response["mode"] in {"STOP","CANCEL"}:
                self._status="stopped" if response["mode"]=="STOP" else "paused";self._reason=response["mode"].lower()
                self._audit.append(freeze({"before":before,"after":self.snapshot(),"evidence_id":event_id,"mode":response["mode"],"reason":self._reason,"updated":False}))
                return {"updated":False,"reason":self._reason,"snapshot":self.snapshot()}
            require(result.request is not None,"bound_request_required");q=result.request.data
            require(all(q[k]==self._context[k] and response[k]==self._context[k] for k in CONTEXT_FIELDS),"stale_belief_context")
            require(q["kind"]=="REQUEST" and response["kind"]=="RESPONSE" and response["response_to"]==q["event_id"] and all(q[k]==response[k] for k in BIND_FIELDS),"response_binding_mismatch")
            require(event_id not in self._events and q["event_id"] not in self._request_ids and q["interaction_id"] not in self._interactions,"duplicate_belief_evidence")
            require(response["sequence"]>self._last_sequence.get(response["host_id"],-1),"out_of_order_belief_evidence")
            require(finalized is True,"intermediate_prefix_not_evidence")
            p=response["payload"];status=p["status"]
            require(status in {"OBSERVATION","NO_RESPONSE"},"status_requires_pause")
            require(assumption=="state_constant_within_window","window_stability_not_established")
            if status=="OBSERVATION":
                w=p["sample_window"];require(w is not None,"missing_window")
                require(w["source_kind"]=="offline_replay","offline_evidence_required")
                start,end=[interval({"host_id":response["host_id"],"at":w[k],"offset_ms":response["clock_offset_ms"],"uncertainty_ms":response["clock_uncertainty_ms"]}) for k in ["started_at","ended_at"]]
                require(response["clock_uncertainty_ms"]==0,"filter_requires_exact_reference_window")
                start,end=start[0],end[1]
                duration=(w["sample_end"]-w["sample_start"])/w["sample_rate_hz"]
            else:
                fields(nonresponse_window,{"start","end","clock_domain"})
                require(nonresponse_window["clock_domain"]=="reference","reference_window_required")
                start,end=timestamp(nonresponse_window["start"]),timestamp(nonresponse_window["end"])
                duration=(end-start).total_seconds();number(duration,0)
            require(start>=self._last_at and end>start,"evidence_time_regression")
            require(start>=event_clock(q)[1] and end<=event_clock(response)[0] and end<event_clock(q,"expires_at")[0],"window_outside_interaction")
            if q["mode"]=="CHANNEL_PROBE":
                prompt=q["payload"]["probe_prompt"]
                require(prompt["presentation_ack"] and prompt["presented_at"] is not None,"probe_not_presented")
                require(start>=interval(dict(prompt["presented_at"]))[1],"window_before_probe_prompt")
                require(binding is None,"probe_must_not_use_task_binding")
                symbols={g:prompt["symbol"] for g in self.vocabulary.known_goal_ids}
            elif q["mode"]=="CONFIRM":
                require(binding is None and compliance_id is None,"confirmation_binding_ambiguous")
                c=q["payload"]["contract"]
                require(c["task_id"]==q["task_id"] and c["revision"]==q["task_revision"] and c["scene_id"]==q["scene_id"] and c["scene_revision"]==q["scene_revision"],"contract_version_mismatch")
                require(end<interval({"host_id":q["host_id"],"at":c["valid_until"],"offset_ms":q["clock_offset_ms"],"uncertainty_ms":q["clock_uncertainty_ms"]})[0],"confirmation_contract_expired")
                symbols=self.vocabulary.confirmation_symbols(q)
            else:
                require(q["mode"]=="TASK_QUERY" and isinstance(binding,TaskQuestionBinding),"explicit_task_binding_required")
                require(set(binding.goal_symbols)==set(self.vocabulary.known_goal_ids),"question_goal_mismatch")
                require(compliance_id is None,"compliance_only_for_probe");binding.validate_request(q);symbols=binding.goal_symbols
            gap=(start-self._last_at).total_seconds();predicted=self._predict(gap)
            for (g,z),mass in predicted.items():
                require(mass==0 or (g in self.vocabulary.known_goal_ids and z!=UNKNOWN_CHANNEL),"unmodeled_state_support_requires_pause")
            likelihoods={}
            for g in self.vocabulary.known_goal_ids:
                values=self.channel_model.accepted_result_likelihoods(result,intended_symbol=symbols[g],layout_id=self.layout_id,assumption=assumption,compliance_id=compliance_id,nonresponse_duration_s=duration if status=="NO_RESPONSE" else None)
                for z,value in values.items():
                    likelihoods[(g,z)]=(1-self.outlier_weight)*value+self.outlier_weight/(len(self.channel_model.data["scope"]["symbols"])+1)
            logs={state:math.log(mass)+math.log(likelihoods[state]) for state,mass in predicted.items() if mass>0 and likelihoods.get(state,0)>0}
            require(bool(logs),"zero_evidence_probability")
            scale=max(logs.values());scaled={state:math.exp(value-scale) for state,value in logs.items()}
            require(all(v>0 and math.isfinite(v) for v in scaled.values()),"posterior_underflow_or_nonfinite")
            total=math.fsum(scaled.values());number(total,0);require(total>0,"zero_evidence_probability")
            posterior={state:scaled.get(state,0)/total for state in self._mass}
            require(all(posterior[state]>0 for state in scaled),"posterior_underflow_or_nonfinite")
            JointBelief(self._view(posterior,event_id))
            if status=="OBSERVATION":self._ledger.claim(result,finalized=True)
            self._mass=posterior;self._last_at=end;self._as_of=event_id
            self._events.add(event_id);self._request_ids.add(q["event_id"]);self._interactions.add(q["interaction_id"]);self._last_sequence[response["host_id"]]=response["sequence"]
            self._audit.append(freeze({"before":before,"prediction":self._view(predicted,"prediction"),"after":self.snapshot(),"evidence_id":event_id,"request_id":q["event_id"],"mode":q["mode"],"signal_status":status,"context":dict(self._context),"gap_transition_s":gap,"stable_window_s":duration,"probe_prompt_symbol":q["payload"]["probe_prompt"]["symbol"] if q["mode"]=="CHANNEL_PROBE" else None,"outlier_weight":self.outlier_weight,"likelihoods":[{"goal_id":g,"mode_id":z,"probability":v} for (g,z),v in likelihoods.items()],"reason":"joint_bayes_update","updated":True}))
            return {"updated":True,"reason":"joint_bayes_update","snapshot":self.snapshot()}
        except (ProtocolError,OverflowError,ValueError) as exc:
            reason=str(exc) if isinstance(exc,ProtocolError) else "numerical_error"
            self._status="paused";self._reason=reason
            self._audit.append(freeze({"before":before,"prediction":None if predicted is None else self._view(predicted,"uncommitted_prediction"),"after":self.snapshot(),"evidence_id":event_id,"mode":q["mode"] if q else None,"context":dict(self._context),"reason":reason,"updated":False}))
            return {"updated":False,"reason":reason,"snapshot":self.snapshot()}
