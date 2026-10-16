"""Goal-conditional independent endpoint evidence and explicit offline cost tables."""
from dataclasses import dataclass
import math

from ..protocols import fields, freeze, thaw, integer, json_object, number, require, text, validate_intent
from ..scene._legacy.planning import candidate_key

COMPONENTS={"wrong_start","task_failure","recovery_s","user_time_s"}
PREDICATES={"holding","inside","folded"}


def validate_goal(goal):
    fields(goal,{"goal_id","kind","intent"}); text(goal["goal_id"])
    require(goal["kind"] in {"task","none","unknown"},"invalid_goal_kind")
    if goal["kind"]=="task": validate_intent(goal["intent"])
    else: require(goal["intent"] is None,"non_task_has_intent")


@dataclass(frozen=True)
class ObservedTrajectory:
    data: object

    def __post_init__(self):
        v=json_object(self.data)
        fields(v,{"trajectory_id","executor_id","checkpoint_ref","predicate_version","provenance","split","source_ref","command_dispatched","commanded_intent","final_predicates","user_time_s","recovery_by_goal"})
        for k in ["trajectory_id","executor_id","predicate_version","source_ref"]: text(v[k])
        text(v["checkpoint_ref"],nullable=True)
        require(v["provenance"] in {"simulation_fixture","independent_annotation"},"independent_endpoint_source_required")
        require(v["provenance"] != "independent_annotation" or v["checkpoint_ref"] is not None,"real_trajectory_checkpoint_required")
        require(v["split"] in {"calibration","calibration_holdout","test"},"invalid_trajectory_split")
        require(type(v["command_dispatched"]) is bool,"invalid_dispatch_flag")
        if v["command_dispatched"]: validate_intent(v["commanded_intent"])
        else: require(v["commanded_intent"] is None,"undispatched_intent")
        require(isinstance(v["final_predicates"],list),"invalid_predicates")
        seen=set()
        for p in v["final_predicates"]:
            fields(p,{"name","subject_id","object_id","value","evidence_ref"})
            require(p["name"] in PREDICATES,"unsupported_endpoint_predicate")
            text(p["subject_id"]); text(p["object_id"],nullable=True); text(p["evidence_ref"])
            require(p["value"] is None or type(p["value"]) is bool,"invalid_predicate_value")
            require((p["name"]=="inside")== (p["object_id"] is not None),"invalid_predicate_arguments")
            key=(p["name"],p["subject_id"],p["object_id"])
            require(key not in seen,"duplicate_predicate"); seen.add(key)
        if v["user_time_s"] is not None: number(v["user_time_s"],0)
        require(isinstance(v["recovery_by_goal"],dict),"invalid_recovery")
        for g,item in v["recovery_by_goal"].items():
            text(g); fields(item,{"seconds","evidence_ref"}); number(item["seconds"],0); text(item["evidence_ref"])
        object.__setattr__(self,"data",freeze(v))


def evaluate_trajectory(trajectory,goals):
    """Evaluate one observed final state against many hypotheses, never label intent."""
    require(isinstance(trajectory,ObservedTrajectory),"validated_trajectory_required")
    require(isinstance(goals,list),"invalid_goals")
    ids=set(); result={}; v=trajectory.data
    for goal in goals:
        validate_goal(goal); g=goal["goal_id"]
        require(g not in ids,"duplicate_goal"); ids.add(g)
        match=None; wrong=None
        if goal["kind"]=="none":
            match=not v["command_dispatched"]; wrong=int(v["command_dispatched"])
        elif goal["kind"]=="task":
            i=goal["intent"]
            wrong=int(candidate_key(thaw(v["commanded_intent"])) != candidate_key(i)) if v["command_dispatched"] else 0
            key=({"pick":"holding","place":"inside","fold":"folded"}[i["action"]],i["source_id"],i["destination_id"])
            p=next((p for p in v["final_predicates"] if (p["name"],p["subject_id"],p["object_id"])==key),None)
            match=p["value"] if p is not None else None
        # Do not transfer one goal's observed recovery to other counterfactual goals.
        recovery=v["recovery_by_goal"].get(g)
        result[g]={"endpoint_matches_goal":match,"components":{"wrong_start":wrong,"task_failure":None if match is None else int(not match),"recovery_s":None if recovery is None else recovery["seconds"],"user_time_s":v["user_time_s"]},"provenance":v["provenance"],"source_ref":v["source_ref"]}
    return result


def estimate_components(trajectories,goal,*,decision_intent,executor_id,checkpoint_ref,predicate_version):
    """Calibration-only observational means, with per-component denominators."""
    validate_goal(goal); text(executor_id); text(checkpoint_ref,nullable=True); text(predicate_version)
    if decision_intent is not None: validate_intent(decision_intent)
    seen=set(); rows=[]; sources=[]; simulation=False
    for trajectory in trajectories:
        v=trajectory.data
        require(v["split"]=="calibration","test_trajectory_cannot_calibrate")
        require((v["executor_id"],v["checkpoint_ref"],v["predicate_version"]) == (executor_id,checkpoint_ref,predicate_version),"backend_calibration_mismatch")
        require(v["command_dispatched"] == (decision_intent is not None),"action_calibration_mismatch")
        if decision_intent is not None:
            require(candidate_key(thaw(v["commanded_intent"])) == candidate_key(decision_intent),"action_calibration_mismatch")
        require(v["trajectory_id"] not in seen,"duplicate_trajectory"); seen.add(v["trajectory_id"])
        rows.append(evaluate_trajectory(trajectory,[goal])[goal["goal_id"]]["components"])
        sources.append(v["source_ref"]); simulation|=v["provenance"]=="simulation_fixture"
    means={}; counts={}
    for component in COMPONENTS:
        values=[row[component] for row in rows if row[component] is not None]
        means[component]=math.fsum(values)/len(values) if values else None; counts[component]=len(values)
    return {"status":"unavailable" if not rows else ("simulation_only" if simulation else "calibration_estimate_pending_review"),"conditioned_intent":decision_intent,"trajectory_count":len(rows),"means":means,"component_counts":counts,"source_refs":sources,"uncertainty":None,"deployment_allowed":False}


class OutcomeTable:
    """Hypothetical per-decision components; no optimizer, model confidence or fallback."""
    def __init__(self,config):
        v=json_object(config)
        require(isinstance(v,dict) and v.get("status")!="uncalibrated","real_consequence_calibration_unavailable")
        fields(v,{"schema_version","model_id","executor_id","checkpoint_ref","predicate_version","status","provenance","deployment_allowed","goal_ids","channel_mode_ids","rows"})
        require(v["schema_version"]=="outcome-table/1.0","unsupported_outcome_schema")
        require(v["status"]=="simulation_only" and v["provenance"]=="illustrative_not_fitted" and v["deployment_allowed"] is False,"real_consequence_calibration_unavailable")
        for k in ["model_id","executor_id","predicate_version"]: text(v[k])
        text(v["checkpoint_ref"],nullable=True)
        for k in ["goal_ids","channel_mode_ids"]:
            require(isinstance(v[k],list) and bool(v[k]) and len(v[k])==len(set(v[k])),"invalid_state_ids")
            for item in v[k]: text(item)
        require(isinstance(v["rows"],list),"invalid_outcome_rows")
        self._rows={}
        for row in v["rows"]:
            fields(row,{"decision","stage","goal_id","channel_mode_id","components","source_ref","included_time_phases"})
            require(row["decision"] in {"ACCEPT_NEW","KEEP_OLD","CHECK_CHANNEL","REQUEST_RECALIBRATION","WAIT"},"unknown_decision")
            require(row["stage"] in {"terminal","continuation"},"unknown_decision_stage")
            require((row["decision"] in {"ACCEPT_NEW","KEEP_OLD"})==(row["stage"]=="terminal"),"decision_stage_mismatch")
            require(row["goal_id"] in v["goal_ids"] and row["channel_mode_id"] in v["channel_mode_ids"],"unknown_outcome_state")
            fields(row["components"],COMPONENTS); text(row["source_ref"])
            for k,n in row["components"].items():
                if n is not None: number(n,0,1 if k in {"wrong_start","task_failure"} else None)
            require(isinstance(row["included_time_phases"],list) and all(isinstance(p,str) for p in row["included_time_phases"]),"invalid_time_scope")
            if row["stage"]=="continuation":
                require(row["components"]["task_failure"] is None and row["components"]["recovery_s"] is None,"continuation_future_outcome_unknown")
            key=(row["decision"],row["goal_id"],row["channel_mode_id"])
            require(key not in self._rows,"duplicate_outcome_row")
            self._rows[key]=freeze(row)
        self.data=freeze(v)

    def get(self,decision,goal_id,channel_mode_id,*,execution_mode="offline"):
        require(execution_mode=="offline","simulation_consequence_cannot_run_live")
        require((decision,goal_id,channel_mode_id) in self._rows,"unknown_outcome_state")
        return thaw(self._rows[(decision,goal_id,channel_mode_id)])

    def score(self,decision,goal_id,channel_mode_id,weights,*,execution_mode="offline"):
        fields(weights,COMPONENTS)
        for n in weights.values(): number(n,0)
        row=self.get(decision,goal_id,channel_mode_id,execution_mode=execution_mode)
        c=row["components"]
        row["scalar_total"]=math.fsum(weights[k]*c[k] for k in COMPONENTS) if row["stage"]=="terminal" and all(c[k] is not None for k in COMPONENTS) else None
        row["known_immediate_user_cost"]=None if c["user_time_s"] is None else weights["user_time_s"]*c["user_time_s"]
        row["warning"]="Continuation requires downstream value; not comparable to a terminal total. Do not charge included time phases again."
        return row

    def sensitivity(self,decision,goal_id,channel_mode_id,weight_sets):
        require(isinstance(weight_sets,dict),"invalid_weights")
        return {name:self.score(decision,goal_id,channel_mode_id,weights)["scalar_total"] for name,weights in weight_sets.items()}
