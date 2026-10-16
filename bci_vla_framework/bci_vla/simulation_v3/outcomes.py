import math
from ..protocols import fields,freeze,thaw,json_object,require,text,number
from ..models.consequence import COMPONENTS
class OutcomeTable:
    """Hypothetical per-decision components; no optimizer, model confidence or fallback."""
    def __init__(self,config):
        v=json_object(config)
        require(isinstance(v,dict) and v.get("status")!="uncalibrated","real_consequence_calibration_unavailable")
        fields(v,{"schema_version","model_id","executor_id","checkpoint_ref","predicate_version","status","provenance","deployment_allowed","goal_ids","channel_mode_ids","rows"})
        require(v["schema_version"]=="outcome-table/1.0","unsupported_outcome_schema")
        require(v["status"]=="simulation_only" and v["provenance"]=="empirical_simulator_calibration" and v["deployment_allowed"] is False,"real_consequence_calibration_unavailable")
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
