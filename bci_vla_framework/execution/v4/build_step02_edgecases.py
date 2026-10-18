"""Extend the contract fixture catalogue; never run models or create training data."""
from copy import deepcopy
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from bci_vla.interaction_contract_v4 import load_json


def main():
    folder = ROOT/"examples/v4"
    def write(name,value):
        (folder/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    public = load_json(folder/"public_input.json")
    expired = deepcopy(public)
    expired["history"][0].update(ended_at_s=7.5,valid=False,invalid_reason="EXPIRED")
    expired["selection_event_id"] = None
    write("expired_task_evidence.json",expired)
    rejected = deepcopy(load_json(folder/"confirmed_task_survives_nine_probes.json"))
    reject = deepcopy(rejected["public_state"]["confirmed_task"]["confirmation_receipt"])
    reject.update(event_id="e-reject",sequence=99,role="REJECT",original_role="CONFIRM",started_at_s=11.1,ended_at_s=11.5)
    reject["question"].update(question_id="q-reject",issued_at_s=11,expires_at_s=12)
    reject["evidence"]["selected_option_id"] = next(o["option_id"] for o in reject["question"]["options"] if o["reference"]=="reject")
    rejected.update(history=[reject],current_event_id="e-reject",selection_event_id=None)
    rejected["public_state"]["confirmed_task"] = None
    write("reject_clears_task.json",rejected)
    normal = load_json(folder/"module_output.json")
    extra = []
    for name,reason,outside,probabilities,allowed,missing in [
        ("target_outside_candidates_output.json","target_outside_candidates",1,[0,0,0],[],[]),
        ("missing_action_slot_output.json","missing_task_slot",0,[1,0,0],["cup-blue"],["ACTION"]),
        ("abstain_without_missing_slot_output.json","ambiguous_target",0,[.5,.5,0],["cup-blue","cup-red"],[]),
    ]:
        value=deepcopy(normal)
        value["interaction"]["reason_code"] = reason
        value["intent"].update(outside_probability=outside,allowed_candidate_ids=allowed,missing_slots=missing,abstain=True)
        for row,probability in zip(value["intent"]["candidate_distribution"],probabilities):row["probability"]=probability
        write(name,value); extra.append(name)
    index=load_json(folder/"fixture_index.json")
    for name in ["expired_task_evidence.json","reject_clears_task.json"]:
        if name not in index["valid_public_inputs"]:index["valid_public_inputs"].append(name)
    index["additional_module_outputs"] = extra
    write("fixture_index.json",index)
    print("Extended catalogue: 11 public inputs, 2 private labels, 4 outputs; all synthetic.")


if __name__=="__main__":main()
