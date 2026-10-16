"""Synthetic software smoke fixtures. These are NOT real scientific evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random

from .schema import assert_disjoint_scenes, validate_case


def make_synthetic_splits(output_dir, scenes=60, seed=7):
    if scenes < 10:
        raise ValueError("use at least 10 independent synthetic scene groups")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    group_ids = list(range(scenes))
    rng.shuffle(group_ids)
    n_train, n_val = int(scenes * 0.6), max(2, int(scenes * 0.2))
    assignment = {i: "train" if j < n_train else "val" if j < n_train+n_val else "test" for j,i in enumerate(group_ids)}
    splits = {"train": [], "val": [], "test": []}
    for scene in range(scenes):
        prefix = f"s{scene:04d}"
        fruit_name = rng.choice(["grapes", "starfruit", "cherry"])
        fruit = {"id": prefix+"_fruit", "label": fruit_name, "position": [rng.uniform(.15,.85),rng.uniform(.2,.8)], "affordances": ["pickable"], "attributes": []}
        cloth = {"id": prefix+"_cloth", "label": "cloth", "position": [rng.uniform(.15,.85),rng.uniform(.2,.8)], "affordances": ["pickable", "foldable"], "attributes": []}
        left = {"id": prefix+"_bowlA", "label": "bowl", "position": [rng.uniform(.05,.3),rng.uniform(.2,.8)], "affordances": ["container"], "attributes": []}
        right = {"id": prefix+"_bowlB", "label": "bowl", "position": [rng.uniform(.7,.95),rng.uniform(.2,.8)], "affordances": ["container"], "attributes": []}
        objects = [fruit,cloth,left,right]
        rng.shuffle(objects)
        relations = []
        for first in objects:
            for second in objects:
                if first["id"] == second["id"]:
                    continue
                relations.append({"subject": first["id"], "predicate": "left_of" if first["position"][0]<second["position"][0] else "right_of", "object": second["id"]})
                distance = sum((a-b)**2 for a,b in zip(first["position"],second["position"]))
                if distance < .15:
                    relations.append({"subject":first["id"],"predicate":"near","object":second["id"]})
        base = {"objects":objects,"relations":relations,"selection":{"object_id":fruit["id"]},"context":{"goal":"unspecified"}}
        def add(name, changes, status, action=None, destination=None, slot=None, source=None):
            obs=copy.deepcopy(base)
            for key,value in changes.items():
                obs[key]=copy.deepcopy(value)
            intent=None if status!="ready" else {"action":action,"source_id":source or fruit["id"],"destination_id":destination}
            case={"case_id":prefix+"_"+name,"scene_id":prefix,"provenance":"synthetic_smoke_only",
                "observation":obs,"annotation":{"status":status,"intent":intent,"clarification_slot":slot},
                "scenario":name}
            splits[assignment[scene]].append(validate_case(case))
        add("pick",{"context":{"goal":"pick"}},"ready","pick")
        add("place_left",{"context":{"goal":"put_away","destination_hint":"leftmost"}},"ready","place",left["id"])
        add("place_right",{"context":{"goal":"put_away","destination_hint":"rightmost"}},"ready","place",right["id"])
        nearer=min([left,right],key=lambda obj:sum((a-b)**2 for a,b in zip(obj["position"],fruit["position"])))
        add("place_nearest",{"context":{"goal":"put_away","destination_hint":"nearest_source"}},"ready","place",nearer["id"])
        add("fold",{"selection":{"object_id":cloth["id"]},"context":{"goal":"fold"}},"ready","fold",source=cloth["id"])
        add("missing_action",{},"clarify",slot="action")
        add("missing_destination",{"context":{"goal":"put_away"}},"clarify",slot="destination")
        add("uncertain_bci",{"selection":{"object_id":None,"probabilities":{fruit["id"]:.52,cloth["id"]:.48}},"context":{"goal":"pick"}},"clarify",slot="source")
        add("unsupported_selection",{"selection":{"object_id":left["id"]},"context":{"goal":"pick"}},"reject")
        add("conflicting_context",{"context":{"goal":"pick","confirmed_action":"fold"}},"reject")
        add("repeat_place",{"context":{"goal":"repeat_last","history":[{"action":"place","source_id":cloth["id"],"destination_id":right["id"]}]}},"ready","place",right["id"])
        add("repeat_missing_history",{"context":{"goal":"repeat_last","history":[]}},"clarify",slot="action")
    assert_disjoint_scenes(*splits.values())
    paths={}
    for name,cases in splits.items():
        path=output_dir/(name+".jsonl")
        with path.open("w",encoding="utf-8") as stream:
            for case in cases:
                stream.write(json.dumps(case,ensure_ascii=False)+"\n")
        paths[name]=str(path.resolve())
    manifest={"provenance":"synthetic_smoke_only","warning":"Generated labels follow a small grammar; results do not establish research performance, real user intent, robot success or workload.","seed":seed,"scenes":scenes,"counts":{k:len(v) for k,v in splits.items()},"paths":paths}
    (output_dir/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return {**paths,"manifest":str((output_dir/"manifest.json").resolve()),"counts":manifest["counts"]}
