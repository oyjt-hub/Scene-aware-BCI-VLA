"""Append-only, public-protocol supervision for the step-08 development pilot.

This module reads only train/validation observations, from their frozen JSONL
files or reconstructed after filtering the host-local capture metadata.
Original noisy observations never become point labels by taking an argmax. New
point labels teach a restricted, explicitly declared selection protocol; they are
not claims about latent human intent or robustness to a real EEG decoder.

The language target and auxiliary labels are loss-only. ``public`` remains the
unchanged 4.1 contract; the tensorizer must never receive a supervision sidecar.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from bci_vla.data_v4.contracts import validate_private, validate_public
from bci_vla.data_v4.channel import diagnostic_summary
from bci_vla.interaction_contract_v4.validation import canonical_digest

SCHEMA_VERSION = "step08-supervision/0.1"
PROTOCOL = "explicit_public_declaration_v1"
SPLITS = ("train", "validation")
FAMILIES = (
    "noisy_ambiguous", "declared_pick", "missing_action", "missing_source",
    "conflicting_declarations", "no_response", "missing_destination",
    "next_pick_symbolic",
)
SLOTS = ("ACTION", "SOURCE", "DESTINATION")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _objects(public):
    return [obj for obj in public["scene"]["objects"] if obj["object_valid"]]


def _task(public, source, action="pick", destination=None):
    key = canonical_digest([action, source, destination])[:12]
    return {
        "task_id": "declared-task-" + key, "task_revision": 1,
        "suggestion_id": "declared-suggestion-" + key,
        "context": deepcopy(public["context"]), "action": action,
        "source_id": source, "destination_id": destination,
        "valid_until_s": 120.0, "capability_verified": False,
    }


def _event(public, role, references, chosen, *, kinds=None, proposal=None,
           no_response=False):
    sequence = len(public["history"]) + 1
    issued = 2.0 + sequence * 2
    options = [{"option_id": "o" + str(i), "symbol": "s" + str(i),
                "kind": (kinds or ["OBJECT"] * len(references))[i],
                "reference": ref} for i, ref in enumerate(references)]
    q = {"question_id": "declared-q-" + str(sequence), "role": role,
         "context": deepcopy(public["context"]),
         "layout_id": "declared-" + role.lower() + "-" + str(len(options)),
         "issued_at_s": issued, "expires_at_s": issued + 1,
         "presentation_ack": True, "options": options,
         "diagnostic_option_id": None, "proposal": deepcopy(proposal)}
    chosen_id = None if no_response else options[references.index(chosen)]["option_id"]
    evidence = {"score_kind": "NONE" if no_response else "HARD_LABEL",
                "values": [] if no_response else [
                    {"option_id": opt["option_id"], "value": int(opt["option_id"] == chosen_id)}
                    for opt in options], "selected_option_id": chosen_id,
                "calibration": None}
    event = {"event_id": "declared-e-" + str(sequence), "sequence": sequence,
             "role": "NO_RESPONSE" if no_response else role,
             "original_role": role if no_response else None, "question": q,
             "started_at_s": issued + 0.1,
             "ended_at_s": issued + (1.0 if no_response else 0.6),
             "valid": True, "invalid_reason": None, "evidence": evidence}
    public["history"].append(event)
    public["current_event_id"] = event["event_id"]
    public["as_of_s"] = event["ended_at_s"]
    if role in {"TASK", "INTENT_QUERY"} and not no_response and all(o["kind"] == "OBJECT" for o in options):
        public["selection_event_id"] = event["event_id"]
    return event


def _selected_reference(event):
    if event["role"] == "NO_RESPONSE":
        return None
    selected = event["evidence"]["selected_option_id"]
    return next(opt["reference"] for opt in event["question"]["options"] if opt["option_id"] == selected)


def make_declared_public(source_public, family, target_index=0):
    """Create simulated public declarations over an existing actual LIBERO frame.

    Every selected object is carried by a public event. No sample ID, hidden goal,
    filename, or fault stratum is used as a model feature or teacher decision.
    """
    if family not in FAMILIES or family == "noisy_ambiguous":
        raise ValueError("unsupported declaration family")
    public = deepcopy(source_public)
    objects = _objects(public)
    if len(objects) < 2:
        raise ValueError("declaration curriculum requires at least two objects")
    ids = [obj["instance_id"] for obj in objects]
    source = ids[target_index % len(ids)]
    other = ids[(target_index + 1) % len(ids)]
    public.update(history=[], current_event_id=None, selection_event_id=None,
                  task_options=[], as_of_s=max(1.0, public["scene"]["observed_at_s"]))
    if public["scene"]["observed_at_s"] > 0:
        raise ValueError("use actual initial-frame sources; no future frame relabelling")
    state = public["public_state"]
    state.update(confirmed_task=None, pending_suggestion=None,
                 pending_question=None, stop_latched=False, execution_authorized=False)
    if family != "missing_action":
        public["task_options"] = [_task(public, oid) for oid in ids]
    if family == "missing_destination":
        public["task_options"] = [_task(public, oid, "place", dest)
                                  for oid, dest in zip(ids, ids[1:] + ids[:1])]
    if family == "missing_source":
        # The available menu conveys the action; there is no decoded choice.
        pass
    elif family == "no_response":
        _event(public, "TASK", ids, None, no_response=True)
    else:
        _event(public, "TASK", ids, source)
        if family in {"declared_pick", "next_pick_symbolic"}:
            # An explicit second task declaration must agree with the object.
            tasks = [t["task_id"] for t in public["task_options"]]
            chosen = next(t["task_id"] for t in public["task_options"] if t["source_id"] == source)
            _event(public, "INTENT_QUERY", tasks, chosen, kinds=["TASK"] * len(tasks))
        elif family == "conflicting_declarations":
            _event(public, "INTENT_QUERY", ids, other)
        if family == "next_pick_symbolic":
            task = next(t for t in public["task_options"] if t["source_id"] == source)
            receipt = _event(public, "CONFIRM", ["accept", "reject"], "accept",
                             kinds=["DECISION", "DECISION"], proposal=task)
            state["confirmed_task"] = {"task": deepcopy(task), "confirmation_receipt": deepcopy(receipt)}
    state["diagnostics"] = diagnostic_summary(public["context"], [], public["as_of_s"], "probe-" + str(len(ids)))
    public["sample_id"] = canonical_digest([source_public["context"], family, target_index])[:32]
    validate_public(public)
    return public


def declared_teacher(public):
    """Public-only restricted protocol teacher; never accepts private labels.

    HARD_LABEL alone is not interpreted as a general reliable EEG observation.
    This teacher is valid only under the common explicit-declaration curriculum.
    """
    validate_public(public)
    ids = [obj["instance_id"] for obj in _objects(public)]
    aliases = {obj["instance_id"]: obj["alias"] for obj in _objects(public)}
    object_choices, task_choices = [], []
    lookup = {task["task_id"]: task for task in public["task_options"]}
    for event in public["history"]:
        if not event["valid"] or event["question"] is None or event["role"] not in {"TASK", "INTENT_QUERY"}:
            continue
        # Only synthetic, hard declared choices belong to this restricted teacher.
        if event["evidence"]["score_kind"] != "HARD_LABEL":
            continue
        q = event["question"]
        ref = _selected_reference(event)
        chosen = next(o for o in q["options"] if o["reference"] == ref)
        if chosen["kind"] == "OBJECT":
            object_choices.append(ref)
        elif chosen["kind"] == "TASK":
            task_choices.append(lookup[ref])
    confirmed = public["public_state"]["confirmed_task"]
    sources = set(object_choices)
    sources.update(task["source_id"] for task in task_choices)
    if confirmed:
        sources.add(confirmed["task"]["source_id"])
    resolved = len(sources) == 1
    source = next(iter(sources)) if resolved else None
    actions = {task["action"] for task in task_choices}
    if not actions:
        actions = {task["action"] for task in public["task_options"]}
    missing = [not bool(actions) or len(actions) != 1, not resolved, False]
    selected_task = task_choices[-1] if len(task_choices) == 1 else None
    if confirmed:
        selected_task = confirmed["task"]
    if actions == {"place"} and selected_task is None:
        missing[2] = True
    abstain = any(missing)
    target = None
    if not abstain and actions == {"pick"}:
        target = _task(public, source)
        text, action = "Pick up " + aliases[source] + ".", "PROPOSE_TASK"
    elif not abstain and selected_task is not None:
        target = deepcopy(selected_task)
        text = "Place " + aliases[target["source_id"]] + " into " + aliases[target["destination_id"]] + "."
        action = "PROPOSE_TASK"
    else:
        if missing[1]:
            text = "Which object do you mean?"
        elif missing[0]:
            text = "Which task do you want?"
        else:
            text = "Where should I place it?"
        action = "ASK_INTENT"
    # Missing task slots do not erase an otherwise explicit object binding.
    intent = {"mode": "POINT" if resolved else "SET",
              "candidate_ids": [source] if resolved else ids,
              "outside": False if resolved else True, "mask": True}
    return {"intent": intent, "identified": resolved, "task": target,
            "text": text, "interaction": action, "missing_slots": missing,
            "abstain": abstain}


def language_prompt(public, protocol, kind="main_task", gripper_state=None):
    """Online public prompt: dictionary/protocol only, never the selected answer.

    Full/B3/B4 receive event evidence through their token module. B2 may append
    ``baseline_evidence_text`` containing the *same* allowlisted public evidence.
    """
    dictionary = []
    for obj in public["scene"]["objects"]:
        dictionary.append({"alias": obj["alias"], "label": obj["label"],
                           "bbox_xyxy": obj["bbox_xyxy"], "object_valid": obj["object_valid"],
                           "roi_feature": obj["roi_feature"], "roi_valid": obj["roi_valid"],
                           "semantic_feature": obj["semantic_feature"], "semantic_valid": obj["semantic_valid"]})
    header = ("Protocol: explicit_public_declaration_v1. In this restricted protocol, public hard selections declare object/task choices. "
              "Use the interaction evidence; clarify missing or conflicting declarations. "
              if protocol == PROTOCOL else
              "Protocol: noisy_observation_v1. Decoded choices are noisy evidence, not confirmed latent intent. Clarify unresolved intent. ")
    question = ("Mode: NEXT_SUBTASK. Use the public confirmed task from interaction evidence. Current gripper state: " + gripper_state + ". State the next subtask. "
                if kind == "next_subtask" else
                "Mode: MAIN_TASK. Use public task choices and available action options from interaction evidence. State a task suggestion or ask for missing information. ")
    return "answer en Objects: " + json.dumps(dictionary, ensure_ascii=False, separators=(",", ":")) + ". " + header + question + "Output only the answer using the OBJ aliases.\n"


def baseline_evidence_text(public):
    """Delegate B2 serialization to the common allowlisted tensor-view API."""
    from bci_vla.training_v4.text_baseline import baseline_evidence_text as serialize
    return serialize(public)


def make_example(source_public, source_labels, split, family, target_index=0, *, image_sha256=None):
    if split not in SPLITS:
        raise ValueError("only train/validation are allowed")
    if not source_labels:
        raise ValueError("private provenance variants required")
    for labels in source_labels:
        validate_private(labels, source_public)
        if labels["split"] != split:
            raise ValueError("source split mismatch")
    # Same-observation latent counterfactuals must not increase training weight or
    # force different targets. This guard is also applied by the loader below.
    if any(labels["labels"] != source_labels[0]["labels"] for labels in source_labels):
        raise ValueError("same public has contradictory labels")
    source = {
        "sample_id": source_public["sample_id"],
        "base_episode_id": source_public["context"]["session_id"].removeprefix("session-"),
        "group_id": source_public["context"]["scene_id"],
        "image_ref": source_public["scene"]["image_ref"], "image_sha256": image_sha256,
    }
    if family == "noisy_ambiguous":
        public = deepcopy(source_public)
        labels = deepcopy(source_labels)
        protocol = "noisy_observation_v1"
        text = labels[0]["labels"]["main_task"]["text"]
        mask = labels[0]["labels"]["main_task"]["mask"]
        # Only the unresolved source slot is established; unknown ACTION and
        # DESTINATION are masked rather than assigned arbitrary negatives.
        auxiliary = {"missing_slots": [False, True, False],
                     "missing_slots_mask": [False, True, False],
                     "abstain": True, "abstain_mask": True}
        kind, visual, gripper = "main_task", True, None
    else:
        public = make_declared_public(source_public, family, target_index)
        teaching = declared_teacher(public)
        protocol = PROTOCOL
        sha = canonical_digest(public)
        # Hidden audit data in this synthetic protocol is not a network input.
        # Ambiguous examples retain NONE goal, never an arbitrary selected truth.
        goal = teaching["task"] if teaching["identified"] else None
        if goal is None and teaching["identified"]:
            goal = _task(public, teaching["intent"]["candidate_ids"][0])
        label = {
            "record_type": "PRIVATE_LABELS", "schema_version": "bci-vla-interaction/4.1",
            "sample_id": public["sample_id"], "variant_id": "public-protocol",
            "public_input_sha256": sha, "split": split,
            "hidden_truth": {"goal": goal, "channel_injected_mode": "public-declaration-curriculum", "changed_at_s": None},
            "identifiability": {"intent": "IDENTIFIED" if teaching["identified"] else "AMBIGUOUS", "channel": "INSUFFICIENT"},
            "labels": {
                "intent": teaching["intent"],
                "main_task": {"mode": "TASK" if teaching["task"] else "CLARIFICATION",
                              "task": teaching["task"], "text": teaching["text"], "mask": True},
                "channel": {"mask": False, "next_probe_success": None, "target_time_s": None},
                "interaction": {"mask": True, "action": teaching["interaction"],
                                "teacher": "public_rule_v1", "teacher_input_sha256": sha},
            },
        }
        labels = [label]
        auxiliary = {"missing_slots": teaching["missing_slots"],
                     "missing_slots_mask": [True] * 3,
                     "abstain": teaching["abstain"], "abstain_mask": True}
        kind = "next_subtask" if family == "next_pick_symbolic" else "main_task"
        gripper = "empty" if kind == "next_subtask" else None
        # Explicit symbolic state interventions are TEXT ONLY. No initial frame
        # is misrepresented as evidence of a grasp/placement endpoint.
        visual = kind != "next_subtask"
        text, mask = teaching["text"], True
    for label in labels:
        validate_private(label, public)
    supervision = {
        "schema_version": SCHEMA_VERSION, "sample_id": public["sample_id"],
        "public_input_sha256": canonical_digest(public), "split": split,
        "family": family, "source": source, "protocol": protocol,
        "teacher": "deterministic_public_protocol_templates_not_human_annotations",
        "auxiliary": auxiliary,
        "language": {"kind": kind, "prompt": language_prompt(public, protocol, kind, gripper),
                     "target": text, "mask": mask, "visual": visual,
                     "gripper_state": gripper,
                     "gripper_provenance": "declared_symbolic_intervention_not_RGB_inference" if gripper else None},
        "private_variant_count": len(labels),
    }
    return {"public": public, "private_labels": labels[0], "private_variants": labels, "supervision": supervision}


def validate_example(example):
    public, private, supervision = example["public"], example["private_labels"], example["supervision"]
    validate_public(public)
    validate_private(private, public)
    if supervision["schema_version"] != SCHEMA_VERSION or supervision["split"] not in SPLITS:
        raise ValueError("supervision version/split")
    if supervision["sample_id"] != public["sample_id"] or supervision["public_input_sha256"] != canonical_digest(public):
        raise ValueError("supervision public binding")
    if private["split"] != supervision["split"]:
        raise ValueError("supervision private split")
    auxiliary = supervision["auxiliary"]
    for key in ("missing_slots", "missing_slots_mask"):
        if len(auxiliary[key]) != 3 or any(type(v) is not bool for v in auxiliary[key]):
            raise ValueError("missing-slot target/mask")
    if any(type(auxiliary[key]) is not bool for key in ("abstain", "abstain_mask")):
        raise ValueError("abstain target/mask")
    language = supervision["language"]
    family = supervision["family"]
    if family not in FAMILIES:
        raise ValueError("unregistered supervision family")
    expected_protocol = "noisy_observation_v1" if family == "noisy_ambiguous" else PROTOCOL
    expected_kind = "next_subtask" if family == "next_pick_symbolic" else "main_task"
    if supervision["protocol"] != expected_protocol or language["kind"] != expected_kind:
        raise ValueError("family/protocol/language-kind mismatch")
    expected_prompt = language_prompt(public, supervision["protocol"], language["kind"], language["gripper_state"])
    if language["prompt"] != expected_prompt:
        raise ValueError("language prompt includes non-public or inconsistent fields")
    if language["mask"] and (not isinstance(language["target"], str) or not language["target"]):
        raise ValueError("language target required")
    if language["target"] != private["labels"]["main_task"]["text"] or language["mask"] != private["labels"]["main_task"]["mask"]:
        raise ValueError("language/private teacher target mismatch")
    if language["kind"] == "next_subtask" and (language["visual"] or language["gripper_state"] != "empty"):
        raise ValueError("unverified phase cannot be visual language supervision")
    variants = example.get("private_variants", [private])
    if len(variants) != supervision["private_variant_count"]:
        raise ValueError("counterfactual multiplicity")
    for variant in variants:
        validate_private(variant, public)
        if variant["split"] != supervision["split"]:
            raise ValueError("counterfactual variant split mismatch")
        if variant["labels"] != private["labels"]:
            raise ValueError("identical public has contradictory private targets")
    if len({canonical_digest(v["hidden_truth"]) for v in variants}) > 1:
        if private["identifiability"]["intent"] == "IDENTIFIED" or private["labels"]["main_task"]["mode"] == "TASK":
            raise ValueError("hidden-truth ambiguity cannot force point/task target")
    if family != "noisy_ambiguous":
        teacher = declared_teacher(public)
        if private["labels"]["intent"] != teacher["intent"] or private["labels"]["main_task"]["task"] != teacher["task"] or language["target"] != teacher["text"]:
            raise ValueError("declaration labels disagree with public-only teacher")
        if auxiliary["missing_slots"] != teacher["missing_slots"] or auxiliary["abstain"] != teacher["abstain"] or auxiliary["missing_slots_mask"] != [True] * 3 or not auxiliary["abstain_mask"]:
            raise ValueError("declaration auxiliary labels disagree with public teacher")
    elif private["identifiability"]["intent"] == "IDENTIFIED" or private["labels"]["main_task"]["mode"] == "TASK":
        raise ValueError("noisy evidence cannot force an identified intent/task")
    return True


def load_partition(root, split):
    """Return one training row per public observation; variants are audit-only."""
    if split not in SPLITS:
        raise ValueError("calibration and formal test are forbidden in step08")
    folder = Path(root) / split
    publics = _read_jsonl(folder / "public_input.jsonl")
    sidecars = _read_jsonl(folder / "supervision.jsonl")
    private = defaultdict(list)
    for label in _read_jsonl(folder / "private_labels.jsonl"):
        private[label["sample_id"]].append(label)
    if len({p["sample_id"] for p in publics}) != len(publics):
        raise ValueError("duplicate public sample would increase training weight")
    by_id = {s["sample_id"]: s for s in sidecars}
    if len(by_id) != len(sidecars) or set(by_id) != {p["sample_id"] for p in publics} or set(private) != set(by_id):
        raise ValueError("public/private/supervision ID mismatch")
    rows = []
    for public in publics:
        sid = public["sample_id"]
        row = {"public": public, "private_labels": private[sid][0],
               "private_variants": private[sid], "supervision": by_id[sid]}
        validate_example(row)
        rows.append(row)
    return rows


def _capture_rows(captures, split):
    """Recreate ONLY allowed development observations from host-local captures.

    This imports the original pure generator, never its filesystem-writing main.
    The original ambiguous latent twin is reconstructed with the original task
    constructor, preserving its identical public observation and equal labels.
    """
    from execution.v4.build_step04_data import generate, task
    publics, private = [], []
    for cap in captures:
        if cap["partition"] != split:
            continue
        records, _ = generate(cap, "ambiguous")
        for public, label, _future_audit in records:
            publics.append(public)
            private.append(label)
            twin = deepcopy(label)
            twin["variant_id"] = "latent-1"
            alternate = next(obj["instance_id"] for obj in public["scene"]["objects"]
                             if obj["object_valid"] and obj["instance_id"] != label["hidden_truth"]["goal"]["source_id"])
            twin["hidden_truth"]["goal"] = task(public["context"], alternate)
            private.append(twin)
    return publics, private


def build_training_data(source_root, output_root, config=None, *, capture_root=None):
    """Build bounded train/validation data, refusing to overwrite any output.

    The source root is the existing step04 directory on either local host or
    server. No upload is needed. JSONL mode never opens mixed capture/split
    manifests. Capture mode reads the cross-partition capture manifest as metadata
    and filters partitions BEFORE generation/image processing. Neither mode reads
    calibration examples/images, future-probe files, or sealed-test examples.
    """
    config = config or {"budgets": {"train": 128, "validation": 32}}
    budgets = config["budgets"]
    if set(budgets) != set(SPLITS) or any(type(budgets[s]) is not int or budgets[s] <= 0 for s in SPLITS):
        raise ValueError("budgets require positive train/validation only")
    if budgets["train"] > 128 or budgets["validation"] > 32:
        raise ValueError("step08 bounded pilot budget exceeded")
    source_root = Path(capture_root or source_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("output must be outside the frozen source data directory")
    if output_root.exists():
        raise FileExistsError("append-only dataset: choose a new output directory")
    source_hashes = []
    captures = None
    if capture_root is not None:
        capture_manifest = source_root / "captures/capture_manifest.json"
        source_hashes.append({"path": "captures/capture_manifest.json", "sha256": _sha(capture_manifest),
                              "usage": "cross-partition metadata; filter before processing"})
        metadata = json.loads(capture_manifest.read_text(encoding="utf-8-sig"))
        # Only these records reach generate(). No calibration/test frame is opened.
        captures = [cap for cap in metadata["episodes"] if cap["partition"] in SPLITS]
    examples = {}
    groups = {}
    for split in SPLITS:
        folder = source_root / split
        if captures is None:
            paths = [folder / "public_input.jsonl", folder / "private_labels.jsonl"]
            for path in paths:
                source_hashes.append({"path": path.relative_to(source_root).as_posix(), "sha256": _sha(path)})
            publics, private_rows = _read_jsonl(paths[0]), _read_jsonl(paths[1])
        else:
            publics, private_rows = _capture_rows(captures, split)
        private = defaultdict(list)
        for label in private_rows:
            private[label["sample_id"]].append(label)
        episodes = defaultdict(list)
        noisy_episodes = defaultdict(list)
        for public in publics:
            if len(_objects(public)) >= 2:
                noisy_episodes[public["context"]["session_id"]].append(public)
            if public["scene"]["observed_at_s"] == 0 and len(_objects(public)) >= 2:
                episodes[public["context"]["session_id"]].append(public)
        if not episodes:
            raise ValueError("no eligible actual LIBERO initial frames")
        groups[split] = {p["context"]["scene_id"] for p in publics}
        seeds = []
        noisy_seeds = []
        for episode in sorted(episodes):
            candidates = episodes[episode]
            # Prefer actual frozen identical-public latent twins as ambiguity
            # anchors, not a new fictitious counterfactual.
            candidates.sort(key=lambda p: (len(private[p["sample_id"]]) < 2, p["as_of_s"], p["sample_id"]))
            seed = candidates[0]
            if len(private[seed["sample_id"]]) < 2:
                raise ValueError("each episode needs a real frozen ambiguity anchor")
            seeds.append(seed)
            # The noisy anchor uses an actually captured later frame with valid
            # public PROBE support. This preserves useful private future-Bernoulli
            # labels without opening any future-probe artifact or revealing it.
            noisy_candidates = noisy_episodes[episode]
            noisy_candidates.sort(key=lambda p: (
                len(private[p["sample_id"]]) < 2,
                p["public_state"]["diagnostics"]["status"] != "SUFFICIENT",
                -p["as_of_s"], p["sample_id"]))
            noisy_seeds.append(noisy_candidates[0])
        image_hashes = {}
        for seed in seeds + noisy_seeds:
            image_ref = seed["scene"]["image_ref"]
            image = (source_root / image_ref).resolve()
            if source_root not in image.parents or image.suffix.lower() != ".png":
                raise ValueError("image path escapes source data")
            if image_ref not in image_hashes:
                image_hashes[image_ref] = _sha(image)
                source_hashes.append({"path": image_ref, "sha256": image_hashes[image_ref]})
        selected = []
        for index in range(budgets[split]):
            episode_index = index % len(seeds)
            cycle = index // len(seeds)
            family = FAMILIES[(episode_index + cycle) % len(FAMILIES)]
            seed = noisy_seeds[episode_index] if family == "noisy_ambiguous" else seeds[episode_index]
            row = make_example(seed, private[seed["sample_id"]], split, family,
                               target_index=episode_index + cycle,
                               image_sha256=image_hashes[seed["scene"]["image_ref"]])
            validate_example(row)
            selected.append(row)
        if len({r["public"]["sample_id"] for r in selected}) != len(selected):
            raise ValueError("budget revisits the same original ambiguous observation")
        examples[split] = selected
    if groups["train"] & groups["validation"]:
        raise ValueError("train/validation scene-group overlap")
    # Check all read source assets again before any output creation.
    for entry in source_hashes:
        if _sha(source_root / entry["path"]) != entry["sha256"]:
            raise ValueError("source changed during data construction")
    output_root.mkdir(parents=True)
    files = []
    counts = {}
    for split in SPLITS:
        folder = output_root / split
        folder.mkdir()
        rows = examples[split]
        for name, values in (
            ("public_input", [r["public"] for r in rows]),
            ("private_labels", [v for r in rows for v in r["private_variants"]]),
            ("supervision", [r["supervision"] for r in rows]),
        ):
            path = folder / (name + ".jsonl")
            _write_jsonl(path, values)
            files.append({"path": path.relative_to(output_root).as_posix(), "sha256": _sha(path)})
        counts[split] = {
            "public_rows": len(rows), "private_variants": sum(len(r["private_variants"]) for r in rows),
            "base_episodes": len({r["supervision"]["source"]["base_episode_id"] for r in rows}),
            "scene_groups": len(groups[split]),
            "family_counts": dict(Counter(r["supervision"]["family"] for r in rows)),
            "point_targets": sum(r["private_labels"]["labels"]["intent"]["mode"] == "POINT" for r in rows),
            "counterfactual_anchors": sum(len(r["private_variants"]) > 1 for r in rows),
            "channel_targets": sum(r["private_labels"]["labels"]["channel"]["mask"] for r in rows),
            "channel_outcomes": dict(Counter(str(r["private_labels"]["labels"]["channel"]["next_probe_success"])
                                              for r in rows if r["private_labels"]["labels"]["channel"]["mask"])),
            "episode_counts": dict(Counter(r["supervision"]["source"]["base_episode_id"] for r in rows)),
        }
    manifest = {
        "version": "step08-public-protocol-pilot/0.1", "schema_version": SCHEMA_VERSION,
        "parent": "frozen step04 4.1 train/validation only", "counts": counts,
        "source_mode": "host_local_capture_reconstruction" if captures is not None else "frozen_public_private_jsonl",
        "cross_partition_capture_metadata_read": captures is not None,
        "source_files_read": source_hashes, "files": files,
        "actual_LIBERO_images": True, "simulated_BCI_only": True,
        "real_EEG": False, "human_annotations": False,
        "calibration_accessed": False, "formal_test_accessed": False,
        "calibration_examples_processed": False, "test_examples_processed": False,
        "future_probe_files_accessed": False, "source_data_modified": False,
        "same_public_hidden_variants_weighted_once": True,
        "same_information_for_methods": ["Full", "B2", "B3", "B4"],
        "language_prompts_do_not_include_selected_target": True,
        "general_hidden_intent_supervision_ready": False,
        "place_phase_supervision_ready": False,
        "restricted_public_protocol_pilot_ready": True,
        "limitations": [
            "Point labels teach explicit public declarations, not latent intent under arbitrary decoder noise.",
            "Original noisy counterfactuals remain ambiguous; private variants are not independently weighted.",
            "Main-task proposals use an explicit public pick-only action menu; no open-ended task planning.",
            "Next-subtask empty-gripper examples are explicit symbolic TEXT-ONLY interventions, not observed manipulation phases.",
            "No placement-stage language target, new dynamics rollout, successful robot task, real EEG, or formal-test evidence.",
            "RGB crop statistics and projected privileged boxes are inherited unchanged; Florence/SAM are not connected.",
        ],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for split in SPLITS:
        load_partition(output_root, split)
    return manifest
