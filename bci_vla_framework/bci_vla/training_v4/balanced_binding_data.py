"""Append-only, target-balanced public-declaration teaching for binding repair.

The original 4.1 public/private contract and step08 row interface are preserved.
Binding groups/pairs are sidecar metadata for sampling, losses and audit only;
they must never be passed to the connector. No target, fault label or answer is
added to the shared native VLM prompt. This remains simulated declaration
teaching, not reliable latent human-intent supervision or action execution.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

from bci_vla.interaction_contract_v4.validation import canonical_digest
from bci_vla.training_v4.data import (
    SPLITS, _capture_rows, _objects, _read_jsonl, _sha, _write_jsonl,
    declared_teacher, language_prompt, load_partition, make_example, validate_example,
)

VERSION = "step09-balanced-public-binding/0.1"
BINDING_FAMILIES = ("declared_pick", "next_pick_symbolic")
AUXILIARY_FAMILIES = ("missing_action", "missing_source", "conflicting_declarations",
                      "no_response", "missing_destination", "noisy_ambiguous")
DEFAULT_CONFIG = {"max_rows": {"train": 512, "validation": 128},
                  "alias_policy": "complete_cyclic_permutation_for_alternating_family_per_episode"}


def _key(*values):
    return canonical_digest([VERSION, *values])[:32]


def joint_alias_permutation(row, shift):
    """Rename dictionary/answer aliases jointly while preserving all physical IDs."""
    result = deepcopy(row)
    public = result["public"]
    aliases = [obj["alias"] for obj in public["scene"]["objects"]]
    if type(shift) is not int or not 0 < shift < len(aliases):
        raise ValueError("non-identity cyclic alias shift required")
    mapping = dict(zip(aliases, aliases[shift:] + aliases[:shift]))
    for obj in public["scene"]["objects"]:
        obj["alias"] = mapping[obj["alias"]]
    public["sample_id"] = _key("joint_alias", row["public"]["sample_id"], mapping)
    sha = canonical_digest(public)
    teaching = declared_teacher(public)
    for label in result["private_variants"]:
        label.update(sample_id=public["sample_id"], public_input_sha256=sha)
        label["labels"]["main_task"]["text"] = teaching["text"]
        label["labels"]["interaction"]["teacher_input_sha256"] = sha
    result["private_labels"] = result["private_variants"][0]
    supervision = result["supervision"]
    supervision.update(sample_id=public["sample_id"], public_input_sha256=sha)
    language = supervision["language"]
    language["target"] = teaching["text"]
    language["prompt"] = language_prompt(public, supervision["protocol"], language["kind"], language["gripper_state"])
    validate_example(result)
    return result, mapping


def _initial_and_noisy_sources(publics, labels):
    """Use initial observed images for declarations, real later PROBE anchors for diagnostics."""
    episodes = defaultdict(list)
    for public in publics:
        if len(_objects(public)) >= 2:
            episodes[public["context"]["session_id"]].append(public)
    sources = []
    for episode in sorted(episodes):
        rows = episodes[episode]
        initial = [p for p in rows if p["scene"]["observed_at_s"] == 0]
        initial.sort(key=lambda p: (len(labels[p["sample_id"]]) < 2, p["as_of_s"], p["sample_id"]))
        noisy = sorted(rows, key=lambda p: (len(labels[p["sample_id"]]) < 2,
                                           p["public_state"]["diagnostics"]["status"] != "SUFFICIENT",
                                           -p["as_of_s"], p["sample_id"]))
        if not initial or len(labels[noisy[0]["sample_id"]]) < 2:
            raise ValueError("each episode requires an initial frame and a real frozen ambiguous pair")
        if not all(label["labels"]["channel"]["mask"] for label in labels[noisy[0]["sample_id"]]):
            raise ValueError("each noisy anchor needs valid public-supported channel supervision")
        sources.append((initial[0], noisy[0]))
    if not sources:
        raise ValueError("no eligible train/validation episode")
    return sources


def _bind_metadata(row, *, choice_group, alias_group, alias_permutation, source_position):
    row["supervision"]["binding"] = {
        "version": VERSION, "is_binding_row": True,
        "choice_group_id": choice_group, "alias_group_id": alias_group,
        "alias_permutation_id": alias_permutation,
        "source_position_audit_only": source_position,
        "metadata_never_model_features": True,
    }
    return row


def _choice_group(rows, episode, family, alias_permutation):
    ids = [row["public"]["sample_id"] for row in rows]
    prompts = {row["supervision"]["language"]["prompt"] for row in rows}
    targets = [row["supervision"]["language"]["target"] for row in rows]
    physical = [row["private_labels"]["labels"]["intent"]["candidate_ids"][0] for row in rows]
    if len(prompts) != 1 or len(set(targets)) != len(rows) or len(set(physical)) != len(rows):
        raise ValueError("choice group must preserve native prompt and cover distinct physical targets")
    pairs = [{"first": ids[i], "second": ids[j]} for i in range(len(ids)) for j in range(i + 1, len(ids))]
    return {"group_id": rows[0]["supervision"]["binding"]["choice_group_id"],
            "episode": episode, "family": family, "alias_permutation_id": alias_permutation,
            "sample_ids": ids, "physical_target_ids_audit_only": physical,
            "target_aliases": [target.removeprefix("Pick up ").removesuffix(".") for target in targets],
            "shared_native_prompt_sha256": canonical_digest(next(iter(prompts))),
            "all_candidate_targets_covered": True, "pairs": pairs}


def build_episode_rows(initial, noisy, private, split, *, episode_position,
                       scene_episode_position, image_hashes):
    """Pure per-episode complete groups; caps are checked before writing anything."""
    n = len(_objects(initial))
    episode = initial["context"]["session_id"].removeprefix("session-")
    rows, choices, aliases = [], [], []
    rotated_family = BINDING_FAMILIES[scene_episode_position % len(BINDING_FAMILIES)]
    # Rotate every physical candidate, not one arbitrarily selected answer.
    # Within each scene the first four episodes cover cyclic shifts in sequence.
    shift = 1 + (scene_episode_position % (n - 1))
    permutation = _key("cyclic", n, shift)
    for family in BINDING_FAMILIES:
        base = []
        for index in range(n):
            row = make_example(initial, private[initial["sample_id"]], split, family, index,
                               image_sha256=image_hashes[initial["scene"]["image_ref"]])
            alias_group = _key(episode, family, "physical", index)
            _bind_metadata(row, choice_group=_key(episode, family, "identity"),
                           alias_group=alias_group, alias_permutation="identity", source_position=index)
            base.append(row)
        rows.extend(base)
        choices.append(_choice_group(base, episode, family, "identity"))
        if family == rotated_family:
            rotated = []
            for index, first in enumerate(base):
                second, mapping = joint_alias_permutation(first, shift)
                _bind_metadata(second, choice_group=_key(episode, family, permutation),
                               alias_group=first["supervision"]["binding"]["alias_group_id"],
                               alias_permutation=permutation, source_position=index)
                aliases.append({"pair_id": _key("alias_pair", first["public"]["sample_id"], second["public"]["sample_id"]),
                                "episode": episode, "family": family, "first": first["public"]["sample_id"],
                                "second": second["public"]["sample_id"], "mapping": mapping,
                                "physical_target_id_audit_only": first["private_labels"]["labels"]["intent"]["candidate_ids"][0]})
                rotated.append(second)
            rows.extend(rotated)
            choices.append(_choice_group(rotated, episode, family, permutation))
    for family in AUXILIARY_FAMILIES:
        source = noisy if family == "noisy_ambiguous" else initial
        row = make_example(source, private[source["sample_id"]], split, family,
                           target_index=episode_position % n,
                           image_sha256=image_hashes[source["scene"]["image_ref"]])
        row["supervision"]["binding"] = {"version": VERSION, "is_binding_row": False,
                                         "metadata_never_model_features": True}
        rows.append(row)
    for row in rows:
        validate_example(row)
    return rows, choices, aliases


def _counts(rows, choices, aliases):
    binding = [r for r in rows if r["supervision"]["binding"]["is_binding_row"]]
    families = defaultdict(list)
    for row in binding:
        families[row["supervision"]["family"]].append(row)
    targets_by_family = {
        family: dict(Counter(r["supervision"]["language"]["target"] for r in group))
        for family, group in sorted(families.items())}
    point_target_aliases = []
    alias_group_sizes = Counter(r["supervision"]["binding"]["alias_group_id"] for r in binding)
    episode_family_targets = defaultdict(lambda: defaultdict(Counter))
    for row in binding:
        episode = row["supervision"]["source"]["base_episode_id"]
        family = row["supervision"]["family"]
        target = row["private_labels"]["labels"]["intent"]["candidate_ids"][0]
        episode_family_targets[episode][family][target] += 1
    for row in rows:
        intent = row["private_labels"]["labels"]["intent"]
        if intent["mode"] == "POINT":
            alias = next(obj["alias"] for obj in row["public"]["scene"]["objects"] if obj["instance_id"] == intent["candidate_ids"][0])
            point_target_aliases.append(alias)
    return {
        "public_rows": len(rows), "private_variants": sum(len(r["private_variants"]) for r in rows),
        "base_episodes": len({r["supervision"]["source"]["base_episode_id"] for r in rows}),
        "scene_groups": len({r["supervision"]["source"]["group_id"] for r in rows}),
        "family_counts": dict(Counter(r["supervision"]["family"] for r in rows)),
        "binding_rows": len(binding), "choice_groups": len(choices),
        "complete_choice_pairs": sum(len(group["pairs"]) for group in choices),
        "alias_groups": len(alias_group_sizes),
        "alias_group_size_distribution": dict(Counter(str(n) for n in alias_group_sizes.values())),
        "joint_alias_pairs": len(aliases),
        "joint_alias_pairs_by_family": dict(Counter(pair["family"] for pair in aliases)),
        "binding_language_targets_by_family": targets_by_family,
        "physical_target_counts_by_episode_and_family_audit_only": {
            episode: {family: dict(targets) for family, targets in sorted(groups.items())}
            for episode, groups in sorted(episode_family_targets.items())},
        "point_target_alias_distribution_including_auxiliary": dict(Counter(point_target_aliases)),
        "channel_targets": sum(r["private_labels"]["labels"]["channel"]["mask"] for r in rows),
        "channel_outcomes": dict(Counter(str(r["private_labels"]["labels"]["channel"]["next_probe_success"])
                                        for r in rows if r["private_labels"]["labels"]["channel"]["mask"])),
        "counterfactual_anchors": sum(len(r["private_variants"]) > 1 for r in rows),
        "episode_counts": dict(Counter(r["supervision"]["source"]["base_episode_id"] for r in rows)),
    }


def build_balanced_binding_data(source_root, output_root, config=None, *, capture_root=None):
    """Build new complete groups from frozen train/validation, locally on either host.

    JSONL mode reads only the four train/validation public/private files and used
    images. Capture mode reads cross-partition metadata, filters FIRST and calls
    the original pure generator only for train/validation. No source is mutated.
    A cap too small for all complete episode groups raises BEFORE output exists;
    it never silently drops an episode, candidate, pair side or alias group.
    """
    config = deepcopy(config or DEFAULT_CONFIG)
    caps = config["max_rows"]
    if set(caps) != set(SPLITS) or any(type(caps[s]) is not int or caps[s] <= 0 for s in SPLITS):
        raise ValueError("only positive train/validation caps are allowed")
    source_root = Path(capture_root or source_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("output must be outside frozen source directory")
    if output_root.exists():
        raise FileExistsError("append-only data: choose a new output directory")
    source_files, captures = [], None
    if capture_root is not None:
        path = source_root / "captures/capture_manifest.json"
        source_files.append({"path": "captures/capture_manifest.json", "sha256": _sha(path),
                             "usage": "cross-partition metadata; filter before example/image processing"})
        metadata = json.loads(path.read_text(encoding="utf-8-sig"))
        captures = [cap for cap in metadata["episodes"] if cap["partition"] in SPLITS]
    built, partition_groups = {}, {}
    for split in SPLITS:
        if captures is None:
            paths = [source_root / split / name for name in ("public_input.jsonl", "private_labels.jsonl")]
            for path in paths:
                source_files.append({"path": path.relative_to(source_root).as_posix(), "sha256": _sha(path)})
            publics, labels = _read_jsonl(paths[0]), _read_jsonl(paths[1])
        else:
            publics, labels = _capture_rows(captures, split)
        private = defaultdict(list)
        for label in labels:
            private[label["sample_id"]].append(label)
        sources = _initial_and_noisy_sources(publics, private)
        partition_groups[split] = {p["context"]["scene_id"] for p in publics}
        image_hashes = {}
        for initial, noisy in sources:
            for public in (initial, noisy):
                ref = public["scene"]["image_ref"]
                path = (source_root / ref).resolve()
                if source_root not in path.parents or path.suffix.lower() != ".png":
                    raise ValueError("source image path escapes allowed data root")
                if ref not in image_hashes:
                    image_hashes[ref] = _sha(path)
                    source_files.append({"path": ref, "sha256": image_hashes[ref]})
        scene_positions = Counter()
        rows, choices, aliases = [], [], []
        for index, (initial, noisy) in enumerate(sources):
            scene = initial["context"]["scene_id"]
            position = scene_positions[scene]
            scene_positions[scene] += 1
            generated = build_episode_rows(initial, noisy, private, split, episode_position=index,
                                           scene_episode_position=position, image_hashes=image_hashes)
            rows.extend(generated[0]); choices.extend(generated[1]); aliases.extend(generated[2])
        if len(rows) > caps[split]:
            raise ValueError(f"{split}: complete episode/target/alias groups require {len(rows)} rows, cap={caps[split]}; no group was truncated")
        if len({r["public"]["sample_id"] for r in rows}) != len(rows):
            raise ValueError("duplicate public row would change training weight")
        built[split] = (rows, choices, aliases)
    if partition_groups["train"] & partition_groups["validation"]:
        raise ValueError("frozen train/validation scene-group overlap")
    for item in source_files:
        if _sha(source_root / item["path"]) != item["sha256"]:
            raise ValueError("frozen source changed during build")
    output_root.mkdir(parents=True)
    files, counts = [], {}
    for split in SPLITS:
        rows, choices, aliases = built[split]
        folder = output_root / split
        folder.mkdir()
        for name, values in (
            ("public_input", [r["public"] for r in rows]),
            ("private_labels", [v for r in rows for v in r["private_variants"]]),
            ("supervision", [r["supervision"] for r in rows]),
            ("choice_groups", choices), ("joint_alias_pairs", aliases),
        ):
            path = folder / (name + ".jsonl")
            _write_jsonl(path, values)
            files.append({"path": path.relative_to(output_root).as_posix(), "sha256": _sha(path)})
        counts[split] = _counts(rows, choices, aliases)
        if load_partition(output_root, split) != rows:
            raise ValueError("written row interface failed lossless reload")
    manifest = {
        "version": VERSION, "parent": "frozen step04 4.1 train/validation; prior04/08 untouched",
        "source_mode": "host_local_capture_reconstruction" if captures is not None else "frozen_public_private_jsonl",
        "counts": counts, "config": config, "source_files_read": source_files, "files": files,
        "cross_partition_capture_metadata_read": captures is not None,
        "calibration_examples_processed": False, "test_examples_processed": False,
        "calibration_accessed": False, "formal_test_accessed": False,
        "real_EEG": False, "physical_robot": False, "new_dynamics_rollout": False,
        "actual_LIBERO_images": True, "simulated_BCI_only": True,
        "source_data_modified": False, "same_public_hidden_variants_weighted_once": True,
        "complete_episode_candidate_alias_groups_never_truncated": True,
        "module_alias_ID_features_added": False, "shared_prompt_contains_selected_answer": False,
        "same_information_for_methods": ["Full", "B2", "B3", "B4"],
        "general_hidden_intent_supervision_ready": False, "place_phase_supervision_ready": False,
        "scope": "public-only explicit declaration teaching for token-language binding repair",
        "limitations": [
            "Balanced declaration targets do not establish hidden intent under arbitrary EEG decoder noise.",
            "next_pick_symbolic remains TEXT ONLY with declared empty gripper, not manipulation phase evidence.",
            "Target/alias/pair sidecars are sampler/loss audit metadata, never connector forward features.",
            "Alias counts across all scenes reflect varying candidate counts; within each fixed prompt all candidates are exactly balanced.",
            "More rows/pairs/aliases reuse the same 24 train and 8 validation episodes and are not new independent robot experiments.",
        ],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
