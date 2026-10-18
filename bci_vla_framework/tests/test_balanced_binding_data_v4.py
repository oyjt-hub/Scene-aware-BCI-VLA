"""Balanced-data source preservation, full-pair coverage and model-input controls."""
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bci_vla.interaction_contract_v4.validation import canonical_digest
from bci_vla.training_v4.balanced_binding_data import (
    AUXILIARY_FAMILIES, BINDING_FAMILIES, _initial_and_noisy_sources,
    build_balanced_binding_data, build_episode_rows, joint_alias_permutation,
)
from bci_vla.training_v4.data import _read_jsonl, _sha, load_partition, validate_example

ROOT = Path(__file__).resolve().parents[1]


class BalancedBindingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="bci_balanced_binding_")
        cls.output = Path(cls.temp.name) / "jsonl"
        cls.source = ROOT / "data/v4"
        cls.frozen_paths = [cls.source / split / name for split in ("train", "validation")
                            for name in ("public_input.jsonl", "private_labels.jsonl")]
        cls.frozen_paths += [ROOT / "configs/v4/train_data_step08.json", ROOT / "bci_vla/training_v4/data.py"]
        optional = ROOT / "execution/v4/08_training_audit.json"
        if optional.is_file():
            cls.frozen_paths.append(optional)
        cls.before = {str(path): _sha(path) for path in cls.frozen_paths}
        cls.manifest = build_balanced_binding_data(cls.source, cls.output)
        cls.rows = {s: load_partition(cls.output, s) for s in ("train", "validation")}
        cls.groups = {s: _read_jsonl(cls.output / s / "choice_groups.jsonl") for s in cls.rows}
        cls.aliases = {s: _read_jsonl(cls.output / s / "joint_alias_pairs.jsonl") for s in cls.rows}
        cls.by_id = {s: {r["public"]["sample_id"]: r for r in cls.rows[s]} for s in cls.rows}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_01_old_data_and_old_step08_code_are_unchanged(self):
        for path, expected in self.before.items():
            self.assertEqual(_sha(path), expected, path)
        self.assertEqual(len(self.rows["train"]), 432)
        self.assertEqual(len(self.rows["validation"]), 120)
        self.assertFalse(self.manifest["source_data_modified"])
        self.assertEqual(self.manifest["counts"]["train"]["base_episodes"], 24)
        self.assertEqual(self.manifest["counts"]["validation"]["base_episodes"], 8)
        a = {r["supervision"]["source"]["group_id"] for r in self.rows["train"]}
        b = {r["supervision"]["source"]["group_id"] for r in self.rows["validation"]}
        self.assertFalse(a & b)

    def test_02_every_fixed_prompt_covers_every_physical_candidate_exactly_once(self):
        for split in self.rows:
            for group in self.groups[split]:
                rows = [self.by_id[split][sid] for sid in group["sample_ids"]]
                valid = {o["instance_id"] for o in rows[0]["public"]["scene"]["objects"] if o["object_valid"]}
                targets = [r["private_labels"]["labels"]["intent"]["candidate_ids"][0] for r in rows]
                self.assertEqual(set(targets), valid)
                self.assertEqual(len(targets), len(valid))
                self.assertEqual(len({r["supervision"]["language"]["prompt"] for r in rows}), 1)
                self.assertEqual(len({r["supervision"]["language"]["target"] for r in rows}), len(valid))
                self.assertEqual(len(group["pairs"]), len(valid) * (len(valid) - 1) // 2)
                self.assertTrue(group["all_candidate_targets_covered"])

    def test_03_same_prompt_pairs_change_evidence_and_answer(self):
        import torch
        from bci_vla.neural_interaction_v4.tensorizer import collate_public
        for group in self.groups["validation"]:
            pair = group["pairs"][0]
            a, b = [self.by_id["validation"][pair[side]] for side in ("first", "second")]
            self.assertEqual(a["supervision"]["language"]["prompt"], b["supervision"]["language"]["prompt"])
            self.assertNotEqual(a["supervision"]["language"]["target"], b["supervision"]["language"]["target"])
            self.assertEqual(a["public"]["scene"], b["public"]["scene"])
            x, _ = collate_public([a["public"]])
            y, _ = collate_public([b["public"]])
            self.assertTrue(any(not torch.equal(x[k], y[k]) for k in x))

    def test_04_joint_alias_pairs_preserve_all_connector_tensors_and_physical_pointer(self):
        import torch
        from bci_vla.neural_interaction_v4.tensorizer import collate_public
        for pair in self.aliases["validation"]:
            a, b = [self.by_id["validation"][pair[side]] for side in ("first", "second")]
            x, _ = collate_public([a["public"]])
            y, _ = collate_public([b["public"]])
            self.assertTrue(all(torch.equal(x[k], y[k]) for k in x))
            self.assertEqual(a["private_labels"]["labels"]["intent"], b["private_labels"]["labels"]["intent"])
            self.assertNotEqual(a["supervision"]["language"]["prompt"], b["supervision"]["language"]["prompt"])
            self.assertNotEqual(a["supervision"]["language"]["target"], b["supervision"]["language"]["target"])
            self.assertEqual(a["public"]["history"], b["public"]["history"])
            self.assertEqual(a["public"]["public_state"], b["public"]["public_state"])

    def test_05_family_alias_amounts_and_validation_targets_are_balanced(self):
        for split in self.rows:
            counts = self.manifest["counts"][split]
            self.assertEqual(counts["joint_alias_pairs_by_family"][BINDING_FAMILIES[0]],
                             counts["joint_alias_pairs_by_family"][BINDING_FAMILIES[1]])
            self.assertEqual(counts["binding_language_targets_by_family"][BINDING_FAMILIES[0]],
                             counts["binding_language_targets_by_family"][BINDING_FAMILIES[1]])
            self.assertEqual(counts["alias_group_size_distribution"]["2"], counts["joint_alias_pairs"])
            for families in counts["physical_target_counts_by_episode_and_family_audit_only"].values():
                for physical_counts in families.values():
                    self.assertEqual(len(set(physical_counts.values())), 1)
        targets = self.manifest["counts"]["validation"]["binding_language_targets_by_family"]["declared_pick"]
        self.assertGreater(len(targets), 1)
        self.assertLess(max(targets.values()), sum(targets.values()))
        self.assertEqual(sum(targets.values()), 36)

    def test_06_all_episodes_keep_auxiliary_families_and_valid_channel_anchors(self):
        for split, episode_count in (("train", 24), ("validation", 8)):
            aux_by_episode = defaultdict(list)
            for row in self.rows[split]:
                if not row["supervision"]["binding"]["is_binding_row"]:
                    aux_by_episode[row["supervision"]["source"]["base_episode_id"]].append(row)
            self.assertEqual(len(aux_by_episode), episode_count)
            for rows in aux_by_episode.values():
                self.assertEqual(Counter(r["supervision"]["family"] for r in rows), Counter(AUXILIARY_FAMILIES))
                anchor = next(r for r in rows if r["supervision"]["family"] == "noisy_ambiguous")
                self.assertEqual(len(anchor["private_variants"]), 2)
                self.assertNotEqual(anchor["private_variants"][0]["hidden_truth"], anchor["private_variants"][1]["hidden_truth"])
                self.assertEqual(anchor["private_variants"][0]["labels"], anchor["private_variants"][1]["labels"])
                self.assertEqual(anchor["public"]["public_state"]["diagnostics"]["status"], "SUFFICIENT")
                self.assertTrue(anchor["private_labels"]["labels"]["channel"]["mask"])
                self.assertEqual(anchor["private_labels"]["labels"]["main_task"]["mode"], "CLARIFICATION")

    def test_07_too_small_budget_never_truncates_or_writes_partial_data(self):
        target = Path(self.temp.name) / "undersized"
        with self.assertRaisesRegex(ValueError, "no group was truncated"):
            build_balanced_binding_data(self.source, target, {"max_rows": {"train": 128, "validation": 32}})
        self.assertFalse(target.exists())
        with self.assertRaises(FileExistsError):
            build_balanced_binding_data(self.source, self.output)
        with self.assertRaises(ValueError):
            build_balanced_binding_data(self.source, self.source / "forbidden_child")
        with self.assertRaises(ValueError):
            build_balanced_binding_data(self.source, Path(self.temp.name) / "bad_split",
                                        {"max_rows": {"train": 512, "test": 128}})

    def test_08_capture_reconstruction_matches_frozen_jsonl_and_filters_partitions_first(self):
        from bci_vla.training_v4 import balanced_binding_data as module
        original = module._capture_rows
        processed = []
        def checked(captures, split):
            processed.extend(cap["partition"] for cap in captures)
            self.assertTrue(all(cap["partition"] in ("train", "validation") for cap in captures))
            return original(captures, split)
        output = Path(self.temp.name) / "capture"
        with patch.object(module, "_capture_rows", side_effect=checked):
            manifest = build_balanced_binding_data(self.source, output, capture_root=self.source)
        self.assertEqual(set(processed), {"train", "validation"})
        self.assertTrue(manifest["cross_partition_capture_metadata_read"])
        self.assertFalse(manifest["calibration_examples_processed"])
        self.assertFalse(manifest["test_examples_processed"])
        for split in self.rows:
            self.assertEqual(self.rows[split], load_partition(output, split))
            self.assertEqual(self.groups[split], _read_jsonl(output / split / "choice_groups.jsonl"))
            self.assertEqual(self.aliases[split], _read_jsonl(output / split / "joint_alias_pairs.jsonl"))

    def test_09_jsonl_mode_does_not_read_calibration_test_or_future_probe_files(self):
        from bci_vla.training_v4 import balanced_binding_data as module
        original = module._read_jsonl
        paths = []
        def tracked(path):
            paths.append(Path(path).relative_to(self.source).as_posix())
            return original(path)
        with patch.object(module, "_read_jsonl", side_effect=tracked):
            build_balanced_binding_data(self.source, Path(self.temp.name) / "guarded")
        self.assertEqual(set(paths), {s + "/" + n for s in ("train", "validation") for n in ("public_input.jsonl", "private_labels.jsonl")})

    def test_10_public_teacher_independent_of_private_hidden_goal(self):
        publics = _read_jsonl(self.source / "train/public_input.jsonl")
        labels = defaultdict(list)
        for label in _read_jsonl(self.source / "train/private_labels.jsonl"):
            labels[label["sample_id"]].append(label)
        initial, noisy = _initial_and_noisy_sources(publics, labels)[0]
        hashes = {p["scene"]["image_ref"]: _sha(self.source / p["scene"]["image_ref"]) for p in (initial, noisy)}
        original = build_episode_rows(initial, noisy, labels, "train", episode_position=0, scene_episode_position=0, image_hashes=hashes)
        changed = deepcopy(labels)
        for variants in changed.values():
            for label in variants:
                label["hidden_truth"]["goal"] = None
                label["hidden_truth"]["channel_injected_mode"] = "private-change"
        alternate = build_episode_rows(initial, noisy, changed, "train", episode_position=0, scene_episode_position=0, image_hashes=hashes)
        a = [r for r in original[0] if r["supervision"]["family"] != "noisy_ambiguous"]
        b = [r for r in alternate[0] if r["supervision"]["family"] != "noisy_ambiguous"]
        self.assertEqual(a, b)
        self.assertEqual(original[1:], alternate[1:])

    def test_11_sidecar_pair_alias_ids_are_not_model_features(self):
        import torch
        from bci_vla.neural_interaction_v4.tensorizer import collate_public
        row = next(r for r in self.rows["validation"] if r["supervision"]["binding"]["is_binding_row"])
        changed = deepcopy(row)
        changed["supervision"]["binding"] = {"is_binding_row": True, "choice_group_id": "private-id-change",
                                             "source_position_audit_only": 987654}
        x, metadata = collate_public([row["public"]])
        y, changed_metadata = collate_public([changed["public"]])
        self.assertTrue(all(torch.equal(x[k], y[k]) for k in x))
        self.assertEqual(metadata, changed_metadata)
        self.assertFalse(any("alias" in key or "pair" in key or "target" in key for key in x))
        self.assertNotIn("choice_group", row["supervision"]["language"]["prompt"])
        self.assertNotIn("selected object:", row["supervision"]["language"]["prompt"].lower())

    def test_12_written_manifest_binds_all_rows_groups_and_actual_images(self):
        for entry in self.manifest["files"]:
            self.assertEqual(_sha(self.output / entry["path"]), entry["sha256"])
        for split in self.rows:
            for row in self.rows[split]:
                self.assertTrue(validate_example(row))
                self.assertEqual(row["supervision"]["public_input_sha256"], canonical_digest(row["public"]))
                self.assertFalse(row["public"]["provenance"]["real_eeg"])
                if row["supervision"]["family"] == "next_pick_symbolic":
                    self.assertFalse(row["supervision"]["language"]["visual"])
                self.assertFalse(row["supervision"]["language"]["target"].startswith("Place "))
        self.assertFalse(self.manifest["general_hidden_intent_supervision_ready"])
        self.assertFalse(self.manifest["place_phase_supervision_ready"])


if __name__ == "__main__":
    unittest.main()
