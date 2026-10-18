"""Public-only supervision, ambiguity, source preservation, and actual CPU binding."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bci_vla.interaction_contract_v4.validation import canonical_digest
from bci_vla.training_v4.data import (
    FAMILIES, PROTOCOL, baseline_evidence_text, build_training_data,
    declared_teacher, language_prompt, load_partition, make_declared_public,
    make_example, validate_example,
)

ROOT = Path(__file__).resolve().parents[1]


class TrainingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="bci_vla_step08_data_")
        cls.output = Path(cls.temp.name) / "pilot"
        cls.source = ROOT / "data/v4"
        cls.source_hashes = {
            (split, name): hashlib.sha256((cls.source / split / name).read_bytes()).hexdigest()
            for split in ("train", "validation")
            for name in ("public_input.jsonl", "private_labels.jsonl")
        }
        cls.manifest = build_training_data(cls.source, cls.output)
        cls.rows = {split: load_partition(cls.output, split) for split in ("train", "validation")}
        cls.seed_row = next(row for row in cls.rows["train"] if row["supervision"]["family"] == "noisy_ambiguous")
        # Declaration source must be an actual initial frame even though noisy
        # counterfactual anchors now preserve later-frame diagnostic supervision.
        cls.initial = next(json.loads(line) for line in (cls.source / "train/public_input.jsonl").open(encoding="utf-8")
                           if json.loads(line)["context"]["session_id"] == cls.seed_row["public"]["context"]["session_id"]
                           and json.loads(line)["scene"]["observed_at_s"] == 0)
        cls.initial_variants = [json.loads(line) for line in (cls.source / "train/private_labels.jsonl").open(encoding="utf-8")
                                if json.loads(line)["sample_id"] == cls.initial["sample_id"]]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_01_source_data_unchanged_balanced_bounded(self):
        for (split, name), before in self.source_hashes.items():
            self.assertEqual(before, hashlib.sha256((self.source / split / name).read_bytes()).hexdigest())
        for split, count in (("train", 128), ("validation", 32)):
            rows = self.rows[split]
            self.assertEqual(len(rows), count)
            counts = Counter(row["supervision"]["source"]["base_episode_id"] for row in rows)
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
            self.assertEqual(set(Counter(row["supervision"]["family"] for row in rows)), set(FAMILIES))
        train_groups = {r["supervision"]["source"]["group_id"] for r in self.rows["train"]}
        val_groups = {r["supervision"]["source"]["group_id"] for r in self.rows["validation"]}
        self.assertFalse(train_groups & val_groups)

    def test_02_append_only_and_sealed_split_rejection(self):
        with self.assertRaises(FileExistsError):
            build_training_data(self.source, self.output)
        with self.assertRaises(ValueError):
            build_training_data(self.source, self.source / "new_data")
        for split in ("calibration_fit", "calibration_audit", "test", "test_sealed"):
            with self.assertRaises(ValueError):
                load_partition(self.output, split)
        with self.assertRaises(ValueError):
            build_training_data(self.source, Path(self.temp.name) / "oversized", {"budgets": {"train": 129, "validation": 32}})

    def test_03_real_counterfactuals_remain_ambiguous_and_weighted_once(self):
        for split in self.rows:
            anchors = [r for r in self.rows[split] if r["supervision"]["family"] == "noisy_ambiguous"]
            self.assertTrue(anchors)
            self.assertEqual(len({r["public"]["sample_id"] for r in self.rows[split]}), len(self.rows[split]))
            for row in anchors:
                self.assertEqual(len(row["private_variants"]), 2)
                self.assertEqual(row["private_variants"][0]["labels"], row["private_variants"][1]["labels"])
                self.assertNotEqual(row["private_variants"][0]["hidden_truth"], row["private_variants"][1]["hidden_truth"])
                self.assertEqual(row["private_labels"]["identifiability"]["intent"], "AMBIGUOUS")
                self.assertEqual(row["private_labels"]["labels"]["main_task"]["mode"], "CLARIFICATION")

    def test_04_public_slot_targets_are_specific(self):
        expected = {
            "declared_pick": ([False, False, False], False, "Pick up "),
            "missing_action": ([True, False, False], True, "Which task do you want?"),
            "missing_source": ([False, True, False], True, "Which object do you mean?"),
            "conflicting_declarations": ([False, True, False], True, "Which object do you mean?"),
            "no_response": ([False, True, False], True, "Which object do you mean?"),
            "missing_destination": ([False, False, True], True, "Where should I place it?"),
            "next_pick_symbolic": ([False, False, False], False, "Pick up "),
        }
        for row in self.rows["train"]:
            family = row["supervision"]["family"]
            if family not in expected:
                continue
            slots, abstain, text = expected[family]
            aux = row["supervision"]["auxiliary"]
            self.assertEqual(aux["missing_slots"], slots, family)
            self.assertEqual(aux["missing_slots_mask"], [True] * 3)
            self.assertEqual(aux["abstain"], abstain)
            self.assertTrue(row["supervision"]["language"]["target"].startswith(text))

    def test_05_teacher_never_uses_hidden_truth(self):
        first = make_example(self.initial, self.initial_variants, "train", "declared_pick", 0)
        changed = deepcopy(self.initial_variants)
        for variant in changed:
            variant["hidden_truth"]["goal"] = None
            variant["hidden_truth"]["channel_injected_mode"] = "arbitrary-private-change"
        second = make_example(self.initial, changed, "train", "declared_pick", 0)
        self.assertEqual(first, second)
        self.assertEqual(declared_teacher(first["public"])["text"], first["supervision"]["language"]["target"])

    def test_06_target_changes_only_in_public_evidence_not_VLM_prompt(self):
        a = make_example(self.initial, self.initial_variants, "train", "declared_pick", 0)
        b = make_example(self.initial, self.initial_variants, "train", "declared_pick", 1)
        self.assertNotEqual(a["private_labels"]["labels"]["intent"]["candidate_ids"], b["private_labels"]["labels"]["intent"]["candidate_ids"])
        self.assertNotEqual(a["supervision"]["language"]["target"], b["supervision"]["language"]["target"])
        self.assertEqual(a["supervision"]["language"]["prompt"], b["supervision"]["language"]["prompt"])
        self.assertNotEqual(baseline_evidence_text(a["public"]), baseline_evidence_text(b["public"]))
        prompt = a["supervision"]["language"]["prompt"]
        self.assertNotIn("Selected object:", prompt)
        self.assertNotIn(a["public"]["sample_id"], prompt)
        self.assertNotIn("hidden_truth", baseline_evidence_text(a["public"]))
        self.assertNotIn("session-", baseline_evidence_text(a["public"]))

    def test_07_alias_rebinding_propagates_answer_and_dictionary(self):
        public = make_declared_public(self.initial, "declared_pick", 0)
        before = declared_teacher(public)["text"]
        mapping = {obj["alias"]: "OBJ_" + str(15 - i) for i, obj in enumerate(public["scene"]["objects"])}
        for obj in public["scene"]["objects"]:
            obj["alias"] = mapping[obj["alias"]]
        after = declared_teacher(public)["text"]
        self.assertNotEqual(before, after)
        self.assertEqual(after, before.replace(before.split()[2].rstrip("."), mapping[before.split()[2].rstrip(".")]))
        self.assertIn(mapping["OBJ_0"], language_prompt(public, PROTOCOL))

    def test_08_bindings_reject_label_prompt_and_hash_corruption(self):
        seed = deepcopy(self.seed_row)
        seed["supervision"]["public_input_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_example(seed)
        seed = deepcopy(self.seed_row)
        seed["supervision"]["language"]["prompt"] += " Selected object: OBJ_0."
        with self.assertRaises(ValueError):
            validate_example(seed)
        seed = deepcopy(self.seed_row)
        seed["private_variants"][1]["labels"]["main_task"]["text"] = "Pick up OBJ_0."
        with self.assertRaises(ValueError):
            validate_example(seed)
        declaration = deepcopy(next(row for row in self.rows["train"] if row["supervision"]["family"] == "declared_pick"))
        declaration["supervision"]["auxiliary"]["missing_slots"][0] = True
        with self.assertRaises(ValueError):
            validate_example(declaration)

    def test_09_no_fake_visual_place_or_gripper_inference(self):
        sub = [r for r in self.rows["train"] if r["supervision"]["language"]["kind"] == "next_subtask"]
        self.assertTrue(sub)
        for row in sub:
            language = row["supervision"]["language"]
            self.assertFalse(language["visual"])
            self.assertEqual(language["gripper_state"], "empty")
            self.assertEqual(language["gripper_provenance"], "declared_symbolic_intervention_not_RGB_inference")
            self.assertIsNotNone(row["public"]["public_state"]["confirmed_task"])
        for row in self.rows["train"] + self.rows["validation"]:
            self.assertFalse(row["supervision"]["language"]["target"].startswith("Place "))
            self.assertFalse(row["public"]["provenance"]["real_eeg"])
            self.assertTrue(all(not o["semantic_valid"] for o in row["public"]["scene"]["objects"]))
        self.assertFalse(self.manifest["place_phase_supervision_ready"])

    def test_10_only_train_validation_public_private_and_used_images_are_read(self):
        from bci_vla.training_v4 import data
        read_paths = []
        original_read = data._read_jsonl
        def tracked(path):
            read_paths.append(Path(path).relative_to(self.source).as_posix())
            return original_read(path)
        with patch.object(data, "_read_jsonl", side_effect=tracked):
            # Track just source reads; output loader is separately allowed.
            with patch.object(data, "load_partition", return_value=[]):
                manifest = build_training_data(self.source, Path(self.temp.name) / "guarded", {"budgets": {"train": 24, "validation": 8}})
        self.assertEqual(set(read_paths), {s + "/" + n for s in ("train", "validation") for n in ("public_input.jsonl", "private_labels.jsonl")})
        self.assertFalse(manifest["calibration_accessed"])
        self.assertFalse(manifest["formal_test_accessed"])
        self.assertFalse(manifest["future_probe_files_accessed"])
        self.assertTrue(all(not any(x in f["path"] for x in ("calibration", "test", "future_probe", "manifest")) for f in manifest["source_files_read"]))

    def test_11_actual_CPU_tensorization_and_nontrivial_auxiliary_gradient(self):
        import torch
        from bci_vla.neural_interaction_v4.model import IntentChannelConnector
        from bci_vla.neural_interaction_v4.tensorizer import collate_public
        from bci_vla.neural_interaction_v4.losses import auxiliary_losses
        torch.set_num_threads(2)
        selected = []
        for family in FAMILIES:
            selected.append(next(r for r in self.rows["train"] if r["supervision"]["family"] == family))
        x, metadata = collate_public([r["public"] for r in selected])
        model = IntentChannelConnector()
        y = model(x)
        loss = auxiliary_losses(y, [r["public"] for r in selected], [r["private_labels"] for r in selected], metadata)
        targets = torch.tensor([r["supervision"]["auxiliary"]["missing_slots"] for r in selected], dtype=torch.float32)
        masks = torch.tensor([r["supervision"]["auxiliary"]["missing_slots_mask"] for r in selected], dtype=torch.bool)
        missing = torch.nn.functional.binary_cross_entropy_with_logits(y["missing_slot_logits"][masks], targets[masks])
        abstain = torch.nn.functional.binary_cross_entropy_with_logits(y["abstain_logit"], torch.tensor([r["supervision"]["auxiliary"]["abstain"] for r in selected], dtype=torch.float32))
        total = loss["total"] + missing + abstain
        self.assertTrue(torch.isfinite(total))
        self.assertGreater(float(loss["intent"].detach()), 0)
        total.backward()
        self.assertGreater(float(model.missing_slots.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.abstain.weight.grad.abs().sum()), 0)
        self.assertFalse(any("hidden" in key for key in x))

    def test_12_capture_reconstruction_matches_frozen_public_and_only_processes_allowed_splits(self):
        from bci_vla.training_v4 import data
        original = data._capture_rows
        partitions = []
        def tracked(captures, split):
            partitions.extend(cap["partition"] for cap in captures)
            self.assertTrue(all(cap["partition"] in ("train", "validation") for cap in captures))
            return original(captures, split)
        output = Path(self.temp.name) / "capture_pilot"
        with patch.object(data, "_capture_rows", side_effect=tracked):
            manifest = build_training_data(self.source, output, capture_root=self.source)
        self.assertEqual(set(partitions), {"train", "validation"})
        self.assertTrue(manifest["cross_partition_capture_metadata_read"])
        self.assertFalse(manifest["calibration_examples_processed"])
        self.assertFalse(manifest["test_examples_processed"])
        for split in ("train", "validation"):
            self.assertEqual(self.rows[split], load_partition(output, split))

    def test_13_channel_supervision_has_public_support_and_never_enters_prompt(self):
        for split in ("train", "validation"):
            channel_rows = [r for r in self.rows[split] if r["private_labels"]["labels"]["channel"]["mask"]]
            self.assertTrue(channel_rows)
            for row in channel_rows:
                self.assertEqual(row["public"]["public_state"]["diagnostics"]["status"], "SUFFICIENT")
                self.assertGreater(row["private_labels"]["labels"]["channel"]["target_time_s"], row["public"]["as_of_s"])
                self.assertNotIn("next_probe_success", row["supervision"]["language"]["prompt"])
                self.assertNotIn("next_probe_success", baseline_evidence_text(row["public"]))


if __name__ == "__main__":
    unittest.main()
