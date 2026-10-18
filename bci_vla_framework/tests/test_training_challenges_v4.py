"""Challenge generator mechanics on contract fixtures, not model performance."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import torch

from bci_vla.neural_interaction_v4.tensorizer import collate_public
from bci_vla.training_v4.challenges import build_development_challenges
from bci_vla.training_v4.data import make_example, validate_example

ROOT = Path(__file__).resolve().parents[1]


class TrainingChallengeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        publics = json.loads((ROOT / 'execution/v4/06_fixtures/public.json').read_text(encoding='utf-8'))
        private = json.loads((ROOT / 'execution/v4/06_fixtures/private.json').read_text(encoding='utf-8'))
        public = next(p for p in publics if p['scene']['observed_at_s'] == 0 and len(p['scene']['objects']) >= 2)
        labels = [deepcopy(row) for row in private if row['sample_id'] == public['sample_id']]
        # Fixture-only partition binding; this creates no performance dataset.
        for label in labels:
            label['split'] = 'validation'
        cls.rows = [make_example(public, labels, 'validation', family, 1, image_sha256='fixture-image-sha')
                    for family in ('declared_pick', 'next_pick_symbolic')]

    def test_same_native_prompt_different_public_evidence_changes_answer(self):
        result = build_development_challenges(self.rows)
        self.assertEqual(result['counts'], {'source_rows': 2, 'paired_evidence_pairs': 2,
                                           'joint_alias_pairs': 2, 'unique_public_examples': 6})
        for pair in result['paired_evidence']:
            self.assertTrue(pair['invariants']['native_prompt_equal'])
            self.assertFalse(pair['invariants']['connector_tensors_equal'])
            self.assertFalse(pair['invariants']['physical_source_equal'])
            self.assertTrue(pair['invariants']['image_equal'])
            self.assertNotEqual(pair['expected']['first_target'], pair['expected']['second_target'])
            for side in ('first', 'second'):
                self.assertTrue(validate_example(pair[side]))

    def test_joint_alias_preserves_all_connector_tensors_and_physical_source(self):
        for pair in build_development_challenges(self.rows)['joint_alias']:
            first, second = pair['first'], pair['second']
            a, _ = collate_public([first['public']])
            b, _ = collate_public([second['public']])
            self.assertTrue(all(torch.equal(a[key], b[key]) for key in a))
            self.assertTrue(pair['invariants']['physical_source_equal'])
            self.assertFalse(pair['invariants']['native_prompt_equal'])
            self.assertNotEqual(first['supervision']['language']['target'], second['supervision']['language']['target'])
            self.assertTrue(validate_example(second))
            self.assertEqual(first['public']['history'], second['public']['history'])
            for original, changed in zip(first['public']['scene']['objects'], second['public']['scene']['objects']):
                original, changed = deepcopy(original), deepcopy(changed)
                self.assertEqual(pair['alias_mapping'][original.pop('alias')], changed.pop('alias'))
                self.assertEqual(original, changed)

    def test_generation_is_deterministic_and_does_not_mutate_sources(self):
        original = deepcopy(self.rows)
        first = build_development_challenges(self.rows)
        self.assertEqual(self.rows, original)
        self.assertEqual(first, build_development_challenges(self.rows))
        self.assertEqual(build_development_challenges(self.rows, max_source_rows=1)['counts']['source_rows'], 1)

    def test_hidden_goal_does_not_choose_challenge_or_targets(self):
        changed = deepcopy(self.rows)
        for row in changed:
            for label in row['private_variants']:
                label['hidden_truth']['goal'] = None
                label['hidden_truth']['channel_injected_mode'] = 'arbitrary-hidden-state'
            row['private_labels'] = row['private_variants'][0]
        self.assertEqual(build_development_challenges(self.rows), build_development_challenges(changed))

    def test_nonvalidation_partitions_are_rejected(self):
        for split in ('train', 'test', 'test_sealed', 'calibration_fit', 'calibration_audit'):
            rows = deepcopy(self.rows)
            rows[0]['supervision']['split'] = split
            with self.assertRaises(ValueError):
                build_development_challenges(rows)
        with self.assertRaises(ValueError):
            build_development_challenges([])
        with self.assertRaises(ValueError):
            build_development_challenges(self.rows, max_source_rows=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
