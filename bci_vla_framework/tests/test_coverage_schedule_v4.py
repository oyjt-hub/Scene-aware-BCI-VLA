"""Full-row development schedule mechanics; no model performance assertions."""
from copy import deepcopy
import unittest

from bci_vla.training_v4.coverage_schedule import coverage_schedule


def rows(count=432):
    return [{'supervision': {'split': 'train', 'sample_id': 'sample-' + str(index),
                             'binding': {'choice_group_id': 'group-' + str(index // 6)}},
             'private_labels': {'hidden_truth': {'goal': index % 3}, 'labels': {'target': index % 5}}}
            for index in range(count)]


class CoverageScheduleTests(unittest.TestCase):
    def test_each_432_row_round_covers_every_row_and_all_72_choice_groups(self):
        data = rows()
        schedule = coverage_schedule(data, 17, 432 * 2 + 37)
        self.assertEqual(len(schedule), 901)
        for start in (0, 432):
            round_indices = schedule[start:start + 432]
            self.assertEqual(set(round_indices), set(range(432)))
            self.assertEqual(len(round_indices), len(set(round_indices)))
            groups = {data[index]['supervision']['binding']['choice_group_id'] for index in round_indices}
            self.assertEqual(len(groups), 72)
        self.assertEqual(len(set(schedule[864:])), 37)

    def test_seed_is_reproducible_and_shorter_schedule_is_identical_prefix(self):
        data = rows()
        first = coverage_schedule(data, 0, 1000)
        self.assertEqual(first, coverage_schedule(data, 0, 1000))
        self.assertEqual(first[:500], coverage_schedule(data, 0, 500))
        self.assertNotEqual(first, coverage_schedule(data, 1, 1000))

    def test_private_truth_and_language_targets_cannot_change_schedule(self):
        data = rows()
        changed = deepcopy(data)
        for row in changed:
            row['private_labels'] = {'arbitrary': 'changed hidden goal and labels'}
            row['supervision']['language'] = {'target': 'changed answer', 'mask': False}
        self.assertEqual(coverage_schedule(data, 3, 900), coverage_schedule(changed, 3, 900))
        # Sampler does not even require private payloads to exist.
        stripped = [{'supervision': {'split': 'train'}} for _ in data]
        self.assertEqual(coverage_schedule(data, 3, 900), coverage_schedule(stripped, 3, 900))

    def test_nontrain_rows_are_rejected_even_for_zero_steps(self):
        for split in ('validation', 'calibration_fit', 'calibration_audit', 'test', 'test_sealed'):
            data = rows(3)
            data[-1]['supervision']['split'] = split
            for steps in (0, 1, 10):
                with self.assertRaises(ValueError):
                    coverage_schedule(data, 0, steps)

    def test_zero_steps_does_not_mutate_data_and_invalid_inputs_are_rejected(self):
        data = rows(3)
        before = deepcopy(data)
        self.assertEqual(coverage_schedule(data, 0, 0), [])
        self.assertEqual(data, before)
        for data, seed, steps in (([], 0, 0), ([{}], 0, 0), ([None], 0, 1),
                                  (rows(3), 0, -1), (rows(3), True, 1),
                                  (rows(3), 0, True), (rows(3), 0, 1.5)):
            with self.assertRaises(ValueError):
                coverage_schedule(data, seed, steps)


if __name__ == '__main__':
    unittest.main(verbosity=2)
