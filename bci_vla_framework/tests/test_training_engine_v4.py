"""CPU contract/mechanics tests; no checkpoint, EEG or LIBERO success claims.

The pre-existing public-contract fixture supplies legal tensorizer inputs. The
tiny reference attention backend only checks real differentiable optimizer
updates through the production prefix adapter, not PaliGemma model quality.
"""
import copy
import json
from collections import Counter
from pathlib import Path
import unittest

import torch

from bci_vla.interaction_contract_v4.validation import canonical_digest
from bci_vla.neural_interaction_v4.config import ModuleConfig
from bci_vla.neural_interaction_v4.model import IntentChannelConnector
from bci_vla.neural_interaction_v4.tensorizer import collate_public
from bci_vla.prefix_integration_v4.adapter import Pi05HighLevelPrefix
from bci_vla.training_v4.engine import (
    backward_and_step, compute_training_losses, episode_balanced_schedule,
    parameter_change_audit, parameter_snapshot, scene_group_mean,
    validation_selection,
)
from tests.test_prefix_integration_v4 import ReferencePi05

ROOT = Path(__file__).resolve().parents[1]


def mask_original_labels(label):
    label = copy.deepcopy(label)
    label['labels']['intent'] = {'mode': 'NONE', 'candidate_ids': [], 'outside': False, 'mask': False}
    label['labels']['channel'] = {'mask': False, 'next_probe_success': None, 'target_time_s': None}
    label['labels']['main_task'] = {'mode': 'NONE', 'task': None, 'text': None, 'mask': False}
    label['labels']['interaction']['mask'] = False
    return label


def schedule_row(group, episode, family, split='train'):
    return {'supervision': {'split': split, 'source': {'group_id': group, 'base_episode_id': episode}, 'family': family}}


class TrainingEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        publics = json.loads((ROOT / 'execution/v4/06_fixtures/public.json').read_text(encoding='utf-8'))
        labels = json.loads((ROOT / 'execution/v4/06_fixtures/private.json').read_text(encoding='utf-8'))
        cls.public = next(p for p in publics if p['public_state']['diagnostics']['status'] == 'SUFFICIENT'
                          and not p['public_state']['pending_question'] and not p['public_state']['stop_latched'])
        cls.label = next(label for label in labels if label['sample_id'] == cls.public['sample_id'])
        cls.inputs, cls.metadata = collate_public([cls.public])

    def setUp(self):
        torch.manual_seed(808)
        self.model = IntentChannelConnector(ModuleConfig(hidden_dim=32, heads=4, fusion_blocks=1,
                                                         ffn_dim=64, vlm_dim=2048)).eval()
        self.row = {
            'public': copy.deepcopy(self.public), 'private_labels': mask_original_labels(self.label),
            'supervision': {
                'sample_id': self.public['sample_id'], 'public_input_sha256': canonical_digest(self.public),
                'split': 'train', 'family': 'interface_only',
                'source': {'sample_id': self.public['sample_id'], 'base_episode_id': 'fixture-episode', 'group_id': 'fixture-scene'},
                'auxiliary': {'missing_slots': [True, False, False], 'missing_slots_mask': [True, False, False],
                              'abstain': False, 'abstain_mask': False},
                'language': {'mask': False},
            },
        }

    def test_masked_targets_do_not_change_loss_or_gradients(self):
        alternate = copy.deepcopy(self.row)
        alternate['supervision']['auxiliary']['missing_slots'] = [True, float('nan'), None]
        alternate['supervision']['auxiliary']['abstain'] = float('nan')
        gradients = []
        values = []
        for row in [self.row, alternate]:
            self.model.zero_grad(set_to_none=True)
            output = self.model(self.inputs)
            losses = compute_training_losses(output, [row], self.metadata)
            losses['total'].backward()
            gradients.append({k: p.grad.clone() for k, p in self.model.named_parameters() if p.grad is not None})
            values.append(float(losses['total'].detach()))
            self.assertEqual(losses['supervised_counts']['missing_slots'], 1)
            self.assertEqual(losses['supervised_counts']['abstain'], 0)
            self.assertTrue(torch.equal(self.model.missing_slots.weight.grad[1:], torch.zeros_like(self.model.missing_slots.weight.grad[1:])))
        self.assertEqual(values[0], values[1])
        self.assertEqual(set(gradients[0]), set(gradients[1]))
        for key in gradients[0]:
            torch.testing.assert_close(gradients[0][key], gradients[1][key], rtol=0, atol=0)

    def test_private_sidecar_and_tensor_hash_mismatches_are_rejected(self):
        output = self.model(self.inputs)
        for target in ['private', 'sidecar', 'tensor']:
            row, metadata = copy.deepcopy(self.row), copy.deepcopy(self.metadata)
            if target == 'private':
                row['private_labels']['public_input_sha256'] = '0' * 64
            elif target == 'sidecar':
                row['supervision']['public_input_sha256'] = '0' * 64
            else:
                metadata[0]['input_sha256'] = '0' * 64
            with self.assertRaises(ValueError, msg=target):
                compute_training_losses(output, [row], metadata)
            # B2 cannot skip data-binding checks merely because it lacks heads.
            with self.assertRaises(ValueError, msg='B2-' + target):
                compute_training_losses(None, [row], metadata, language_loss=torch.tensor(1., requires_grad=True), auxiliary_enabled=False)

    def test_empty_masks_produce_differentiable_zero_and_no_decay_step(self):
        self.row['supervision']['auxiliary']['missing_slots_mask'] = [False] * 3
        output = self.model(self.inputs)
        loss = compute_training_losses(output, [self.row], self.metadata,
                                       language_loss=torch.tensor(float('nan'), requires_grad=True))
        self.assertEqual(float(loss['total'].detach()), 0.)
        self.assertTrue(loss['total'].requires_grad)
        self.assertEqual(sum(loss['supervised_counts'].values()), 0)
        before = parameter_snapshot(self.model)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.01, weight_decay=.1)
        info = backward_and_step(loss['total'], optimizer, self.model.parameters())
        self.assertFalse(info['stepped'])
        self.assertFalse(parameter_change_audit(self.model, before)['changed'])

    def test_language_mask_is_independent_and_partial_scalar_is_rejected(self):
        rows, metadata = [self.row, copy.deepcopy(self.row)], self.metadata * 2
        for row in rows:
            row['supervision']['language']['mask'] = True
        logits = torch.tensor([2., float('nan')], requires_grad=True)
        loss = compute_training_losses(None, rows, metadata, language_loss=logits,
                                       language_mask=[True, False], auxiliary_enabled=False)
        self.assertEqual(float(loss['total'].detach()), 2.)
        loss['total'].backward()
        torch.testing.assert_close(logits.grad, torch.tensor([1., 0.]))
        self.assertEqual(loss['supervised_counts']['language'], 1)
        self.assertEqual(loss['supervised_counts']['intent'], 0)
        with self.assertRaises(ValueError):
            compute_training_losses(None, rows, metadata, language_loss=torch.tensor(1., requires_grad=True),
                                    language_mask=[True, False], auxiliary_enabled=False)

    def test_caller_cannot_enable_a_masked_language_label(self):
        with self.assertRaises(ValueError):
            compute_training_losses(None, [self.row], self.metadata, language_loss=torch.tensor(1., requires_grad=True),
                                    language_mask=[True], auxiliary_enabled=False)

    def test_frozen_source_lineage_id_can_differ_from_derived_public_id(self):
        self.row['supervision']['source']['sample_id'] = 'original-source-before-augmentation'
        losses = compute_training_losses(self.model(self.inputs), [self.row], self.metadata)
        self.assertTrue(torch.isfinite(losses['total']))

    def test_full_optimizer_step_uses_language_prefix_and_keeps_backbone_frozen(self):
        base = ReferencePi05()
        adapter = Pi05HighLevelPrefix(base, {}, engineering_only=True, checkpoint_tag='cpu-reference-only')
        base_before = parameter_snapshot(base)
        full_before = parameter_snapshot(self.model)
        output = self.model(self.inputs)
        ids = torch.tensor([[2, 3, 4]])
        pack = adapter.prepare([], [], ids, torch.ones_like(ids, dtype=torch.bool),
                               soft_tokens=output['prefix'], enabled=True)
        target = torch.tensor([[7, 9, 1]])
        language = adapter.teacher_forcing(pack, target, torch.ones_like(target, dtype=torch.bool), eos_id=1)['loss']
        self.row['supervision']['language']['mask'] = True
        loss = compute_training_losses(output, [self.row], self.metadata, language_loss=language)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001)
        info = backward_and_step(loss['total'], optimizer, self.model.parameters(), max_grad_norm=.5)
        audit = parameter_change_audit(self.model, full_before, require_trainable_change=True)
        self.assertTrue(info['stepped'])
        self.assertGreater(info['grad_norm_before_clip'], 0.)
        self.assertTrue(any(name.startswith('prefix_projection') for name in audit['trainable_changed']))
        self.assertTrue(any(name.startswith('intent_blocks') for name in audit['trainable_changed']))
        self.assertFalse(parameter_change_audit(base, base_before)['changed'])
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in base.parameters()))
        self.assertEqual(base.action_calls, 0)

    def test_nonfinite_gradient_and_optimizer_scope_rejected(self):
        parameter = torch.nn.Parameter(torch.tensor(0.))
        optimizer = torch.optim.SGD([parameter], lr=.1)
        # sqrt(0) has finite loss but infinite derivative.
        with self.assertRaises(ValueError):
            backward_and_step(parameter.sqrt(), optimizer, [parameter])
        self.assertEqual(float(parameter.detach()), 0.)
        other = torch.nn.Parameter(torch.tensor(1.))
        with self.assertRaises(ValueError):
            backward_and_step(parameter.square(), optimizer, [parameter, other])

    def test_same_seed_common_schedule_balances_episodes_and_families(self):
        rows = [schedule_row('s0', 'e0', 'a') for _ in range(9)]
        rows += [schedule_row('s0', 'e0', 'b'), schedule_row('s1', 'e1', 'a')]
        for method in ['B2', 'B3_aux', 'B4_aux', 'Full']:
            result = episode_balanced_schedule(rows, seed=0, steps=80)
            self.assertEqual(result, episode_balanced_schedule(rows, seed=0, steps=80), method)
            episodes = Counter(rows[i]['supervision']['source']['base_episode_id'] for i in result)
            self.assertEqual(episodes, {'e0': 40, 'e1': 40})
            families = Counter(rows[i]['supervision']['family'] for i in result if rows[i]['supervision']['source']['base_episode_id'] == 'e0')
            self.assertEqual(families, {'a': 20, 'b': 20})
        self.assertNotEqual(result, episode_balanced_schedule(rows, seed=1, steps=80))

    def test_validation_selection_balances_scenes_without_duplicates(self):
        rows = [schedule_row('s0', 'e0', 'a', 'validation') for _ in range(20)]
        rows += [schedule_row('s1', 'e1', 'a', 'validation') for _ in range(5)]
        rows += [schedule_row('s2', 'e2', 'a', 'validation') for _ in range(5)]
        rows += [schedule_row('train-scene', 'train-episode', 'a')]
        result = validation_selection(rows, seed=9, max_examples=9)
        self.assertEqual(result, validation_selection(rows, seed=9, max_examples=9))
        self.assertEqual(len(set(result)), 9)
        counts = Counter(rows[i]['supervision']['source']['group_id'] for i in result)
        self.assertEqual(counts, {'s0': 3, 's1': 3, 's2': 3})
        selected = [rows[i] for i in result]
        self.assertEqual(scene_group_mean([1.] * 9, selected)['mean'], 1.)
        # Scene averaging differs from micro-averaging when scene sizes differ.
        self.assertEqual(scene_group_mean([0., 0., 0., 10.], rows[:3] + rows[20:21])['mean'], 5.)

    def test_sealed_test_and_calibration_partitions_never_enter_schedule_or_loss(self):
        for split in ['test', 'calibration_fit', 'calibration_audit']:
            sealed = schedule_row('sealed', 'sealed', 'a', split)
            with self.assertRaises(ValueError):
                episode_balanced_schedule([sealed])
            with self.assertRaises(ValueError):
                validation_selection([sealed])
            row = copy.deepcopy(self.row)
            row['supervision']['split'] = split
            with self.assertRaises(ValueError):
                compute_training_losses(self.model(self.inputs), [row], self.metadata)


if __name__ == '__main__':
    unittest.main(verbosity=2)
