"""Public-reference binding mechanics only; no model/robot success claims."""
from copy import deepcopy
import json
import math
from pathlib import Path
import unittest

import torch
from torch import nn

from bci_vla.neural_interaction_v4.config import ModuleConfig
from bci_vla.neural_interaction_v4.model import IntentChannelConnector
from bci_vla.neural_interaction_v4.tensorizer import collate_public
from bci_vla.prefix_integration_v4.binding import ObjectReferenceBinding, build_reference_bank

ROOT = Path(__file__).resolve().parents[1]


def alias_ids(alias):
    # A contract tokenizer fixture, deliberately without BOS/EOS.
    return [1, 2, int(alias.split('_')[1]) + 3]


class ReferenceBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        publics = json.loads((ROOT / 'execution/v4/06_fixtures/public.json').read_text(encoding='utf-8'))
        cls.public = next(p for p in publics if len(p['scene']['objects']) >= 2)

    def embedding(self, dimension=4):
        module = nn.Embedding(32, dimension)
        with torch.no_grad():
            module.weight.copy_(torch.arange(32 * dimension).reshape(32, dimension).float() / 10)
        return module

    def test_bank_right_aligns_and_scales_native_embedding_exactly_once(self):
        embedding = self.embedding()
        bank = build_reference_bank([self.public], alias_ids, embedding, hidden_size=4)
        self.assertEqual(bank.shape, (1, 16, 4, 4))
        self.assertFalse(bank.requires_grad)
        for index, obj in enumerate(self.public['scene']['objects']):
            ids = torch.tensor(alias_ids(obj['alias']))
            torch.testing.assert_close(bank[0, index, 1:], embedding(ids).detach() * math.sqrt(4), rtol=0, atol=0)
            self.assertTrue(torch.equal(bank[0, index, 0], torch.zeros(4)))
        self.assertEqual(float(bank[0, len(self.public['scene']['objects']):].abs().sum()), 0.)
        p = torch.zeros(1, 16)
        p[0, 0] = 1.
        out = ObjectReferenceBinding(residual_scale=0.)(
            {'prefix': torch.zeros(1, 8, 4), 'candidate_probabilities': p}, bank, p.bool())
        torch.testing.assert_close(out[:, 4:], bank[:, 0], rtol=0, atol=0)

    def test_alias_changes_leave_connector_unchanged_but_change_bound_prefix(self):
        changed = deepcopy(self.public)
        aliases = [obj['alias'] for obj in changed['scene']['objects']]
        for obj, alias in zip(changed['scene']['objects'], aliases[1:] + aliases[:1]):
            obj['alias'] = alias
        a, _ = collate_public([self.public])
        b, _ = collate_public([changed])
        self.assertTrue(all(torch.equal(a[key], b[key]) for key in a))
        torch.manual_seed(9)
        connector = IntentChannelConnector(ModuleConfig(hidden_dim=32, ffn_dim=64, fusion_blocks=1, vlm_dim=16)).eval()
        with torch.no_grad():
            first, second = connector(a), connector(b)
        torch.testing.assert_close(first['prefix'], second['prefix'], rtol=0, atol=0)
        torch.testing.assert_close(first['candidate_probabilities'], second['candidate_probabilities'], rtol=0, atol=0)
        embedding = self.embedding(16)
        original_bank = build_reference_bank([self.public], alias_ids, embedding, hidden_size=16)
        changed_bank = build_reference_bank([changed], alias_ids, embedding, hidden_size=16)
        head = ObjectReferenceBinding()
        original = head(first, original_bank, a['candidate_mask'])
        rebound = head(second, changed_bank, b['candidate_mask'])
        self.assertFalse(torch.equal(original_bank, changed_bank))
        self.assertGreater(float((original[:, 4:] - rebound[:, 4:]).abs().max()), 1e-6)
        torch.testing.assert_close(original[:, :4], first['prefix'][:, 4:], rtol=0, atol=0)
        torch.testing.assert_close(rebound[:, :4], original[:, :4], rtol=0, atol=0)

    def test_outside_mass_is_not_renormalized_or_forced_to_a_candidate(self):
        prefix = torch.zeros(1, 8, 2)
        bank = torch.stack([torch.ones(4, 2), torch.full((4, 2), 3.)])[None]
        output = {'prefix': prefix, 'candidate_probabilities': torch.tensor([[.25, .5]])}
        bound = ObjectReferenceBinding()(output, bank, torch.ones(1, 2, dtype=torch.bool))
        torch.testing.assert_close(bound[:, 4:], torch.full((1, 4, 2), 1.75), rtol=0, atol=0)
        output['candidate_probabilities'].zero_()
        torch.testing.assert_close(ObjectReferenceBinding()(output, bank, torch.ones(1, 2, dtype=torch.bool)), torch.zeros_like(prefix))

    def test_probabilities_and_soft_prefix_receive_gradients_embedding_does_not(self):
        embedding = self.embedding()
        bank = build_reference_bank([self.public], alias_ids, embedding, hidden_size=4)
        prefix = torch.randn(1, 8, 4, requires_grad=True)
        probability = torch.zeros(1, 16)
        probability[0, :2] = torch.tensor([.2, .6])
        probability.requires_grad_()
        mask = torch.zeros(1, 16, dtype=torch.bool)
        mask[:, :2] = True
        result = ObjectReferenceBinding()({'prefix': prefix, 'candidate_probabilities': probability}, bank, mask)
        result[:, 4:].square().mean().backward()
        self.assertGreater(float(probability.grad[:, :2].abs().sum()), 0.)
        self.assertGreater(float(prefix.grad[:, :4].abs().sum()), 0.)
        self.assertEqual(float(prefix.grad[:, 4:].abs().sum()), 0.)
        self.assertIsNone(embedding.weight.grad)
        self.assertFalse(bank.requires_grad)

    def test_diagnostic_stream_never_reads_reference_or_candidate_weights(self):
        prefix = torch.randn(1, 8, 4, requires_grad=True)
        probability = torch.tensor([[.2, .6]], requires_grad=True)
        bank = torch.randn(1, 2, 4, 4, requires_grad=True)
        mask = torch.ones(1, 2, dtype=torch.bool)
        result = ObjectReferenceBinding()({'prefix': prefix, 'candidate_probabilities': probability}, bank, mask)
        torch.testing.assert_close(result[:, :4], prefix[:, 4:], rtol=0, atol=0)
        gradients = torch.autograd.grad(result[:, :4].sum(), [prefix, probability, bank], allow_unused=True)
        self.assertEqual(float(gradients[0][:, :4].abs().sum()), 0.)
        torch.testing.assert_close(gradients[0][:, 4:], torch.ones_like(prefix[:, 4:]))
        self.assertEqual(float(gradients[1].abs().sum()), 0.)
        self.assertIsNone(gradients[2])

    def test_head_uses_only_predictions_not_extra_labels_or_outside_metadata(self):
        output = {'prefix': torch.randn(1, 8, 4), 'candidate_probabilities': torch.tensor([[.7, .2]])}
        bank = torch.randn(1, 2, 4, 4)
        mask = torch.ones(1, 2, dtype=torch.bool)
        first = ObjectReferenceBinding()(output, bank, mask)
        output.update(private_labels={'target': 'arbitrary-object'}, hidden_truth={'goal': 'wrong'}, outside_probability=torch.tensor([.9]))
        second = ObjectReferenceBinding()(output, bank, mask)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(list(ObjectReferenceBinding().parameters()), [])

    def test_reorder_only_is_unscaled_and_needs_neither_bank_nor_probabilities(self):
        prefix = torch.randn(2, 8, 4, requires_grad=True)
        result = ObjectReferenceBinding(mode='reorder_only')(
            {'prefix': prefix}, None, torch.ones(2, 3, dtype=torch.bool))
        torch.testing.assert_close(result, torch.cat([prefix[:, 4:], prefix[:, :4]], 1), rtol=0, atol=0)
        result.sum().backward()
        torch.testing.assert_close(prefix.grad, torch.ones_like(prefix))

    def test_joint_object_row_permutation_preserves_mixture(self):
        prefix = torch.randn(1, 8, 4)
        p = torch.tensor([[.1, .7, .15]])
        bank = torch.randn(1, 3, 4, 4)
        mask = torch.ones(1, 3, dtype=torch.bool)
        head = ObjectReferenceBinding()
        original = head({'prefix': prefix, 'candidate_probabilities': p}, bank, mask)
        permutation = [2, 0, 1]
        changed = head({'prefix': prefix, 'candidate_probabilities': p[:, permutation]}, bank[:, permutation], mask[:, permutation])
        torch.testing.assert_close(original, changed, rtol=1e-6, atol=1e-6)

    def test_dtype_cast_keeps_predicted_weight_gradients(self):
        prefix = torch.randn(1, 8, 4, dtype=torch.bfloat16, requires_grad=True)
        p = torch.tensor([[.2, .7]], requires_grad=True)
        result = ObjectReferenceBinding()({'prefix': prefix, 'candidate_probabilities': p},
                                         torch.ones(1, 2, 4, 4, dtype=torch.bfloat16),
                                         torch.ones(1, 2, dtype=torch.bool))
        self.assertEqual(result.dtype, torch.bfloat16)
        result[:, 4:].float().sum().backward()
        self.assertGreater(float(p.grad.abs().sum()), 0.)

    def test_bank_illegal_inputs_and_long_alias_tokenization_fail_closed(self):
        embedding = self.embedding()
        for ids in ([], [1, 2, 3, 4, 5], [-1], [True], ['1']):
            with self.assertRaises(ValueError):
                build_reference_bank([self.public], lambda alias: ids, embedding, hidden_size=4)
        with self.assertRaises(ValueError):
            build_reference_bank([self.public], alias_ids, embedding, hidden_size=4, max_objects=1)
        with self.assertRaises(ValueError):
            build_reference_bank([self.public], alias_ids, embedding, hidden_size=8)
        private_public = deepcopy(self.public)
        private_public['private_labels'] = {}
        with self.assertRaises(ValueError):
            build_reference_bank([private_public], alias_ids, embedding, hidden_size=4)
        with torch.no_grad():
            embedding.weight.fill_(float('nan'))
        with self.assertRaises(ValueError):
            build_reference_bank([self.public], alias_ids, embedding, hidden_size=4)

    def test_invalid_prefix_mask_bank_or_probability_is_rejected(self):
        head = ObjectReferenceBinding()
        good = {'prefix': torch.zeros(1, 8, 4), 'candidate_probabilities': torch.tensor([[.2, .7]])}
        bank = torch.ones(1, 2, 4, 4)
        mask = torch.ones(1, 2, dtype=torch.bool)
        for probability in (torch.tensor([[-.1, .7]]), torch.tensor([[.8, .7]]), torch.tensor([[float('nan'), .7]])):
            with self.assertRaises(ValueError):
                head({**good, 'candidate_probabilities': probability}, bank, mask)
        with self.assertRaises(ValueError):
            head(good, bank, torch.tensor([[True, False]]))
        with self.assertRaises(ValueError):
            head(good, bank[:, :, :3], mask)
        with self.assertRaises(ValueError):
            head(good, bank * float('nan'), mask)
        with self.assertRaises(ValueError):
            head({**good, 'prefix': good['prefix'][:, :7]}, bank, mask)
        with self.assertRaises(ValueError):
            head(good, bank, mask.long())
        with self.assertRaises(ValueError):
            ObjectReferenceBinding(mode='oracle')
        with self.assertRaises(ValueError):
            ObjectReferenceBinding(reference_scale=float('nan'))

    def test_empty_candidate_scene_keeps_only_intent_residual(self):
        public = json.loads((ROOT / 'examples/v4/empty_scene_and_history.json').read_text(encoding='utf-8'))
        bank = build_reference_bank([public], alias_ids, self.embedding(), hidden_size=4, max_objects=0)
        self.assertEqual(bank.shape, (1, 0, 4, 4))
        prefix = torch.randn(1, 8, 4)
        bound = ObjectReferenceBinding()({'prefix': prefix, 'candidate_probabilities': torch.empty(1, 0)},
                                         bank, torch.empty(1, 0, dtype=torch.bool))
        torch.testing.assert_close(bound[:, :4], prefix[:, 4:], rtol=0, atol=0)
        torch.testing.assert_close(bound[:, 4:], prefix[:, :4] * .1, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
