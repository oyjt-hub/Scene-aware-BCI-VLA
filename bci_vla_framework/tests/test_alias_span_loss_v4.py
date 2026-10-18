"""Loss-only offset controls using deliberately context-sensitive fake proto."""
from dataclasses import dataclass
from types import SimpleNamespace
import unittest

from bci_vla.training_v4.alias_loss import alias_bonus_coefficients, alias_span_weights


@dataclass
class Piece:
    id: int
    begin: int
    end: int


class Tokenizer:
    def __init__(self, text, pieces):
        self.text, self.pieces, self.calls = text, pieces, []

    def encode_as_immutable_proto(self, text):
        self.calls.append(text)
        return SimpleNamespace(text=self.text, pieces=self.pieces)

    def encode(self, _text, **_kwargs):
        raise AssertionError("standalone alias encoding is not permitted")


class AliasSpanLossTests(unittest.TestCase):
    def test_context_tokens_differ_from_standalone_alias_and_EOS_stays_one(self):
        text = "Pick up OBJ_1."
        pieces = [Piece(10, 0, 4), Piece(11, 4, 7), Piece(99, 7, 11),
                  Piece(13, 11, 12), Piece(14, 12, 13), Piece(15, 13, 14)]
        tokenizer = Tokenizer(text, pieces)
        self.assertEqual(alias_span_weights(tokenizer, text, [10, 11, 99, 13, 14, 15, 2], eos_id=2),
                         [1., 1., 4., 4., 4., 1., 1.])
        self.assertEqual(tokenizer.calls, [text])

    def test_multiple_aliases_with_different_piece_widths_use_real_character_spans(self):
        text = "OBJ_12 beside OBJ_3."
        pieces = [Piece(20, 0, 3), Piece(21, 3, 4), Piece(22, 4, 6),
                  Piece(23, 6, 14), Piece(24, 14, 19), Piece(25, 19, 20)]
        tokenizer = Tokenizer(text, pieces)
        self.assertEqual(alias_span_weights(tokenizer, text, [p.id for p in pieces] + [2], alias_weight=7., eos_id=2),
                         [7., 7., 7., 1., 7., 1., 1.])

    def test_no_alias_gives_all_one_weights(self):
        tokenizer = Tokenizer("Which task?", [Piece(30, 0, 5), Piece(31, 5, 10), Piece(32, 10, 11)])
        self.assertEqual(alias_span_weights(tokenizer, "Which task?", [30, 31, 32, 2], eos_id=2), [1.] * 4)

    def test_wrong_actual_tokenization_or_terminal_EOS_is_rejected(self):
        tokenizer = Tokenizer("OBJ_1", [Piece(40, 0, 3), Piece(41, 3, 4), Piece(42, 4, 5)])
        for ids in ([400, 41, 42, 2], [40, 41, 42], [40, 41, 42, 3], [[40, 41, 42, 2]]):
            with self.assertRaises(ValueError):
                alias_span_weights(tokenizer, "OBJ_1", ids, eos_id=2)

    def test_invalid_or_incomplete_offsets_and_invalid_weights_fail_closed(self):
        for pieces in (
            [Piece(50, 0, 2), Piece(51, 3, 5)],
            [Piece(50, -1, 3), Piece(51, 3, 5)],
            [Piece(50, 0, 3), Piece(51, 3, 6)],
            [Piece(50, 3, 5), Piece(51, 0, 3)],
            [Piece(50, 0, 0), Piece(51, 5, 5)],
        ):
            with self.assertRaises(ValueError):
                alias_span_weights(Tokenizer("OBJ_1", pieces), "OBJ_1", [50, 51, 2], eos_id=2)
        tokenizer = Tokenizer("OBJ_1", [Piece(50, 0, 5)])
        for weight in (float("nan"), float("inf"), 0., -1., True):
            with self.assertRaises(ValueError):
                alias_span_weights(tokenizer, "OBJ_1", [50, 2], eos_id=2, alias_weight=weight)

    def test_additive_bonus_preserves_every_nonalias_baseline_coefficient(self):
        text = "Pick up OBJ_1."
        pieces = [Piece(10, 0, 4), Piece(11, 4, 7), Piece(99, 7, 11),
                  Piece(13, 11, 12), Piece(14, 12, 13), Piece(15, 13, 14)]
        tokenizer = Tokenizer(text, pieces)
        coefficients = alias_bonus_coefficients(tokenizer, text, [p.id for p in pieces] + [2], beta=2., eos_id=2)
        for index in (0, 1, 5, 6):
            self.assertEqual(coefficients[index], 1 / 7)
        for index in (2, 3, 4):
            self.assertEqual(coefficients[index], 1 / 7 + 2 / 3)
        self.assertAlmostEqual(sum(coefficients), 3.)
        # Context-sensitive full-answer IDs are used once; no encode(alias).
        self.assertEqual(tokenizer.calls, [text])

    def test_additive_bonus_equals_mean_CE_plus_alias_only_mean(self):
        text = "Pick up OBJ_1."
        pieces = [Piece(10, 0, 4), Piece(11, 4, 7), Piece(99, 7, 11),
                  Piece(13, 11, 12), Piece(14, 12, 13), Piece(15, 13, 14)]
        coefficients = alias_bonus_coefficients(Tokenizer(text, pieces), text, [p.id for p in pieces] + [2], beta=.4, eos_id=2)
        token_losses = [.5, 4., 9., 1., 3., .25, 2.]
        actual = sum(loss * coefficient for loss, coefficient in zip(token_losses, coefficients))
        expected = sum(token_losses) / 7 + .4 * sum(token_losses[2:5]) / 3
        self.assertAlmostEqual(actual, expected)

    def test_additive_bonus_without_alias_is_unchanged_mean_CE(self):
        text = "Which task?"
        tokenizer = Tokenizer(text, [Piece(30, 0, 5), Piece(31, 5, 10), Piece(32, 10, 11)])
        coefficients = alias_bonus_coefficients(tokenizer, text, [30, 31, 32, 2], beta=100., eos_id=2)
        self.assertEqual(coefficients, [.25] * 4)
        self.assertEqual(sum(coefficients), 1.)

    def test_zero_bonus_is_baseline_and_still_validates_complete_proto(self):
        text = "OBJ_1"
        pieces = [Piece(40, 0, 3), Piece(41, 3, 4), Piece(42, 4, 5)]
        coefficients = alias_bonus_coefficients(Tokenizer(text, pieces), text, [40, 41, 42, 2], beta=0., eos_id=2)
        self.assertEqual(coefficients, [.25] * 4)
        with self.assertRaises(ValueError):
            alias_bonus_coefficients(Tokenizer(text, pieces), text, [400, 41, 42, 2], beta=0., eos_id=2)
        with self.assertRaises(ValueError):
            alias_bonus_coefficients(Tokenizer(text, [Piece(40, 0, 2), Piece(41, 3, 5)]), text,
                                     [40, 41, 2], beta=0., eos_id=2)

    def test_invalid_additive_bonus_is_rejected(self):
        for beta in (float('nan'), float('inf'), -1., True, '1'):
            with self.assertRaises(ValueError):
                alias_bonus_coefficients(Tokenizer("OBJ_1", [Piece(50, 0, 5)]), "OBJ_1", [50, 2], beta=beta, eos_id=2)


if __name__ == "__main__":
    unittest.main()
