"""Locate loss-only alias weights in the COMPLETE real answer tokenization.

An alias encoded alone can have different SentencePiece word-boundary tokens.
Use immutable-proto offsets from the full target and verify exact token IDs
before weighting. This helper never builds model inputs or public references.
"""
from __future__ import annotations

import math
import re


def alias_span_weights(tokenizer, text, target_ids, *, alias_weight=4.0, eos_id):
    """Return float weights for full-target pieces followed by one terminal EOS.

    Target IDs must equal encode_as_immutable_proto(text).pieces.ids plus eos_id
    exactly. EOS and non-alias pieces receive 1. A real piece receives alias_weight
    when its [begin,end) text interval intersects an OBJ_digits character span;
    contextual leading spaces are allowed in that same piece. Every alias
    character must be covered by real, positive-length pieces. Private answer
    text is consumed only at this loss boundary.
    """
    if not isinstance(text, str):
        raise ValueError("target text must be a string")
    if (type(eos_id) is not int or eos_id < 0
            or type(alias_weight) not in (int, float)
            or not math.isfinite(alias_weight) or alias_weight <= 0):
        raise ValueError("nonnegative integer EOS and finite positive alias weight required")
    encode = getattr(tokenizer, "encode_as_immutable_proto", None)
    if not callable(encode):
        raise ValueError("tokenizer must provide complete-target immutable proto")
    if hasattr(target_ids, "tolist"):
        target_ids = target_ids.tolist()
    if (not isinstance(target_ids, (list, tuple)) or not target_ids
            or any(type(token) is not int or token < 0 for token in target_ids)):
        raise ValueError("target IDs must be an unbatched nonnegative integer sequence")
    proto = encode(text)
    if hasattr(proto, "text") and proto.text != text:
        raise ValueError("immutable proto text does not match the complete target")
    try:
        pieces = list(proto.pieces)
        ids = [piece.id for piece in pieces]
    except (AttributeError, TypeError) as exc:
        raise ValueError("immutable proto pieces must expose id/begin/end") from exc
    if any(type(token) is not int or token < 0 for token in ids) or ids + [eos_id] != list(target_ids):
        raise ValueError("complete proto piece IDs plus EOS do not match actual target IDs")
    intervals = []
    previous_begin = 0
    for piece in pieces:
        begin, end = getattr(piece, "begin", None), getattr(piece, "end", None)
        if (type(begin) is not int or type(end) is not int
                or not 0 <= begin <= end <= len(text) or begin < previous_begin):
            raise ValueError("invalid or unordered immutable-proto text offsets")
        intervals.append((begin, end))
        previous_begin = begin
    spans = [(match.start(), match.end()) for match in re.finditer(r"OBJ_\d+", text)]
    weights = [1.0] * len(target_ids)
    for start, end in spans:
        covered = [False] * (end - start)
        for index, (begin, finish) in enumerate(intervals):
            left, right = max(start, begin), min(end, finish)
            if left < right:
                weights[index] = float(alias_weight)
                for char in range(left, right):
                    covered[char - start] = True
        if not covered or not all(covered):
            raise ValueError("alias span is not fully covered by positive-length real token offsets")
    weights[-1] = 1.0
    return weights


def alias_bonus_coefficients(tokenizer, text, target_ids, *, beta=1.0, eos_id):
    """Coefficients for baseline mean CE plus an independent alias-only bonus.

    For T complete target tokens (including EOS) and A alias-overlapping pieces,
    c_i = 1/T + beta/A for alias pieces, otherwise 1/T. When no alias exists the
    bonus is absent. Compute ``(token_losses * coefficients).sum()``; do NOT
    divide by their sum or take another mean. This preserves every non-alias
    coefficient, including the first task-decision token and EOS, at exactly
    its baseline 1/T weight. With aliases the coefficient sum is 1 + beta.

    Full-target immutable-proto offsets and actual IDs are validated even when
    beta=0. No standalone alias tokenization or guessed token boundary is used.
    """
    if (type(beta) not in (int, float) or not math.isfinite(beta) or beta < 0):
        raise ValueError('alias bonus beta must be finite and nonnegative')
    markers = alias_span_weights(tokenizer, text, target_ids, alias_weight=2.0, eos_id=eos_id)
    count = sum(weight == 2.0 for weight in markers)
    baseline = 1.0 / len(markers)
    bonus = float(beta) / count if count else 0.0
    return [baseline + bonus if marker == 2.0 else baseline for marker in markers]
