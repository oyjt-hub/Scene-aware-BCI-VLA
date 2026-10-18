"""B2 text evidence from exactly the connector's cleaned public tensor inputs.

The shared language prompt is supplied by the data pipeline, unchanged. This
module never opens private files, serializes provenance IDs, or adds diagnostic
receipts that the connector did not observe. Sparse zeros are implicit; every
nonzero float32 value is preserved, with no rounding or length-driven dropping.
"""
from __future__ import annotations

import json
import math
import re

import torch

from bci_vla.interaction_contract_v4.validation import canonical_digest
from bci_vla.neural_interaction_v4.config import ModuleConfig
from bci_vla.neural_interaction_v4.model import clean_batch, FLOAT_KEYS
from bci_vla.neural_interaction_v4.tensorizer import collate_public


FORMAT = "connector-public-tensors-text/0.1"
# These are static feature definitions, not extra observations. Boolean tensors
# are true flat indices; numeric tensors are [flat-index,value] pairs, row-major.
LAYOUT = {
    "objects": "ROI RGB mean/std; semantic; bbox_xyxy; roi_valid; semantic_valid",
    "selection": "score,score_valid,selected,RAW_SCORE,CALIBRATED_PROBABILITY,HARD_LABEL,NONE",
    "option_numeric": "score,score_valid,selected,OBJECT,TASK,DIAGNOSTIC,DECISION,pick,place,accept,reject",
    "event_numeric": "TASK,INTENT_QUERY,CONFIRM,REJECT,NO_RESPONSE,original_TASK,original_INTENT_QUERY,original_PROBE,original_CONFIRM,RAW_SCORE,CALIBRATED_PROBABILITY,HARD_LABEL,NONE,age/30,duration/5,pick,place",
    "state_numeric": "present,pick,place,pending_TASK,pending_INTENT_QUERY,pending_PROBE,pending_CONFIRM,unused,time_to_expiry/30,stop,confirmed,suggestion,pending",
    "diag_history": "score_kind[4],known_target_match,no_response,known_score,known_score_valid,sorted_other_scores[15],other_score_valid[15],duration/5,age/30",
    "diag_summary": "attempts/64,correct/64,no_response/64,covered/16,age/30,truncated,has_latest,sufficient,expired,insufficient",
    "links": "*_src and *_dst last axis is object row; options/tasks/pending use option_numeric layout; *_mask marks valid rows",
}


def compact_model_view(public, *, config=None):
    """Return a lossless sparse view of the model's single-example tensors.

    Object rows bind only to public aliases. Instance/session/event IDs, hashes,
    filenames and original out-of-window PROBE receipts are never text features.
    The public schema/tensorizer still validates all inputs before use.
    """
    config = config or ModuleConfig()
    tensors, _ = collate_public([public], config)
    tensors = clean_batch(tensors, config)
    aliases = [[i, obj["alias"]] for i, obj in enumerate(public["scene"]["objects"])
               if bool(tensors["candidate_mask"][0, i])]
    packed = {}
    for key in sorted(tensors):
        tensor = tensors[key][0].detach().cpu().contiguous()
        flat = tensor.reshape(-1)
        present = flat != 0
        if key in FLOAT_KEYS:
            present = present | torch.signbit(flat)  # retain signed zero too
        indices = present.nonzero().reshape(-1).tolist()
        values = ([[i, float(flat[i])] for i in indices] if key in FLOAT_KEYS else indices)
        packed[key] = {"shape": list(tensor.shape), "values": values}
    return {"format": FORMAT, "encoding": "row-major sparse; omitted entries=0; booleans=true indices; floats=float32",
            "object_rows": aliases, "layout": LAYOUT.copy(), "tensors": packed}


def _compact_json(value):
    # Nine significant decimal digits exactly round-trip all finite float32
    # values. This removes float64 repr noise without losing a model input bit.
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite model text value")
        if value == 0 and math.copysign(1, value) < 0:
            return "-0.0"
        return format(value, ".9g")
    if isinstance(value, dict):
        return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + _compact_json(v) for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_compact_json(v) for v in value) + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def baseline_evidence_text(public, *, config=None):
    """Text appendage only; use build_b2_prompt for shared-prompt/cap checks."""
    return "Public interaction evidence (decoded choices can be noisy): " + _compact_json(compact_model_view(public, config=config))


def build_b2_prompt(public, language_prompt, *, config=None, tokenizer=None, max_tokens=None):
    """Append same-information evidence to the exact shared language prompt.

    ``tokenizer`` is a callable returning the real runtime token IDs, including
    its actual BOS convention. With ``max_tokens`` it is mandatory. Overflow
    raises, never truncates. Returned hashes/counts are audit metadata and are
    not inserted into the prompt. B2 has no soft prefix and no auxiliary heads.
    """
    if not isinstance(language_prompt, str) or not language_prompt.strip():
        raise ValueError("nonempty shared language prompt required")
    if max_tokens is not None and (not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1):
        raise ValueError("positive token budget required")
    if max_tokens is not None and tokenizer is None:
        raise ValueError("real tokenizer required to enforce text token budget")
    view = compact_model_view(public, config=config)
    for _, alias in view["object_rows"]:
        if not re.search(r"(?<![A-Za-z0-9_])" + re.escape(alias) + r"(?![A-Za-z0-9_])", language_prompt):
            raise ValueError("shared prompt lacks public object alias binding: " + alias)
    text = language_prompt + "\nPublic interaction evidence (decoded choices can be noisy): " + _compact_json(view)
    count = None
    if tokenizer is not None:
        if not callable(tokenizer):
            raise TypeError("tokenizer must be the runtime text-to-token-IDs callable")
        ids = tokenizer(text)
        if isinstance(ids, torch.Tensor):
            if ids.ndim != 1:
                raise ValueError("tokenizer must return one unbatched ID sequence")
            ids = ids.tolist()
        if not isinstance(ids, (list, tuple)) or any(not isinstance(i, int) or isinstance(i, bool) for i in ids):
            raise ValueError("tokenizer must return one unbatched integer ID sequence")
        count = len(ids)
        if max_tokens is not None and count > max_tokens:
            raise ValueError(f"complete B2 prompt needs {count} text tokens, budget {max_tokens}; no truncation")
    return {"prompt": text, "text_tokens": count, "truncated": False, "prefix_tokens": 0,
            "view_sha256": canonical_digest(view), "public_input_sha256": canonical_digest(public),
            "token_count_status": "runtime_tokenizer" if count is not None else "not_counted_requires_runtime_tokenizer"}
