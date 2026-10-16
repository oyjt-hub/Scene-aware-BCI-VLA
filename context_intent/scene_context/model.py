"""A small, finite-grammar context intent network. This module never runs a robot.

Labels use deterministic hash embeddings, so observations do not need a fitted
vocabulary. All supervision belongs in training.py; forward accepts observations,
never case annotations. The model can propose only enumerated grounded intents.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .schema import GOALS as GOAL_NAMES


STATUS_NAMES = ("ready", "clarify", "reject")
SLOT_NAMES = ("source", "action", "destination")
ACTION_NAMES = ("pick", "place", "fold")
HINT_NAMES = ("unspecified", "leftmost", "rightmost", "nearest_source")
AFFORDANCES = ("pickable", "container", "foldable")
RELATION_NAMES = (
    "none", "left_of", "right_of", "above", "below", "near", "inside",
    "contains", "on", "overlapping", "supports", "in_front_of", "behind", "other",
)
VARIANTS = ("full", "no_relations", "no_bci", "no_history", "mlp")


def _bucket(text: Any, buckets: int) -> int:
    """Stable across processes and machines (Python hash() is not)."""
    raw = str(text).strip().casefold().encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little") % buckets


def _one_hot(value: Any, names: tuple[str, ...]) -> list[float]:
    return [float(value == name) for name in names]


@dataclass
class ModelConfig:
    variant: str = "full"
    hidden_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    label_buckets: int = 1024
    attribute_buckets: int = 256
    max_history: int = 8

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown variant {self.variant!r}; choose from {VARIANTS}")
        if self.num_layers < 1 or self.num_heads < 1:
            raise ValueError("num_layers and num_heads must be positive")
        if self.hidden_dim < 8 or self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be >= 8 and divisible by num_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if min(self.label_buckets, self.attribute_buckets) < 2:
            raise ValueError("Embedding bucket counts must be at least 2")
        if self.max_history < 0:
            raise ValueError("max_history must be nonnegative")


class RelationAttention(nn.Module):
    """Directed edge predicates affect both attention logits and message values."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.width = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.relation_bias = nn.Embedding(len(RELATION_NAMES), heads)
        self.relation_value = nn.Embedding(len(RELATION_NAMES), dim)
        self.geometry_bias = nn.Linear(3, heads, bias=False)
        self.geometry_value = nn.Linear(3, dim, bias=False)
        self.output = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edges: torch.Tensor,
                geometry: torch.Tensor, use_relations: bool) -> torch.Tensor:
        n, dim = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q = q.view(n, self.heads, self.width).permute(1, 0, 2)
        k = k.view(n, self.heads, self.width).permute(1, 0, 2)
        v = v.view(n, self.heads, self.width).permute(1, 0, 2)
        logits = torch.einsum("hid,hjd->hij", q, k) / math.sqrt(self.width)
        if use_relations:
            bias = torch.einsum("ijr,rh->ijh", edges, self.relation_bias.weight)
            bias = bias + self.geometry_bias(geometry)
            logits = logits + bias.permute(2, 0, 1)
        attention = self.dropout(torch.softmax(logits, dim=-1))
        values = torch.einsum("hij,hjd->hid", attention, v)
        if use_relations:
            rel_values = torch.einsum("ijr,rd->ijd", edges, self.relation_value.weight)
            rel_values = rel_values + self.geometry_value(geometry)
            rel_values = rel_values.view(n, n, self.heads, self.width).permute(2, 0, 1, 3)
            values = values + torch.einsum("hij,hijd->hid", attention, rel_values)
        values = values.permute(1, 0, 2).reshape(n, dim)
        x = x + self.dropout(self.output(values))
        return x + self.dropout(self.ffn(self.norm2(x)))


class IntentModel(nn.Module):
    """Permutation-equivariant object encoder and BCI-conditioned intent scorer.

    ``forward(obs_or_list)`` returns status/slot logits, per-observation candidate
    logits, and candidate dictionaries. No padding or fixed object count is used.
    ``mlp`` keeps the same input features but replaces relation attention with
    independent per-object MLP layers and permutation-invariant mean pooling.
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        dim = self.config.hidden_dim
        self.label_embedding = nn.Embedding(self.config.label_buckets, dim)
        self.attribute_embedding = nn.Embedding(self.config.attribute_buckets, dim)
        # xy, detector confidence, three affordances, selected indicator, BCI p.
        self.numeric_encoder = nn.Linear(8, dim)
        self.node_norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            RelationAttention(dim, self.config.num_heads, self.config.dropout)
            for _ in range(self.config.num_layers)
        ] if self.config.variant != "mlp" else [])
        self.independent_layers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(),
                          nn.Dropout(self.config.dropout), nn.Linear(dim * 2, dim))
            for _ in range(self.config.num_layers)
        ] if self.config.variant == "mlp" else [])
        self.empty_scene = nn.Parameter(torch.zeros(dim))
        self.no_destination = nn.Parameter(torch.zeros(dim))
        self.action_embedding = nn.Embedding(len(ACTION_NAMES), dim)
        self.history_encoder = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, dim))
        # goal(variable), confirmed action(3), hint(4), confirmed destination presence,
        # BCI max probability, entropy, selection presence, object count.
        self.context_encoder = nn.Linear(len(GOAL_NAMES) + 12, dim)
        self.query_encoder = nn.Sequential(nn.Linear(dim * 4, dim * 2), nn.GELU(),
                                           nn.Dropout(self.config.dropout), nn.Linear(dim * 2, dim))
        self.candidate_encoder = nn.Sequential(nn.Linear(dim * 4 + 3, dim * 2), nn.GELU(),
                                               nn.Dropout(self.config.dropout), nn.Linear(dim * 2, dim))
        self.candidate_score = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 1))
        self.status_head = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 3))
        self.slot_head = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 3))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _encode(self, observation: dict) -> tuple[torch.Tensor, torch.Tensor, dict, torch.Tensor]:
        from .schema import validate_observation
        obs = validate_observation(observation)
        objects = obs["objects"]
        n = len(objects)
        index = {obj["id"]: i for i, obj in enumerate(objects)}
        context = obs.get("context", {})
        selection = obs.get("selection", {})
        selected_id = selection.get("object_id")
        probabilities = selection.get("probabilities", {}) or {}
        use_bci = self.config.variant != "no_bci"
        use_relations = self.config.variant not in ("no_relations", "mlp")
        numeric, label_ids, attribute_vectors = [], [], []
        positions = []
        for obj in objects:
            xy = obj.get("position") or [0.5, 0.5]
            positions.append(xy)
            aff = obj.get("affordances", [])
            p = (float(probabilities.get(obj["id"], 0.0)) if probabilities else float(obj["id"] == selected_id)) if use_bci else 0.0
            numeric.append([float(xy[0]), float(xy[1]), float(obj.get("detection_confidence", 1.0)),
                            *[float(a in aff) for a in AFFORDANCES],
                            float(use_bci and obj["id"] == selected_id), p])
            label_ids.append(_bucket(obj.get("label", "unknown"), self.config.label_buckets))
            attrs = obj.get("attributes", []) or []
            if attrs:
                ids = torch.tensor([_bucket(a, self.config.attribute_buckets) for a in attrs],
                                   device=self.device, dtype=torch.long)
                attribute_vectors.append(self.attribute_embedding(ids).mean(0))
            else:
                attribute_vectors.append(torch.zeros(self.config.hidden_dim, device=self.device))
        if n:
            x = self.label_embedding(torch.tensor(label_ids, device=self.device))
            x = self.node_norm(x + self.numeric_encoder(torch.tensor(numeric, device=self.device))
                               + torch.stack(attribute_vectors))
            xy_tensor = torch.tensor(positions, dtype=x.dtype, device=self.device)
            delta = xy_tensor[None, :, :] - xy_tensor[:, None, :]
            geometry = torch.cat([delta, torch.linalg.vector_norm(delta, dim=-1, keepdim=True)], -1)
            edges = torch.zeros(n, n, len(RELATION_NAMES), device=self.device)
            for edge in obs.get("relations", []):
                i, j = index.get(edge.get("subject")), index.get(edge.get("object"))
                if i is None or j is None:
                    continue
                name = edge.get("predicate", "other")
                relation_id = RELATION_NAMES.index(name) if name in RELATION_NAMES else len(RELATION_NAMES) - 1
                edges[i, j, relation_id] += float(edge.get("confidence", 1.0))
            if self.config.variant == "mlp":
                for layer in self.independent_layers:
                    x = x + layer(x)
            else:
                for layer in self.layers:
                    x = layer(x, edges, geometry, use_relations)
            pooled = x.mean(0)
            weights = torch.tensor([(float(probabilities.get(o["id"], 0.0)) if probabilities else float(o["id"] == selected_id))
                                    for o in objects], dtype=x.dtype, device=self.device)
            selected = (x * weights[:, None]).sum(0) / weights.sum().clamp_min(1e-8) if use_bci else torch.zeros_like(pooled)
            probs = weights / weights.sum().clamp_min(1e-8)
            max_p = float(probs.max().item()) if use_bci else 0.0
            entropy = float((-(probs * probs.clamp_min(1e-8).log()).sum()).item()) if use_bci else 0.0
        else:
            x = torch.empty(0, self.config.hidden_dim, device=self.device)
            pooled = self.empty_scene
            selected = torch.zeros_like(pooled)
            xy_tensor = torch.empty(0, 2, device=self.device)
            max_p, entropy = 0.0, 0.0
        history = context.get("history", []) or []
        history = history[-self.config.max_history:] if self.config.max_history else []
        if history and self.config.variant != "no_history":
            # Encode referenced nodes, not arbitrary instance-ID spellings. Past
            # objects can have disappeared; absent references have zero features.
            encoded = []
            for event in history:
                action = self.action_embedding.weight[ACTION_NAMES.index(event["action"])]
                source_i = index.get(event.get("source_id"))
                destination_i = index.get(event.get("destination_id"))
                source = x[source_i] if source_i is not None else torch.zeros_like(pooled)
                destination = x[destination_i] if destination_i is not None else torch.zeros_like(pooled)
                encoded.append(self.history_encoder(torch.cat([action, source, destination])))
            weights = torch.arange(1, len(history) + 1, device=self.device, dtype=pooled.dtype)
            historical = (torch.stack(encoded) * weights[:, None]).sum(0) / weights.sum()
        else:
            historical = torch.zeros_like(pooled)
        confirmed_destination = context.get("confirmed_destination_id")
        destination_vec = x[index[confirmed_destination]] if confirmed_destination in index else torch.zeros_like(pooled)
        context_features = [*_one_hot(context.get("goal", "unspecified"), GOAL_NAMES),
                            *_one_hot(context.get("confirmed_action"), ACTION_NAMES),
                            *_one_hot(context.get("destination_hint", "unspecified"), HINT_NAMES),
                            float(confirmed_destination in index), max_p, entropy,
                            float(use_bci and selected_id in index), min(n, 100) / 10.0]
        encoded_context = self.context_encoder(torch.tensor(context_features, device=self.device)) + destination_vec
        query = self.query_encoder(torch.cat([pooled, selected, historical, encoded_context]))
        return x, query, obs, xy_tensor

    def forward(self, observations: dict | list[dict]) -> dict[str, Any]:
        from .planning import generate_candidates
        if isinstance(observations, dict):
            observations = [observations]
        if not observations:
            raise ValueError("forward requires at least one observation")
        status_logits, slot_logits, scores, all_candidates = [], [], [], []
        for observation in observations:
            nodes, query, obs, positions = self._encode(observation)
            index = {obj["id"]: i for i, obj in enumerate(obs["objects"])}
            pooled = nodes.mean(0) if len(nodes) else self.empty_scene
            joint = torch.cat([query, pooled])
            status_logits.append(self.status_head(joint))
            slot_logits.append(self.slot_head(joint))
            candidates = generate_candidates(obs)
            features = []
            for candidate in candidates:
                src_i = index[candidate["source_id"]]
                src = nodes[src_i]
                dst_i = index.get(candidate.get("destination_id"))
                dst = nodes[dst_i] if dst_i is not None else self.no_destination
                geometry = torch.zeros(3, device=self.device)
                if dst_i is not None and self.config.variant not in ("no_relations", "mlp"):
                    delta = positions[dst_i] - positions[src_i]
                    geometry = torch.cat([delta, torch.linalg.vector_norm(delta).view(1)])
                action = self.action_embedding.weight[ACTION_NAMES.index(candidate["action"])]
                features.append(torch.cat([action, src, dst, src * dst, geometry]))
            if features:
                encoded = self.candidate_encoder(torch.stack(features))
                q = query[None, :].expand_as(encoded)
                logits = self.candidate_score(torch.cat([encoded, q, encoded * q], -1)).squeeze(-1)
            else:
                logits = torch.empty(0, device=self.device)
            scores.append(logits)
            all_candidates.append(candidates)
        return {"status_logits": torch.stack(status_logits), "slot_logits": torch.stack(slot_logits),
                "candidate_logits": scores, "candidates": all_candidates}


def proposal_from_output(output: dict, index: int = 0,
                         status_temperature: float = 1.0,
                         candidate_temperature: float = 1.0) -> tuple[dict, float]:
    """Convert network logits into an unguarded prediction and ready confidence."""
    status_p = torch.softmax(output["status_logits"][index] / status_temperature, dim=-1)
    status = STATUS_NAMES[int(status_p.argmax().item())]
    candidates = output["candidates"][index]
    scores = output["candidate_logits"][index]
    intent, candidate_p = None, 0.0
    if len(candidates):
        probabilities = torch.softmax(scores / candidate_temperature, dim=-1)
        candidate_idx = int(probabilities.argmax().item())
        intent, candidate_p = candidates[candidate_idx], float(probabilities[candidate_idx].item())
    if status == "ready" and intent is None:
        status = "reject"
    slot = SLOT_NAMES[int(output["slot_logits"][index].argmax().item())] if status == "clarify" else None
    # Keep threshold=1 as a well-defined abstain-all setting even if float32
    # softmax saturates to exactly one on a small validation split.
    confidence = min(float(status_p[0].item()) * candidate_p, 1.0 - 1e-7)
    return {"status": status, "intent": intent if status == "ready" else None,
            "clarification_slot": slot}, confidence


class NeuralPredictor:
    """Load a local checkpoint and apply the shared, confirmation-only guard."""

    def __init__(self, checkpoint_path: str | Path, threshold: float | None = None,
                 device: str = "cpu"):
        path = Path(checkpoint_path)
        if path.is_dir():
            path = path / "model.pt"
        # weights_only prevents arbitrary checkpoint unpickling on modern torch.
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=True)
        except TypeError:  # torch before the weights_only keyword
            checkpoint = torch.load(path, map_location=device)
        self.model = IntentModel(ModelConfig(**checkpoint["config"]))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(device).eval()
        self.threshold = float(checkpoint.get("threshold", 0.0) if threshold is None else threshold)
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        self.status_temperature = float(checkpoint.get("status_temperature", 1.0))
        self.candidate_temperature = float(checkpoint.get("candidate_temperature", 1.0))
        self.checkpoint_path = str(path.resolve())
        self.checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.source = "neural:" + self.model.config.variant

    def predict(self, observation: dict) -> dict:
        from .planning import finalize_prediction
        with torch.inference_mode():
            output = self.model(observation)
            proposal, confidence = proposal_from_output(
                output, status_temperature=self.status_temperature,
                candidate_temperature=self.candidate_temperature)
        return finalize_prediction(observation, proposal, confidence=confidence,
                                   threshold=self.threshold, source=self.source)

    def describe(self) -> dict:
        return {"source": self.source, "config": asdict(self.model.config),
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "threshold": self.threshold, "status_temperature": self.status_temperature,
                "candidate_temperature": self.candidate_temperature,
                "checkpoint_path": self.checkpoint_path, "checkpoint_sha256": self.checkpoint_sha256,
                "torch_version": str(torch.__version__)}

    def reproducibility_settings(self) -> dict:
        return self.describe()
