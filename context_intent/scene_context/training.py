"""CPU training and strictly validation-only calibration for the intent model."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch.nn import functional as F

from .model import IntentModel, ModelConfig, STATUS_NAMES, SLOT_NAMES, proposal_from_output


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _key(intent: dict | None) -> tuple | None:
    if intent is None:
        return None
    return intent.get("action"), intent.get("source_id"), intent.get("destination_id")


def _loss(output: dict, cases: list[dict], *, candidate_weight: float = 1.0,
          slot_weight: float = 0.25) -> torch.Tensor:
    device = output["status_logits"].device
    target = torch.tensor([STATUS_NAMES.index(c["annotation"]["status"]) for c in cases], device=device)
    total = F.cross_entropy(output["status_logits"], target)
    candidate_losses, slot_losses = [], []
    for i, case in enumerate(cases):
        annotation = case["annotation"]
        if annotation["status"] == "ready":
            keys = [_key(c) for c in output["candidates"][i]]
            correct = _key(annotation.get("intent"))
            if correct not in keys:
                raise ValueError(f"Ready annotation is outside candidate grammar: {case['case_id']}: {correct}")
            label = torch.tensor([keys.index(correct)], device=device)
            candidate_losses.append(F.cross_entropy(output["candidate_logits"][i][None, :], label))
        if annotation["status"] == "clarify" and annotation.get("clarification_slot") in SLOT_NAMES:
            label = torch.tensor([SLOT_NAMES.index(annotation["clarification_slot"])], device=device)
            slot_losses.append(F.cross_entropy(output["slot_logits"][i][None, :], label))
    if candidate_losses:
        total = total + candidate_weight * torch.stack(candidate_losses).mean()
    if slot_losses:
        total = total + slot_weight * torch.stack(slot_losses).mean()
    return total


def _check_cases(cases: list[dict], split: str) -> None:
    from .planning import generate_candidates
    if not cases:
        raise ValueError(f"{split} split must contain at least one case")
    seen = set()
    for case in cases:
        if case["case_id"] in seen:
            raise ValueError(f"Duplicate case_id in {split}: {case['case_id']}")
        seen.add(case["case_id"])
        if case["annotation"]["status"] == "ready":
            if _key(case["annotation"].get("intent")) not in [_key(c) for c in generate_candidates(case["observation"])]:
                raise ValueError(f"{split} ready target outside candidate grammar: {case['case_id']}")


def _snapshot(model: IntentModel, cases: list[dict]) -> tuple[dict, float]:
    from .schema import public_observation
    model.eval()
    with torch.inference_mode():
        output = model([public_observation(c) for c in cases])
        loss = float(_loss(output, cases).item())
    # Keep a CPU snapshot detached from a later optimizer step.
    return {"status_logits": output["status_logits"].detach().cpu(),
            "slot_logits": output["slot_logits"].detach().cpu(),
            "candidate_logits": [v.detach().cpu() for v in output["candidate_logits"]],
            "candidates": output["candidates"]}, loss


def _temperature(output: dict, cases: list[dict], kind: str) -> tuple[float, float]:
    temperatures = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
    losses = []
    for temperature in temperatures:
        if kind == "status":
            target = torch.tensor([STATUS_NAMES.index(c["annotation"]["status"]) for c in cases])
            nll = float(F.cross_entropy(output["status_logits"] / temperature, target).item())
        else:
            terms = []
            for i, case in enumerate(cases):
                if case["annotation"]["status"] != "ready":
                    continue
                keys = [_key(c) for c in output["candidates"][i]]
                target_idx = keys.index(_key(case["annotation"]["intent"]))
                terms.append(F.cross_entropy(output["candidate_logits"][i][None, :] / temperature,
                                             torch.tensor([target_idx])))
            nll = float(torch.stack(terms).mean().item()) if terms else 0.0
        losses.append((nll, abs(temperature - 1.0), temperature))
    best = min(losses)
    return float(best[2]), float(best[0])


def _calibrate(output: dict, cases: list[dict], target_precision: float) -> dict:
    status_t, status_nll = _temperature(output, cases, "status")
    candidate_t, candidate_nll = _temperature(output, cases, "candidate")
    records = []
    for i, case in enumerate(cases):
        proposal, confidence = proposal_from_output(output, i, status_t, candidate_t)
        annotation = case["annotation"]
        correct = proposal["status"] == "ready" and annotation["status"] == "ready" and _key(proposal["intent"]) == _key(annotation.get("intent"))
        records.append({"confidence": confidence, "predicted_ready": proposal["status"] == "ready", "correct": bool(correct)})
    thresholds = sorted({0.0, 1.0, *[r["confidence"] for r in records if r["predicted_ready"]]})
    options = []
    for threshold in thresholds:
        accepted = [r for r in records if r["predicted_ready"] and r["confidence"] >= threshold]
        precision = sum(r["correct"] for r in accepted) / len(accepted) if accepted else None
        if accepted and precision >= target_precision:
            options.append((len(accepted), -threshold, threshold, precision))
    if options:
        _, _, threshold, precision = max(options)
        count = sum(r["predicted_ready"] and r["confidence"] >= threshold for r in records)
    else:
        threshold, precision, count = 1.0, None, 0
    return {"threshold": float(threshold), "status_temperature": status_t,
            "candidate_temperature": candidate_t, "validation_status_nll": status_nll,
            "validation_candidate_nll": candidate_nll, "target_precision": target_precision,
            "validation_selected_precision": precision, "validation_selected_count": count,
            "validation_case_count": len(cases), "validation_coverage": count / len(cases),
            "procedure": "validation-only temperature grid NLL; maximize raw ready coverage subject to empirical precision",
            "caveat": "Empirical validation precision is not a deployment guarantee; a small validation split can overfit calibration."}


def train_model(train_path: str | Path, val_path: str | Path, output_dir: str | Path,
                variant: str = "full", seed: int = 0, epochs: int = 20,
                batch_size: int = 16, learning_rate: float = 0.002,
                weight_decay: float = 0.0001, hidden_dim: int = 64,
                num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1,
                target_precision: float = 0.9, device: str = "cpu",
                num_threads: int = 1, synthetic: bool = False, **kwargs: Any) -> dict:
    """Train on train_path; select epoch and calibration only on val_path.

    There is intentionally no test_path argument. All physical scene IDs must be
    disjoint. For actual research, tune on validation once and evaluate a sealed
    test split with the evaluation API. Synthetic runs are pipeline smoke tests.
    """
    from .schema import assert_disjoint_scenes, load_cases, public_observation
    if "lr" in kwargs:
        learning_rate = kwargs.pop("lr")
    if kwargs:
        raise TypeError(f"Unknown training options: {sorted(kwargs)}")
    if epochs < 1 or batch_size < 1 or num_threads < 1:
        raise ValueError("epochs, batch_size, and num_threads must be positive")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative")
    if not 0 < target_precision <= 1:
        raise ValueError("target_precision must be in (0, 1]")
    torch.set_num_threads(num_threads)
    set_seed(seed)
    train_cases, val_cases = load_cases(train_path), load_cases(val_path)
    _check_cases(train_cases, "train")
    _check_cases(val_cases, "validation")
    synthetic = bool(synthetic or any(
        str(c.get("provenance", "")).startswith("synthetic")
        for c in train_cases + val_cases))
    assert_disjoint_scenes(train_cases, val_cases, [])
    if {c["case_id"] for c in train_cases} & {c["case_id"] for c in val_cases}:
        raise ValueError("Case IDs overlap train and validation")
    config = ModelConfig(variant=variant, hidden_dim=hidden_dim, num_heads=num_heads,
                         num_layers=num_layers, dropout=dropout)
    model = IntentModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rng = random.Random(seed)
    best_loss, best_state, best_epoch = float("inf"), None, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        indices = list(range(len(train_cases)))
        rng.shuffle(indices)
        train_loss, train_count = 0.0, 0
        for start in range(0, len(indices), batch_size):
            cases = [train_cases[i] for i in indices[start:start + batch_size]]
            output = model([public_observation(c) for c in cases])
            loss = _loss(output, cases)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item()) * len(cases)
            train_count += len(cases)
        _, val_loss = _snapshot(model, val_cases)
        history.append({"epoch": epoch, "train_loss": train_loss / train_count, "validation_loss": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    validation_output, _ = _snapshot(model, val_cases)
    calibration = _calibrate(validation_output, val_cases, target_precision)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "model.pt"
    checkpoint = {"format_version": 1, "config": asdict(config), "state_dict": best_state,
                  "threshold": calibration["threshold"],
                  "status_temperature": calibration["status_temperature"],
                  "candidate_temperature": calibration["candidate_temperature"]}
    torch.save(checkpoint, checkpoint_path)
    # Record grouped provenance and hashes without storing source data in weights.
    manifest = {"format_version": 1, "variant": variant, "seed": seed,
                "synthetic_smoke_test": bool(synthetic),
                "result_scope": "Synthetic pipeline smoke validation, not paper evidence" if synthetic else "Train/validation development only; independent held-out test evaluation required",
                "config": asdict(config), "parameter_count": sum(p.numel() for p in model.parameters()),
                "training": {"epochs": epochs, "best_epoch": best_epoch, "batch_size": batch_size,
                             "learning_rate": learning_rate, "weight_decay": weight_decay,
                             "best_validation_loss": best_loss, "device": device,
                             "seconds": time.perf_counter() - started},
                "splits": {"train": {"path": str(Path(train_path).resolve()), "sha256": _digest(train_path),
                                      "cases": len(train_cases), "scene_ids": sorted({c["scene_id"] for c in train_cases})},
                           "validation": {"path": str(Path(val_path).resolve()), "sha256": _digest(val_path),
                                           "cases": len(val_cases), "scene_ids": sorted({c["scene_id"] for c in val_cases})}},
                "calibration": calibration, "history": history,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "safety": "Only proposes text/clarification; common guard and explicit human confirmation required; no robot calls"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
