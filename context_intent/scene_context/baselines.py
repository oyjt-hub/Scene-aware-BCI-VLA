"""Conservative rule baseline using the same observation and output contract."""

from __future__ import annotations

from typing import Any

from .planning import compatible_candidates, finalize_prediction, structural_clarification
from .schema import validate_observation


class RulePredictor:
    """Choose only a unique supported intent; otherwise ask for missing information.

    This is a full-input rule baseline, not the original object-name fallback. It
    has no learned weights, annotation access, or robot connection.
    """

    def reproducibility_settings(self) -> dict[str, Any]:
        return {"predictor": "rule", "version": 1, "selection": "unique_candidate"}

    def predict(self, observation: dict[str, Any]) -> dict[str, Any]:
        obs = validate_observation(observation)
        missing = structural_clarification(obs)
        if missing:
            if "status" in missing:
                proposal = dict(missing)
            else:
                proposal = {
                    "status": "clarify", "intent": None,
                    "clarification_slot": missing.get("clarification_slot", missing.get("slot")),
                }
        else:
            candidates = compatible_candidates(obs)
            if len(candidates) == 1:
                proposal = {"status": "ready", "intent": candidates[0], "clarification_slot": None}
            elif not candidates:
                proposal = {"status": "reject", "intent": None, "clarification_slot": None}
            else:
                # The first unresolved factor is asked explicitly. Ties are never
                # resolved by object order or arbitrary alphabetic instance IDs.
                slot = "source"
                for key, name in (("source_id", "source"), ("action", "action"),
                                  ("destination_id", "destination")):
                    if len({item.get(key) for item in candidates}) > 1:
                        slot = name
                        break
                proposal = {"status": "clarify", "intent": None, "clarification_slot": slot}
        return finalize_prediction(obs, proposal, source="rule")
