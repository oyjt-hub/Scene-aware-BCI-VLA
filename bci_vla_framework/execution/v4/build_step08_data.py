"""Build an append-only step08 public-protocol pilot from existing host-local data."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bci_vla.training_v4.data import build_training_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "data/v4")
    parser.add_argument("--capture-root", type=Path, help="Existing step04 capture job root; reconstruct train/validation locally without JSONL upload")
    parser.add_argument("--output-root", type=Path, required=True, help="New directory outside the frozen data/v4 tree")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v4/train_data_step08.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    manifest = build_training_data(args.source_root, args.output_root, config, capture_root=args.capture_root)
    print(json.dumps({"output_root": str(args.output_root.resolve()), "counts": manifest["counts"],
                      "restricted_public_protocol_pilot_ready": manifest["restricted_public_protocol_pilot_ready"],
                      "general_hidden_intent_supervision_ready": False, "place_phase_supervision_ready": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
