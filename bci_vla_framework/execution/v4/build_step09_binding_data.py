"""Build append-only, complete target/alias groups on the original LIBERO host."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bci_vla.training_v4.balanced_binding_data import build_balanced_binding_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "data/v4")
    parser.add_argument("--capture-root", type=Path, help="Existing step04 capture job root; reconstruct allowed partitions in place")
    parser.add_argument("--output-root", type=Path, required=True, help="New directory outside the frozen source tree")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v4/train_binding_data_step09.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    result = build_balanced_binding_data(args.source_root, args.output_root, config, capture_root=args.capture_root)
    summary_keys = ("public_rows", "base_episodes", "scene_groups", "binding_rows", "choice_groups",
                    "complete_choice_pairs", "alias_groups", "joint_alias_pairs", "channel_targets")
    print(json.dumps({"output_root": str(args.output_root.resolve()), "version": result["version"],
                      "counts": {split: {key: value[key] for key in summary_keys} for split, value in result["counts"].items()},
                      "full_distribution_manifest": str(args.output_root.resolve() / "manifest.json"),
                      "general_hidden_intent_supervision_ready": False,
                      "place_phase_supervision_ready": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
