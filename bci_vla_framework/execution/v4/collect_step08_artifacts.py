"""Download only report-selected Step08 checkpoints, with SHA256 verification.

Run explicitly after the local training report exists. The only remote API used
is SFTP file reading; there are no uploads, commands, deletions or test reads.
Existing different files and manifests are never replaced. Connection secrets
are loaded only at runtime by the existing read-only connection helper.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re


ROOT = Path(__file__).resolve().parents[2]
EXECUTION = ROOT / "execution/v4"
METHODS = {"Full", "B3_aux", "B4_aux", "B2"}
WEIGHTS = ("connector.safetensors", "language_lora.safetensors")
COMPLETE = "DEVELOPMENT_TRAINING_COMPLETE"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("invalid report SHA256")
    return value


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path != root and root not in path.parents:
        raise ValueError("local artifact path leaves its intended directory")
    return path


def selected_checkpoint(run, report):
    """Use the local report's selected path and its matching file hash record."""
    method, seed = run.get("method"), run.get("seed")
    step = run.get("selected_step")
    if method not in METHODS or type(seed) is not int or seed < 0 or type(step) is not int or step < 1:
        raise ValueError("invalid completed run method/seed/step")
    job_text, selected_text = report.get("remote_job"), run.get("selected_checkpoint")
    if not isinstance(job_text, str) or not isinstance(selected_text, str):
        raise ValueError("remote job/selected checkpoint path missing")
    job, selected = PurePosixPath(job_text), PurePosixPath(selected_text)
    if (not job.is_absolute() or ".." in job.parts or ".." in selected.parts
            or job.parent != PurePosixPath("/home/oyjt/Scene-aware-BCI-VLA-runs")
            or not job.name.startswith("v4_step08_training_")):
        raise ValueError("remote checkpoint outside the Step08 training job allowlist")
    expected = job / "results" / f"{method}_seed{seed}" / f"step{step}"
    if selected != expected:
        raise ValueError("selected checkpoint does not match method/seed/step inside this job")
    matches = [c for c in run.get("checkpoints", [])
               if c.get("path") == selected_text and c.get("step") == step]
    if len(matches) != 1:
        raise ValueError("selected checkpoint must have one matching hash record")
    hashes = matches[0].get("files", {})
    if "language_lora.safetensors" not in hashes:
        raise ValueError("selected checkpoint has no language LoRA SHA256")
    if method != "B2" and "connector.safetensors" not in hashes:
        raise ValueError("connector method has no selected connector SHA256")
    # Other files (optimizer.pt, validation.json, logs) are intentionally ignored.
    files = {name: checked_hash(hashes[name]) for name in WEIGHTS if name in hashes}
    return selected, files


def connect_readonly():
    helper = ROOT.parent / ".server-inspect/remote_readonly.py"
    spec = importlib.util.spec_from_file_location("step08_remote_readonly", helper)
    if spec is None or spec.loader is None:
        raise RuntimeError("read-only connection helper unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.connect()


def download_verified(sftp, remote_path, destination, expected_hash, artifact_root):
    destination = inside(destination, artifact_root)
    if destination.exists():
        actual = sha256(destination)
        if actual != expected_hash:
            raise FileExistsError("existing local checkpoint differs; it was not overwritten")
        return {"status": "existing_verified", "sha256_actual": actual,
                "bytes": destination.stat().st_size}
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".download-partial")
    suffix = 0
    while partial.exists():
        suffix += 1
        partial = destination.with_name(destination.name + f".download-partial{suffix}")
    partial = inside(partial, artifact_root)
    digest = hashlib.sha256()
    # Exclusive local creation and remote rb mode: never an upload or overwrite.
    with partial.open("xb") as local:
        with sftp.open(str(remote_path), "rb") as remote:
            for chunk in iter(lambda: remote.read(1024 * 1024), b""):
                local.write(chunk)
                digest.update(chunk)
        local.flush()
        os.fsync(local.fileno())
    actual = digest.hexdigest()
    if actual != expected_hash:
        raise ValueError("download SHA256 mismatch; partial retained, destination not published")
    if os.name == "nt":
        # Windows rename fails if a destination appeared concurrently.
        partial.rename(destination)
        retained = None
    else:
        # Atomic no-replace publication on other platforms. Keep the hardlink
        # source because this collector deliberately never deletes any file.
        os.link(partial, destination)
        retained = str(partial)
    result = {"status": "downloaded_verified", "sha256_actual": sha256(destination),
              "bytes": destination.stat().st_size}
    if result["sha256_actual"] != expected_hash:
        raise ValueError("published checkpoint SHA256 mismatch")
    if retained is not None:
        result["retained_download_hardlink"] = retained
    return result


def collect(attempt):
    if type(attempt) is not int or attempt < 1:
        raise ValueError("attempt must be a positive integer")
    report_path = EXECUTION / f"08_real_attempt{attempt}.json"
    manifest_path = EXECUTION / f"08_checkpoint_manifest{attempt}.json"
    if not report_path.is_file():
        raise FileNotFoundError("local training report is not available yet")
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    if any(report.get(flag) is not False for flag in ("formal_test_accessed", "calibration_accessed")):
        raise ValueError("report does not confirm calibration/test remained sealed")
    config = report["config"]
    base_sha = checked_hash(config["base_sha256"])
    initial_lora_sha = checked_hash(config["initial_language_adapter_sha256"])
    expected_runs = {(method, seed) for method in config["methods"] for seed in config["seeds"]}
    if not expected_runs or any(method not in METHODS or type(seed) is not int or seed < 0 for method, seed in expected_runs):
        raise ValueError("invalid configured methods/seeds")
    artifact_root = inside(ROOT / "runs/neural_interaction_v4/dev" / f"step08_attempt{attempt}", ROOT)
    manifest = {"version": "V4-08-selected-checkpoint-manifest/1", "attempt": attempt,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_report": str(report_path), "source_report_sha256": sha256(report_path),
                "training_report_status": report.get("status"), "files": [], "skipped_runs": [], "errors": [],
                "base_initialization_sha256": base_sha, "language_lora_initialization_sha256": initial_lora_sha,
                "formal_test_accessed": False, "calibration_accessed": False,
                "remote_operations": "SFTP read selected safetensors only; no commands/uploads/deletions"}
    completed = set()
    planned = []
    for run in report.get("runs", []):
        summary = {key: run.get(key) for key in ("method", "seed", "status", "optimizer_steps", "selected_step")}
        if run.get("status") != COMPLETE:
            manifest["skipped_runs"].append(summary)
            continue
        try:
            key = (run.get("method"), run.get("seed"))
            if key not in expected_runs or key in completed:
                raise ValueError("completed run is unexpected or duplicated")
            remote_dir, hashes = selected_checkpoint(run, report)
            completed.add(key)
            for filename, expected_hash in hashes.items():
                destination = inside(artifact_root / f"{key[0]}_seed{key[1]}" / filename, artifact_root)
                planned.append({"method": key[0], "seed": key[1], "step": run["selected_step"],
                                "remote_path": str(remote_dir / filename), "local_path": str(destination),
                                "sha256_expected": expected_hash, "base_initialization_sha256": base_sha,
                                "language_lora_initialization_sha256": initial_lora_sha})
        except Exception as exc:
            manifest["errors"].append({**summary, "error_type": type(exc).__name__, "error": str(exc)})
    client = sftp = None
    connection_failed = False
    try:
        for item in planned:
            entry = dict(item)
            try:
                destination = Path(item["local_path"])
                if destination.exists():
                    # Local verified reuse requires no credential/SSH access.
                    actual = sha256(destination)
                    if actual != item["sha256_expected"]:
                        raise FileExistsError("existing local checkpoint differs; it was not overwritten")
                    entry.update(status="existing_verified", sha256_actual=actual, bytes=destination.stat().st_size)
                else:
                    if connection_failed:
                        raise ConnectionError("SSH/SFTP unavailable; credentials are not logged")
                    if sftp is None:
                        try:
                            client = connect_readonly()
                            sftp = client.open_sftp()
                        except Exception:
                            connection_failed = True
                            raise ConnectionError("SSH/SFTP connection failed; credentials are not logged") from None
                    entry.update(download_verified(sftp, item["remote_path"], destination,
                                                   item["sha256_expected"], artifact_root))
            except Exception as exc:
                entry.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
            manifest["files"].append(entry)
    finally:
        if sftp is not None:
            try:
                sftp.close()
            except Exception:
                manifest["errors"].append({"error_type": "ConnectionCloseError", "error": "SFTP close failed; downloaded file evidence retained"})
        if client is not None:
            try:
                client.close()
            except Exception:
                manifest["errors"].append({"error_type": "ConnectionCloseError", "error": "SSH close failed; credentials are not logged"})
    verified = {"downloaded_verified", "existing_verified"}
    manifest["missing_or_incomplete_runs"] = [{"method": m, "seed": s} for m, s in sorted(expected_runs - completed)]
    manifest["all_expected_runs_training_complete"] = completed == expected_runs and not manifest["skipped_runs"] and not manifest["errors"]
    manifest["all_planned_files_verified"] = bool(planned) and all(f["status"] in verified for f in manifest["files"])
    manifest["status"] = ("ALL_EXPECTED_CHECKPOINTS_COLLECTED" if manifest["all_expected_runs_training_complete"] and manifest["all_planned_files_verified"]
                          else "PARTIAL_CHECKPOINT_COLLECTION")
    manifest["full_step08_complete"] = False  # Collection cannot upgrade training/evaluation scope.
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if manifest_path.exists():
        # Preserve every previous ledger; a rerun gets a numbered new ledger.
        counter = 1
        candidate = manifest_path.with_name(manifest_path.stem + f"_recheck{counter}.json")
        while candidate.exists():
            counter += 1
            candidate = manifest_path.with_name(manifest_path.stem + f"_recheck{counter}.json")
        manifest_path = candidate
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
    print(json.dumps({"manifest": str(manifest_path), "status": manifest["status"],
                      "verified_files": sum(f["status"] in verified for f in manifest["files"]),
                      "planned_files": len(planned), "all_expected_runs_training_complete": manifest["all_expected_runs_training_complete"]}, ensure_ascii=False))
    return 0 if manifest["status"] == "ALL_EXPECTED_CHECKPOINTS_COLLECTED" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=int, default=3)
    args = parser.parse_args()
    try:
        return collect(args.attempt)
    except Exception as exc:
        # Do not print connection tracebacks or credential-profile contents.
        print(json.dumps({"status": "COLLECTION_ABORTED", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
