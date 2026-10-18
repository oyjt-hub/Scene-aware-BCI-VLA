"""V4-01 read-only source/resource inventory; never loads policies or models.

Uses the existing private SSH helper without copying or printing credentials.
Writes inspection evidence locally only. The remote payload reads a bounded set
of source paths and metadata, plus read-only Git/GPU status commands.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "01_remote_inventory.json"
PAYLOAD = r'''
import ast, datetime, hashlib, json, platform, subprocess
from pathlib import Path

def run(args, timeout=10):
    try:
        r = subprocess.run(args, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return {"returncode": r.returncode, "stdout": r.stdout.strip()[:18000],
                "stderr": r.stderr.strip()[:2000]}
    except Exception as exc:
        return {"error_type": type(exc).__name__, "detail": str(exc)[:500]}

def path_info(value):
    p = Path(value)
    row = {"path": str(p), "exists": p.exists()}
    if p.exists():
        stat = p.stat()
        row.update(is_directory=p.is_dir(), bytes=stat.st_size,
                   mtime_ns=stat.st_mtime_ns, resolved_path=str(p.resolve()))
        if p.is_dir():
            row["children"] = sorted(x.name for x in p.iterdir())[:65]
    return row

def source_info(value):
    row = path_info(value)
    p = Path(value)
    if not p.is_file():
        return row
    raw = p.read_bytes()
    row["sha256"] = hashlib.sha256(raw).hexdigest()
    if p.suffix == ".py":
        text = raw.decode("utf-8-sig", errors="replace")
        try:
            tree = ast.parse(text)
            row["classes"] = [{"name": n.name, "line": n.lineno}
                              for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            row["relevant_methods"] = [{"name": n.name, "line": n.lineno}
                for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and any(k in n.name.lower() for k in
                        ("embed", "sample", "generate", "infer", "token", "loss", "forward", "intent", "mask", "detect", "segment"))]
            calls = []
            for n in ast.walk(tree):
                if not isinstance(n, ast.Call):
                    continue
                fn = ast.unparse(n.func)
                if not any(k in fn.lower() for k in
                           ("from_pretrained", "build_sam", "sam_model_registry", "load_tokenizer")):
                    continue
                literal_args = [x.value for x in n.args
                                if isinstance(x, ast.Constant) and isinstance(x.value, str)]
                public_kwargs = {x.arg: x.value.value for x in n.keywords
                    if x.arg in ("checkpoint", "model_name", "model_id")
                    and isinstance(x.value, ast.Constant) and isinstance(x.value.value, str)}
                calls.append({"line": n.lineno, "function": fn,
                              "literal_args": literal_args, "public_kwargs": public_kwargs})
            row["model_reference_calls"] = calls
            constants = []
            for n in ast.walk(tree):
                if not isinstance(n, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for target in targets:
                    if not isinstance(target, ast.Name):
                        continue
                    name = target.id
                    if not any(k in name.lower() for k in ("sam", "model_path", "checkpoint", "florence")):
                        continue
                    if any(k in name.lower() for k in ("password", "secret", "api_key", "access_token")):
                        continue
                    if isinstance(n.value, ast.Constant) and isinstance(n.value.value, (str, int, bool)):
                        constants.append({"line": n.lineno, "name": name, "value": n.value.value})
            row["public_model_constants"] = constants
        except SyntaxError as exc:
            row["parse_error"] = str(exc)
    return row

repo = "/home/oyjt/Scene-aware-BCI-VLA"
openpi = "/home/oyjt/openpi"
checkpoint = "/home/oyjt/workspace/RLinf-RECAP/models/baseline_sft/RLinf-Pi05-LIBERO-SFT"
sources = [
    repo + "/scripts/inference.py", repo + "/scripts/server_api.py",
    repo + "/interface.py", repo + "/scripts/robot/RTC_inference.py",
    openpi + "/src/openpi/models/pi0.py",
    openpi + "/src/openpi/models/pi0_config.py",
    openpi + "/src/openpi/models/tokenizer.py",
    openpi + "/src/openpi/models_pytorch/pi0_pytorch.py",
    openpi + "/src/openpi/models_pytorch/gemma_pytorch.py",
    openpi + "/src/openpi/policies/policy.py",
    openpi + "/src/openpi/policies/policy_config.py",
    openpi + "/src/openpi/policies/libero_policy.py",
    openpi + "/src/openpi/training/config.py",
    checkpoint + "/config.json",
    checkpoint + "/assets/physical-intelligence/libero/norm_stats.json",
]
paths = [repo, openpi, checkpoint, checkpoint + "/model.safetensors",
    "/home/oyjt/workspace/Qwen2.5-VL-GR00T-LIBERO-4in1",
    "/home/oyjt/Scene-aware-BCI-VLA-runs/step15_libero_20261016_v4",
    "/home/oyjt/workspace/miniconda3/envs/libero",
    openpi + "/.venv",
    "/home/oyjt/.cache/huggingface/hub/models--microsoft--Florence-2-large",
    "/home/oyjt/.cache/huggingface/hub/models--microsoft--Florence-2-base",
    repo + "/checkpoints", repo + "/weights", repo + "/models"]
environments = []
for env in [Path(openpi)/".venv", Path("/home/oyjt/workspace/miniconda3/envs/libero")]:
    items = []
    for sp in sorted(env.glob("lib/python*/site-packages")):
        for pattern in ("torch-*.dist-info", "transformers-*.dist-info", "safetensors-*.dist-info",
                        "jax-*.dist-info", "mujoco-*.dist-info", "robosuite-*.dist-info"):
            items.extend(str(x) for x in sp.glob(pattern))
    environments.append({"root": str(env), "distribution_metadata": items,
                         "import_or_environment_activation_performed": False})
result = {"scope": "read_only_current_metadata_and_source_inventory_no_model_load",
    "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "host": platform.node(), "inspection_python": platform.python_version(),
    "git": {p: {"head": run(["git", "-C", p, "rev-parse", "HEAD"]),
                "branch": run(["git", "-C", p, "branch", "--show-current"]),
                "status": run(["git", "-C", p, "status", "--porcelain"])}
            for p in (repo, openpi)},
    "gpu": run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]),
    "gpu_processes": run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"]),
    "listen_ports": run(["ss", "-ltn"]),
    "paths": [path_info(p) for p in paths],
    "source_files": [source_info(p) for p in sources],
    "environments": environments,
    "not_performed": ["model_load", "model_inference", "training", "service_launch", "EEG_connection", "robot_connection", "remote_file_write", "dependency_install"]}
print(json.dumps(result, ensure_ascii=True))
'''

SUPPLEMENT = PAYLOAD.partition('repo = ')[0] + r'''
import time
repo = Path("/home/oyjt/Scene-aware-BCI-VLA")
folders = [repo/"src", repo/"Grounded-SAM-2", repo/"Scene-aware-BCI-VLA"]
files = []
for folder in folders:
    files.extend(sorted(folder.glob("*.py")))
selected = [p for p in files if any(k in p.name.lower() for k in
    ("florence", "sam", "interface", "segment", "detect"))]
sources = [source_info(str(p)) for p in selected[:35]]
imports = []
for p in [repo/"scripts/inference.py", repo/"scripts/server_api.py"]:
    tree = ast.parse(p.read_text(encoding="utf-8-sig"))
    imports.append({"path": str(p), "imports": [ast.unparse(n)
        for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]})
weight_candidates = [
    repo/"Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt",
    repo/"Grounded-SAM-2/checkpoints/sam2_hiera_large.pt",
    Path("/home/oyjt/.cache/openpi/big_vision/paligemma_tokenizer.model"),
    Path("/home/oyjt/.cache/openpi/gs/big_vision/paligemma_tokenizer.model"),
    Path("/home/oyjt/LIBERO"),
    Path("/home/oyjt/openpi/.venv/pyvenv.cfg"),
    Path("/home/oyjt/workspace/miniconda3/envs/libero/bin/python"),
]
florence = Path("/home/oyjt/.cache/huggingface/hub/models--microsoft--Florence-2-large")
cache_snapshots = []
for snap in sorted((florence/"snapshots").glob("*")):
    cache_snapshots.append({"snapshot": str(snap), "files": [path_info(str(x))
        for x in sorted(snap.iterdir()) if x.is_file()]})
tokenizer = Path("/home/oyjt/openpi/src/openpi/models/tokenizer.py")
tokenizer_refs = [{"line": i, "text": s.strip()} for i,s in
    enumerate(tokenizer.read_text(encoding="utf-8").splitlines(),1)
    if "paligemma_tokenizer" in s or "gs://" in s]
hash_path = Path("/home/oyjt/workspace/RLinf-RECAP/models/baseline_sft/RLinf-Pi05-LIBERO-SFT/model.safetensors")
start = time.monotonic(); digest = hashlib.sha256(); complete = True; consumed = 0
before = hash_path.stat()
with hash_path.open("rb") as handle:
    while True:
        chunk = handle.read(8*1024*1024)
        if not chunk:
            break
        digest.update(chunk); consumed += len(chunk)
        if time.monotonic()-start > 32:
            complete = False; break
after = hash_path.stat()
stable = (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
weight_hash = {"path": str(hash_path), "bytes": before.st_size,
    "bytes_read": consumed, "duration_s": time.monotonic()-start,
    "complete": complete, "metadata_stable": stable,
    "sha256": digest.hexdigest() if complete and stable else None}
print(json.dumps({"scope": "read_only_visual_tokenizer_and_checkpoint_integrity",
    "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "source_folders": [path_info(str(p)) for p in folders],
    "frontend_source_files": sources, "entry_imports": imports,
    "paths": [path_info(str(p)) for p in weight_candidates],
    "florence_snapshots": cache_snapshots,
    "tokenizer_source_references": tokenizer_refs,
    "pi05_weight_hash": weight_hash,
    "not_performed": ["model_load", "model_inference", "training", "service_launch", "EEG_connection", "robot_connection", "remote_file_write", "dependency_install"]}, ensure_ascii=True))
'''

INTEGRITY = PAYLOAD.partition('repo = ')[0] + r'''
import time
p = Path("/home/oyjt/workspace/RLinf-RECAP/models/baseline_sft/RLinf-Pi05-LIBERO-SFT/model.safetensors")
before = p.stat(); start = time.monotonic(); h = hashlib.sha256(); read = 0
with p.open("rb") as handle:
    while True:
        block = handle.read(16*1024*1024)
        if not block:
            break
        h.update(block); read += len(block)
after = p.stat()
stable = (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
paths = [
 "/home/oyjt/Scene-aware-BCI-VLA/Grounded-SAM-2/checkpoints",
 "/home/oyjt/Scene-aware-BCI-VLA/Grounded-SAM-2/Florence-2-large",
 "/home/oyjt/Grounded-SAM-2",
 "/home/oyjt/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt",
 "/home/oyjt/Grounded-SAM-2/Florence-2-large",
]
tokenizer = Path("/home/oyjt/.cache/openpi/big_vision/paligemma_tokenizer.model")
print(json.dumps({"scope":"read_only_complete_checkpoint_hash_and_bounded_path_checks",
 "checked_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "pi05_weight_hash":{"path":str(p),"bytes":before.st_size,"bytes_read":read,
    "duration_s":time.monotonic()-start,"complete":read==before.st_size,
    "metadata_stable":stable,"sha256":h.hexdigest() if stable else None},
 "tokenizer":{"path":str(tokenizer),"bytes":tokenizer.stat().st_size,
    "sha256":hashlib.sha256(tokenizer.read_bytes()).hexdigest()},
 "frontend_path_checks":[path_info(x) for x in paths],
 "not_performed":["model_load","inference","training","service_launch","remote_write","EEG_connection","robot_connection"]},ensure_ascii=True))
'''

FRONTEND_PATHS = PAYLOAD.partition('repo = ')[0] + r'''
tokenizer = Path("/home/oyjt/.cache/openpi/big_vision/paligemma_tokenizer.model")
paths = [
 "/home/oyjt/Scene-aware-BCI-VLA/Grounded-SAM-2/checkpoints",
 "/home/oyjt/Scene-aware-BCI-VLA/Grounded-SAM-2/Florence-2-large",
 "/home/oyjt/Grounded-SAM-2",
 "/home/oyjt/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt",
 "/home/oyjt/Grounded-SAM-2/Florence-2-large",
]
print(json.dumps({"scope":"read_only_small_asset_hash_and_bounded_path_checks",
 "checked_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "tokenizer":{"path":str(tokenizer),"bytes":tokenizer.stat().st_size,
    "sha256":hashlib.sha256(tokenizer.read_bytes()).hexdigest()},
 "frontend_path_checks":[path_info(x) for x in paths],
 "not_performed":["model_load","inference","training","service_launch","remote_write","EEG_connection","robot_connection"]},ensure_ascii=True))
'''

WORKSPACE = PAYLOAD.partition('repo = ')[0] + r'''
import os, time
workspace = Path("/home/oyjt/workspace")
start = time.monotonic()
entries = []
with os.scandir(workspace) as iterator:
    for item in iterator:
        row = {"name":item.name,"path":item.path,"is_symlink":item.is_symlink()}
        if item.is_symlink():
            row["link_target"] = os.readlink(item.path)
        entries.append(row)
roots = [workspace/e["name"] for e in entries if any(k in e["name"].lower()
    for k in ("sam", "florence", "model", "checkpoint", "weight", "rlinf", "openpi", "vla"))]
known_files = [workspace/"RLinf-RECAP/models/baseline_sft/RLinf-Pi05-LIBERO-SFT/model.safetensors",
    workspace/"Qwen2.5-VL-GR00T-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt",
    workspace/"Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt",
    workspace/"Grounded-SAM-2/Florence-2-large/model.safetensors"]
root_rows = []
for p in roots[:35]:
    if time.monotonic()-start > 18:
        break
    root_rows.append(path_info(str(p)))
files = []
for p in known_files:
    if time.monotonic()-start > 23:
        break
    files.append(path_info(str(p)))
print(json.dumps({"scope":"read_only_workspace_top_level_and_known_weight_paths",
 "checked_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "workspace_top_level":sorted(entries,key=lambda e:e["name"]),
 "candidate_roots":root_rows,"weight_and_config_matches":files,
 "entries_examined":len(entries),"scan_is_exhaustive":False,
 "weight_contents_loaded":False,
 "not_performed":["model_load","inference","training","service_launch","remote_write","EEG_connection","robot_connection"]},ensure_ascii=True))
'''

CANDIDATES = PAYLOAD.partition('repo = ')[0] + r'''
base = Path("/home/oyjt/workspace/checkpoints")
paths = [base/name for name in ("pi05_base", "pi05_base/params", "pi05_droid",
    "pi05_droid/params", "pi05_fold_clothes", "uniform", "uniform_rtc_v2",
    "uniform_table_rtc_v2")]
paths += [Path("/home/oyjt/Grounded-SAM-2/Florence-2-large/model.safetensors"),
    Path("/home/oyjt/Grounded-SAM-2/Florence-2-large/tokenizer.json"),
    Path("/home/oyjt/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")]
small = [Path("/home/oyjt/Grounded-SAM-2/Florence-2-large/config.json"),
    Path("/home/oyjt/Grounded-SAM-2/Florence-2-large/tokenizer_config.json"),
    Path("/home/oyjt/Grounded-SAM-2/grounded_sam2_florence2_image_demo.py")]
print(json.dumps({"scope":"read_only_exact_workspace_checkpoint_and_visual_weight_metadata",
    "checked_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "paths":[path_info(str(p)) for p in paths],
    "small_sources":[source_info(str(p)) for p in small],
    "candidate_selected_for_v4":False,"large_weights_hashed":False,
    "not_performed":["model_load","inference","training","service_launch","remote_write","EEG_connection","robot_connection"]},ensure_ascii=True))
'''


def main() -> None:
    if sys.argv[1:] not in ([], ["--supplement"], ["--integrity"], ["--paths"], ["--workspace"], ["--candidates"]):
        raise SystemExit("Allowed modes: default, --supplement, --integrity, --paths, --workspace or --candidates")
    supplement = sys.argv[1:] == ["--supplement"]
    integrity = sys.argv[1:] == ["--integrity"]
    paths_only = sys.argv[1:] == ["--paths"]
    workspace = sys.argv[1:] == ["--workspace"]
    candidates = sys.argv[1:] == ["--candidates"]
    output = (OUTPUT.with_name("01_candidate_weights.json") if candidates else
              OUTPUT.with_name("01_workspace_weights.json") if workspace else
              OUTPUT.with_name("01_remote_paths.json") if paths_only else
              OUTPUT.with_name("01_remote_integrity.json") if integrity else
              OUTPUT.with_name("01_remote_supplement.json") if supplement else OUTPUT)
    payload = CANDIDATES if candidates else WORKSPACE if workspace else FRONTEND_PATHS if paths_only else INTEGRITY if integrity else SUPPLEMENT if supplement else PAYLOAD
    if output.exists():
        raise SystemExit("Existing remote evidence must not be overwritten.")
    spec = importlib.util.spec_from_file_location(
        "private_remote_reader", ROOT / ".server-inspect" / "remote_readonly.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    client = module.connect()
    try:
        _, stdout, stderr = client.exec_command("python3 - <<'PY'\n" + payload + "\nPY\n", timeout=100 if integrity else 55)
        raw = stdout.read().decode("utf-8", errors="replace")
        errors = stderr.read().decode("utf-8", errors="replace")
        code = stdout.channel.recv_exit_status()
        if code:
            raise RuntimeError(f"Read-only inspection failed: exit={code}; {errors[:1000]}")
        result = json.loads(raw)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary = {"status": "saved", "path": str(output)}
        if candidates:
            summary["paths"] = result["paths"]
        elif workspace:
            summary["entries_examined"] = result["entries_examined"]
            summary["matches"] = len(result["weight_and_config_matches"])
            summary["roots"] = [{"path": r["path"], "exists": r["exists"],
                                 "resolved_path": r.get("resolved_path")}
                                for r in result["candidate_roots"]]
        elif paths_only:
            summary["tokenizer"] = result["tokenizer"]
            summary["frontend_paths"] = result["frontend_path_checks"]
        elif supplement or integrity:
            summary["pi05_weight_hash"] = result["pi05_weight_hash"]
        else:
            summary.update(sources=len(result["source_files"]), gpu=result["gpu"])
        print(json.dumps(summary, ensure_ascii=True))
    finally:
        client.close()


if __name__ == "__main__":
    main()
