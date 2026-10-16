"""Freeze local asset metadata; no training, cloud calls, or hardware imports."""
from pathlib import Path
import ast
import hashlib
import importlib.metadata
import json
import sys

root = Path(__file__).resolve().parents[1]
base = root.parent
assets = list((base / "context_intent").rglob("*"))
paper = base / "Scene_Aware_SSVEP_BCI_Controlled_Robotic_Manipulation_Framework_Integrated_with_Vision_Language_Action_Models (1)"
assets += [p for p in paper.rglob("*") if p.suffix.lower() in {".tex", ".bib"}]
assets += [base / "Scene-aware-BCI-VLA_仓库检查报告.md"]
records = []
for p in sorted(set(assets)):
    if not p.is_file() or "__pycache__" in p.parts or p.suffix == ".pyc":
        continue
    raw = p.read_bytes()
    record = {"path": str(p), "relative_to_workspace": str(p.relative_to(base)), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    if p.suffix == ".py":
        tree = ast.parse(raw.decode("utf-8-sig"))
        record["syntax"] = "PASS"
        record["top_level_symbols"] = [{"name": n.name, "line": n.lineno} for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))]
    records.append(record)
packages = {}
for name in ["torch", "numpy", "paramiko", "requests", "pytest"]:
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
result = {"plan_version": "V2", "scope": "local_asset_inventory", "local_git": "not_a_git_repository", "python": sys.version, "python_executable": sys.executable, "packages": packages, "files": records, "prototype_weight_files": [str(p) for p in (base / "context_intent").rglob("*") if p.is_file() and p.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors"}]}
(root / "execution/02_local_inventory.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"files_frozen": len(records), "python_ast_checked": sum("syntax" in r for r in records), "prototype_weight_files": result["prototype_weight_files"]}, ensure_ascii=True))
