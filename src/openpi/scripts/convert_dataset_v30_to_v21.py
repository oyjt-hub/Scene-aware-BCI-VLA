import argparse
import logging
import shutil
import subprocess
import json
from pathlib import Path
import pandas as pd
import tqdm
import numpy as np
import jsonlines
from huggingface_hub import snapshot_download
import sys
import os

# --- 1. 兼容性导入 ---
try:
    from lerobot.constants import HF_LEROBOT_HOME
except ImportError:
    try:
        from lerobot.utils.constants import HF_LEROBOT_HOME
    except ImportError:
        HF_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot"))

from lerobot.datasets.utils import load_info, write_info, DEFAULT_CHUNK_SIZE
from lerobot.utils.utils import init_logging

V21 = "v2.1"
V30 = "v3.0"

# --- 2. 辅助函数 ---
def save_jsonlines(data: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, "w") as writer:
        writer.write_all(data)

def get_v2_chunk_path(root: Path, episode_index: int, type_dir: str = "data"):
    chunk_idx = episode_index // DEFAULT_CHUNK_SIZE
    chunk_name = f"chunk-{chunk_idx:03d}"
    return root / type_dir / chunk_name

def validate_local_dataset_version(local_path: Path) -> None:
    info = load_info(local_path)
    dataset_version = info.get("codebase_version", "unknown")
    if dataset_version != V30:
        raise ValueError(f"Local dataset has codebase version '{dataset_version}', expected '{V30}'.")

def safe_list(val):
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, np.ndarray):
        if val.ndim == 0:
            return [val.item()]
        return val.tolist()
    return [val]

# --- 3. 核心转换逻辑 ---

def convert_tasks(root: Path, new_root: Path):
    logging.info("Converting tasks...")
    all_tasks = []
    
    # 1. 扩大搜索范围，使用 rglob 递归查找所有 tasks*.parquet
    task_files = list((root / "meta").rglob("tasks*.parquet"))
    
    for f in task_files:
        try:
            df = pd.read_parquet(f)
            logging.info(f"Processing task file: {f}")
            
            for index_val, row in df.iterrows():
                # 处理逻辑：
                # 情况 A: 'task' 在列里
                # 情况 B: 'task' 在 Index 里 (根据你 check_index.py 的输出，属于这种情况)
                task_str = ""
                if "task" in df.columns:
                    task_str = str(row["task"])
                else:
                    task_str = str(index_val) # 如果列里没有，尝试取 index 里的字符串
                
                # 获取 task_index
                t_idx = int(row["task_index"]) if "task_index" in df.columns else 0
                
                if task_str:
                    all_tasks.append({
                        "task_index": t_idx,
                        "task": task_str
                    })
        except Exception as e:
            logging.warning(f"Error processing {f}: {e}")
    
    # 2. 如果真的没搜到，再用保底
    if not all_tasks:
        logging.warning("⚠️ No tasks found. Creating default task 'pick coffee'.")
        all_tasks.append({"task_index": 0, "task": "pick coffee"})
    
    # 3. 去重并排序
    unique_tasks = {t["task_index"]: t for t in all_tasks}.values()
    all_tasks = sorted(list(unique_tasks), key=lambda x: x["task_index"])
    
    save_jsonlines(all_tasks, new_root / "meta/tasks.jsonl")
    logging.info(f"✅ Successfully saved {len(all_tasks)} tasks to tasks.jsonl")

def convert_episodes_metadata(root: Path, new_root: Path):
    logging.info("Converting episodes metadata...")
    episodes_dir = root / "meta/episodes"
    if not episodes_dir.exists(): episodes_dir = root / "meta"
        
    ep_files = sorted(episodes_dir.glob("**/*.parquet"))
    full_meta_df_list = []
    all_episodes_meta = []
    
    default_task_string = "pick coffee"

    for f in tqdm.tqdm(ep_files, desc="Parsing metadata"):
        try:
            df = pd.read_parquet(f)
            full_meta_df_list.append(df)
            for _, row in df.iterrows():
                ep_idx = int(row["episode_index"])
                
                tasks = []
                if "tasks" in row and row["tasks"] is not None:
                    raw = row["tasks"]
                    tasks = raw.tolist() if hasattr(raw, "tolist") else ([raw] if not isinstance(raw, list) else raw)
                
                if not tasks: tasks = [default_task_string]
                
                all_episodes_meta.append({
                    "episode_index": ep_idx,
                    "tasks": tasks,
                    "length": int(row["length"])
                })
        except Exception as e:
            logging.warning(f"Error reading {f}: {e}")

    all_episodes_meta.sort(key=lambda x: x["episode_index"])
    save_jsonlines(all_episodes_meta, new_root / "meta/episodes.jsonl")
    
    return pd.concat(full_meta_df_list, ignore_index=True) if full_meta_df_list else pd.DataFrame()

# =========================================================================
#  Transformation Functions (Aligning Scale to 0~0.7)
# =========================================================================

# 1. State (Observation) Transformation
#    Input: Raw int (e.g. 69370 for gripper)
#    Target: ~0.7
#    Math: / 1000 / 100
def transform_observation(raw_val):
    # Global / 1000
    arr = np.array(raw_val, dtype=np.float32, ndmin=1) 
    # deg2rad = np.pi / 180.0
    
    # Joints -> Radians
    # if len(arr) > 0: arr[0:min(6, len(arr))] *= deg2rad
    # if len(arr) > 7: arr[7:min(13, len(arr))] *= deg2rad
    

    # if len(arr) > 6: arr[6] /= 10
    # if len(arr) > 13: arr[13] /= 10
    
    return arr

# 2. Action Transformation
#    Input: Meters (e.g. 0.074 for gripper)
#    Target: ~0.7 (to match Observation)
#    Math: * 10
def transform_action(raw_val):
    arr = np.array(raw_val, dtype=np.float32, ndmin=1)
    
    # Joints: ALREADY Radians (Do NOT touch)
    # Action joints are usually stored as radians in raw datasets
    
    # Grippers: Meters -> Decimeters (* 10)
    # 0.074 -> 0.74
    # if len(arr) > 6: arr[6] *= 100.0
    # if len(arr) > 13: arr[13] *= 100.0
    
    return arr

def convert_data_and_stats(root: Path, new_root: Path, meta_df: pd.DataFrame):
    logging.info("Converting tabular data and computing stats...")
    
    if "data/episode_chunk" not in meta_df.columns:
        meta_df["data/episode_chunk"] = meta_df.get("episode_chunk", 0)
    if "data/file_index" not in meta_df.columns:
        meta_df["data/file_index"] = meta_df.get("file_index", meta_df["episode_index"])

    grouped = meta_df.groupby(["data/episode_chunk", "data/file_index"])
    computed_stats = []

    for (chunk_idx, file_idx), group in tqdm.tqdm(grouped, desc="Processing chunks"):
        chunk_dir = root / "data" / f"chunk-{int(chunk_idx):03d}"
        found_file = None
        idx = int(file_idx)
        
        candidates = [
            chunk_dir / f"file-{idx:03d}.parquet", 
            chunk_dir / f"file_{idx:04d}.parquet",
            chunk_dir / f"file_{idx}.parquet",
            root / "data" / f"file-{idx:03d}.parquet"
        ]
        
        for p in candidates:
            if p.exists(): found_file = p; break
        
        if not found_file and chunk_dir.exists():
            matches = list(chunk_dir.glob(f"*{idx}*.parquet"))
            if matches: found_file = matches[0]

        if not found_file:
            logging.warning(f"❌ Data file not found! Chunk={chunk_idx}, File={file_idx}")
            continue
            
        try:
            big_df = pd.read_parquet(found_file)
        except Exception: continue

        group = group.sort_values("episode_index")
        current_offset = 0
        
        for _, row in group.iterrows():
            ep_idx = int(row["episode_index"])
            length = int(row["length"])
            
            start = current_offset
            ep_df = big_df.iloc[start : start + length].copy()
            current_offset += length

            # -------------------------------------------------
            # Apply Separate Transformations for State & Action
            # -------------------------------------------------
            
            # 1. Observation: Raw(Large Int) -> 0.7
            if "observation.state" in ep_df.columns:
                try:
                    ep_df["observation.state"] = ep_df["observation.state"].apply(transform_observation)
                except Exception as e:
                    logging.warning(f"Obs conversion error ep {ep_idx}: {e}")

            # 2. Action: Meters(Small Float) -> 0.7
            if "action" in ep_df.columns:
                try:
                    ep_df["action"] = ep_df["action"].apply(transform_action)
                except Exception as e:
                    logging.warning(f"Action conversion error ep {ep_idx}: {e}")

            # -------------------------------------------------
            
            # Compute Stats
            ep_stats = {}
            for col in ep_df.columns:
                if col in ["index", "timestamp", "frame_index", "episode_index", "task_index", "dataset_from_index", "dataset_to_index"]: continue
                if "image" in col: continue

                try:
                    col_values = ep_df[col].tolist()
                    data = np.vstack(col_values) 
                    
                    if not np.issubdtype(data.dtype, np.number): continue

                    ep_stats[col] = {
                        "min": safe_list(data.min(axis=0)),
                        "max": safe_list(data.max(axis=0)),
                        "mean": safe_list(data.mean(axis=0)),
                        "std": safe_list(data.std(axis=0)),
                        "count": int(len(data))
                    }
                except Exception: pass
            
            if ep_stats:
                computed_stats.append({"episode_index": ep_idx, "stats": ep_stats})

            # Save
            target_dir = get_v2_chunk_path(new_root, ep_idx, "data")
            target_dir.mkdir(parents=True, exist_ok=True)
            ep_df.to_parquet(target_dir / f"episode_{ep_idx:06d}.parquet", index=False)
            
    return computed_stats

def aggregate_and_save_global_stats(stats_list, output_path):
    if not stats_list: return
    
    logging.info("🔥 Aggregating global stats.json manually...")
    first_ep = stats_list[0]["stats"]
    global_stats = {}
    
    all_stats_dicts = [item["stats"] for item in stats_list]

    for key, _ in first_ep.items():
        global_stats[key] = {}
        total_count = sum(ep.get(key, {}).get("count", 0) for ep in all_stats_dicts)
        global_stats[key]["count"] = total_count
        
        try:
            all_mins = np.array([ep[key]["min"] for ep in all_stats_dicts])
            global_stats[key]["min"] = safe_list(all_mins.min(axis=0))
        except: global_stats[key]["min"] = [0.0]

        try:
            all_maxs = np.array([ep[key]["max"] for ep in all_stats_dicts])
            global_stats[key]["max"] = safe_list(all_maxs.max(axis=0))
        except: global_stats[key]["max"] = [0.0]
        
        try:
            all_means = np.array([ep[key]["mean"] for ep in all_stats_dicts])
            all_counts = np.array([ep[key]["count"] for ep in all_stats_dicts]).reshape(-1, 1)
            if all_means.ndim > 1 and all_counts.ndim == 2:
                weighted_sum = (all_means * all_counts).sum(axis=0)
                global_stats[key]["mean"] = safe_list(weighted_sum / total_count)
            else:
                global_stats[key]["mean"] = safe_list(all_means.mean(axis=0))
        except: global_stats[key]["mean"] = [0.0]

        try:
            all_stds = np.array([ep[key]["std"] for ep in all_stats_dicts])
            global_stats[key]["std"] = safe_list(all_stds.mean(axis=0))
        except: global_stats[key]["std"] = [1.0]

    with open(output_path, "w") as f:
        json.dump(global_stats, f, indent=4)
    logging.info(f"✅ Created global ticket: {output_path}")

def convert_videos(root: Path, new_root: Path, meta_df: pd.DataFrame):
    logging.info("Converting videos...")
    info = load_info(root)
    features = info.get("features", {})
    video_keys = [k for k, v in features.items() if v["dtype"] == "video"]
    
    for cam_key in video_keys:
        logging.info(f"Processing {cam_key}")
        col_chunk = f"videos/{cam_key}/episode_chunk"
        col_file = f"videos/{cam_key}/file_index"
        
        if col_chunk not in meta_df.columns and "data/episode_chunk" in meta_df.columns:
             meta_df[col_chunk] = meta_df["data/episode_chunk"]
             meta_df[col_file] = meta_df["data/file_index"]

        if col_chunk not in meta_df.columns: 
            logging.warning(f"⚠️ Missing video metadata columns for {cam_key}, skipping.")
            continue
        
        grouped = meta_df.groupby([col_chunk, col_file])
        for (chunk_idx, file_idx), group in tqdm.tqdm(grouped, desc=f"Splitting {cam_key}"):
            idx, c_idx = int(file_idx), int(chunk_idx)
            
            paths = [
                root / "videos" / cam_key / f"chunk-{c_idx:03d}" / f"file-{idx:03d}.mp4",
                root / "videos" / cam_key / f"file-{idx:03d}.mp4",
                root / "videos" / cam_key / f"file_{idx:04d}.mp4",
            ]
            found = next((p for p in paths if p.exists()), None)
            if not found and (root / "videos" / cam_key).exists():
                found = next((root / "videos" / cam_key).glob(f"*{idx}*.mp4"), None)
            
            if not found: continue

            for _, row in group.iterrows():
                ep_idx = int(row["episode_index"])
                t_start = row[f"videos/{cam_key}/from_timestamp"]
                duration = row[f"videos/{cam_key}/to_timestamp"] - t_start
                
                target = get_v2_chunk_path(new_root, ep_idx, "videos") / cam_key
                target.mkdir(parents=True, exist_ok=True)
                
                subprocess.run([
                    "ffmpeg", "-y", "-loglevel", "error", "-ss", str(t_start),
                    "-i", str(found), "-t", str(duration),
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
                    str(target / f"episode_{ep_idx:06d}.mp4")
                ], check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    init_logging()
    root = HF_LEROBOT_HOME / args.repo_id if args.root is None else Path(args.root) / args.repo_id
    
    if not root.exists():
        snapshot_download(args.repo_id, repo_type="dataset", revision=V30, local_dir=root)
    validate_local_dataset_version(root)
    
    new_root = root.parent / f"{root.name}_v21"
    if new_root.exists():
        if args.force: shutil.rmtree(new_root)
        else: raise FileExistsError(f"{new_root} exists. Use --force.")
            
    logging.info(f"Start Conversion: {root} -> {new_root}")
    
    # 1. Info
    info = load_info(root)
    info["codebase_version"] = V21
    info.pop("data_files_size_in_mb", None)
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    write_info(info, new_root)
    
    # 2. Tasks & Meta
    convert_tasks(root, new_root)
    meta_df = convert_episodes_metadata(root, new_root)
    
    # 3. Data & Stats
    computed_stats = convert_data_and_stats(root, new_root, meta_df)
    
    if computed_stats:
        computed_stats.sort(key=lambda x: x["episode_index"])
        save_jsonlines(computed_stats, new_root / "meta/episodes_stats.jsonl")
        aggregate_and_save_global_stats(computed_stats, new_root / "meta/stats.json")
    else:
        logging.warning("⚠️ No stats computed!")

    # 4. Videos
    convert_videos(root, new_root, meta_df)
    
    logging.info("🎉 All Done!")
    logging.info(f"New dataset path: {new_root}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv = ["script.py", "--repo-id", "oy/table", "--force"]
    main()