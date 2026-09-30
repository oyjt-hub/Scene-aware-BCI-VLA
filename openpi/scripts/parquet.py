import pandas as pd
from pathlib import Path

# 替换成你的 data 文件夹的“根目录”路径
base_data_dir = "/home/oyjt/.cache/huggingface/lerobot/oy/pick_v21/data/chunk-000"

print(f"正在扫描文件夹: {base_data_dir} 下的所有 .parquet 文件...\n")

try:
    # 获取所有 parquet 文件，并按文件名（如 episode_000000.parquet）排序，保证输出整齐
    parquet_files = sorted(Path(base_data_dir).rglob("*.parquet"))
    
    if not parquet_files:
        print("❌ 没有找到任何 .parquet 文件，请检查路径。")
    else:
        print(f"✅ 找到 {len(parquet_files)} 个 parquet 文件。正在提取第一帧的 task_index...\n")
        
        # 打印表头
        print(f"{'文件名':<25} | {'Episode (视频序号)':<18} | {'Task Index (文本序号)'}")
        print("-" * 70)
        
        for file_path in parquet_files:
            try:
                # 为了加速读取，只读这三列，内存占用极小
                df = pd.read_parquet(file_path, columns=['frame_index', 'episode_index', 'task_index'])
                
                # 严谨起见：找到 frame_index == 0 的那一行（即第一帧）
                first_frame = df[df['frame_index'] == 0].iloc[0]
                
                episode_idx = int(first_frame['episode_index'])
                task_idx = int(first_frame['task_index'])
                
                # 格式化打印
                print(f"{file_path.name:<25} | {episode_idx:<18} | {task_idx}")
                
            except Exception as e:
                print(f"{file_path.name:<25} | ❌ 读取出错: {e}")
                
except Exception as e:
    print(f"扫描文件夹失败，错误信息: {e}")