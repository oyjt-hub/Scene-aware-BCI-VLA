import pandas as pd
from pathlib import Path

# 尝试读取 tasks 文件
p = Path("/home/oyjt/.cache/huggingface/lerobot/oy/pick/meta/tasks.parquet")
if not p.exists():
    # 尝试在子文件夹里找
    p = list(Path("/home/oyjt/.cache/huggingface/lerobot/oy/pick/meta/tasks").glob("*.parquet"))[0]

df = pd.read_parquet(p)
print(df) # 这里会打印出 0, 1, 2 分别对应什么任务