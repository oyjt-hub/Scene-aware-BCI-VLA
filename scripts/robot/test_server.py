 ,   # import pandas as pd
# from pathlib import Path

# root = Path('/home/agilex/.cache/huggingface/lerobot/zjj/test9')

# # 读取 parquet 数据
# data_files = list(root.glob('data/**/*.parquet'))
# print(f'=== Parquet 文件 ===')
# for f in data_files:
#     print(f'  {f}')
#     df = pd.read_parquet(f)
#     print(f'  行数: {len(df)}')
#     print(f'  列名: {list(df.columns)}')
#     print(f'  前3行:')
#     print(df.head(3))
#     print()

# # 读取 meta
# print(f'=== Meta 文件 ===')
# meta_files = list(root.glob('meta/**/*'))
# for f in meta_files:
#     print(f'  {f}')


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_parquet('/home/agilex/.cache/huggingface/lerobot/zjj/test30/data/chunk-000/file-000.parquet')
# 9.30.32 z7 
states = np.stack(df['observation.state'].values)
actions_raw = np.stack(df['action.raw'].values)
actions_processed = np.stack(df['action'].values)
timestamps = df['timestamp'].values

print(f'数据: {len(df)} 帧, {len(df)/20:.1f} 秒')

# 创建图表
fig, axes = plt.subplots(4, 2, figsize=(16, 12))

# 左臂状态
axes[0,0].plot(timestamps, states[:, :6])
axes[0,0].set_title('Left Arm State (Joints)')
axes[0,0].set_ylabel('Position')
axes[0,0].legend([f'J{i}' for i in range(6)], loc='upper right', fontsize=8)

# 右臂状态
axes[0,1].plot(timestamps, states[:, 7:13])
axes[0,1].set_title('Right Arm State (Joints)')
axes[0,1].legend([f'J{i}' for i in range(6)], loc='upper right', fontsize=8)

# 左臂动作对比
axes[1,0].plot(timestamps, actions_raw[:, :6], '--', alpha=0.5)
axes[1,0].set_prop_cycle(None)
axes[1,0].plot(timestamps, actions_processed[:, :6])
axes[1,0].set_title('Left Arm Action (dashed=raw, solid=processed)')
axes[1,0].set_ylabel('Action')

# 右臂动作对比
axes[1,1].plot(timestamps, actions_raw[:, 7:13], '--', alpha=0.5)
axes[1,1].set_prop_cycle(None)
axes[1,1].plot(timestamps, actions_processed[:, 7:13])
axes[1,1].set_title('Right Arm Action (dashed=raw, solid=processed)')

# 夹爪对比
axes[2,0].plot(timestamps, states[:, 6]*10, label='State', linewidth=2)
axes[2,0].plot(timestamps, actions_raw[:, 6], '--', label='Action Raw', alpha=0.9)
axes[2,0].plot(timestamps, actions_processed[:, 6]*10, label='Action Processed', alpha=0.7)
axes[2,0].set_title('Left Gripper')
axes[2,0].set_ylabel('Gripper')
axes[2,0].legend()

axes[2,1].plot(timestamps, states[:, 13]*10, label='State', linewidth=2)
axes[2,1].plot(timestamps, actions_raw[:, 13], '--', label='Action Raw', alpha=0.9)
axes[2,1].plot(timestamps, actions_processed[:, 13]*10, label='Action Processed', alpha=0.7)
axes[2,1].set_title('Right Gripper')
axes[2,1].legend()

# 动作差异 (processed - raw)
diff = actions_processed - actions_raw
axes[3,0].plot(timestamps, diff[:, :7])
axes[3,0].set_title('Left Arm: Action Difference (processed - raw)')
axes[3,0].set_xlabel('Time (s)')
axes[3,0].set_ylabel('Difference')

axes[3,1].plot(timestamps, diff[:, 7:])
axes[3,1].set_title('Right Arm: Action Difference (processed - raw)')
axes[3,1].set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig('/tmp/test9_visualization.png', dpi=150)
print('✅ 图片保存到 /tmp/test9_visualization.png')


# from torchcodec.decoders import VideoDecoder
# print('✅ torchcodec 加载成功')

# # 测试解码
# decoder = VideoDecoder('/home/agilex/.cache/huggingface/lerobot/zjj/test9/videos/observation.images.cam_high/chunk-000/file-000.mp4')
# print(f'视频帧数: {len(decoder)}')
# frame = decoder[0]
# print(f'第一帧: {frame.shape}')