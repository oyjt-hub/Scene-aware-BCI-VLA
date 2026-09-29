#!/usr/bin/env python3
"""
离线数据转换为 LeRobot 格式

将已有的机器人数据（如 rosbag, pkl, npy 等）转换为 LeRobot 格式，
支持后续使用 LeRobot 的可视化和训练工具。

使用方法:
1. 修改 load_your_data() 函数来读取你的数据格式
2. 运行: python convert_to_lerobot.py --repo_id your_username/dataset_name

示例数据结构 (可根据实际情况修改):
your_data/
├── episode_0/
│   ├── images_high/       # 顶部相机图像
│   ├── images_left/       # 左腕相机图像  
│   ├── images_right/      # 右腕相机图像
│   ├── states.npy         # 机器人状态 (N, 14)
│   └── actions.npy        # 动作 (N, 14)
├── episode_1/
│   └── ...
└── ...
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json

def load_your_data(data_dir: str):
    """
    加载你的数据 - 根据实际数据格式修改此函数
    
    Returns:
        list of dict: 每个 episode 的数据
        [
            {
                "images_high": [img1, img2, ...],  # list of (H, W, C) uint8 RGB
                "images_left": [img1, img2, ...],
                "images_right": [img1, img2, ...],
                "states": np.array (N, 14),
                "actions": np.array (N, 14),
                "task": "task description",
            },
            ...
        ]
    """
    data_path = Path(data_dir)
    episodes = []
    
    # 示例：假设数据组织为 episode_0, episode_1, ...
    episode_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith("episode")])
    
    for ep_dir in episode_dirs:
        print(f"加载: {ep_dir.name}")
        
        # 读取状态和动作
        states = np.load(ep_dir / "states.npy")  # (N, 14)
        actions = np.load(ep_dir / "actions.npy")  # (N, 14)
        
        # 读取图像
        def load_images(img_dir):
            imgs = []
            img_files = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
            for f in img_files:
                img = np.array(Image.open(f).convert("RGB"))
                imgs.append(img)
            return imgs
        
        images_high = load_images(ep_dir / "images_high")
        images_left = load_images(ep_dir / "images_left") 
        images_right = load_images(ep_dir / "images_right")
        
        # 读取任务描述（可选）
        task_file = ep_dir / "task.txt"
        if task_file.exists():
            task = task_file.read_text().strip()
        else:
            task = "bimanual manipulation task"
        
        episodes.append({
            "images_high": images_high,
            "images_left": images_left,
            "images_right": images_right,
            "states": states,
            "actions": actions,
            "task": task,
        })
    
    return episodes


def load_from_pkl(pkl_path: str):
    """
    从 pkl 文件加载数据
    
    假设 pkl 格式为:
    [
        {
            "obs": {
                "images": {"cam_high": ..., "cam_left": ..., "cam_right": ...},
                "state": np.array (N, 14)
            },
            "actions": np.array (N, 14),
            "task": str
        },
        ...
    ]
    """
    import pickle
    
    with open(pkl_path, "rb") as f:
        raw_data = pickle.load(f)
    
    episodes = []
    for ep in raw_data:
        episodes.append({
            "images_high": ep["obs"]["images"]["cam_high"],
            "images_left": ep["obs"]["images"]["cam_left"],
            "images_right": ep["obs"]["images"]["cam_right"],
            "states": ep["obs"]["state"],
            "actions": ep["actions"],
            "task": ep.get("task", "bimanual manipulation task"),
        })
    
    return episodes


def load_from_rosbag(bag_path: str, topic_config: dict = None):
    """
    从 rosbag 加载数据
    
    需要安装: pip install rosbag bagpy
    """
    try:
        import rosbag
        from cv_bridge import CvBridge
    except ImportError:
        raise ImportError("请安装 rosbag: pip install rosbag bagpy")
    
    if topic_config is None:
        topic_config = {
            'img_high': '/camera_f/color/image_raw',
            'img_left': '/camera_l/color/image_raw',
            'img_right': '/camera_r/color/image_raw',
            'puppet_left': '/puppet/joint_left',
            'puppet_right': '/puppet/joint_right',
        }
    
    bridge = CvBridge()
    bag = rosbag.Bag(bag_path)
    
    # 收集所有消息
    img_high_msgs = []
    img_left_msgs = []
    img_right_msgs = []
    state_left_msgs = []
    state_right_msgs = []
    
    for topic, msg, t in bag.read_messages():
        ts = t.to_sec()
        if topic == topic_config['img_high']:
            img_high_msgs.append((ts, bridge.imgmsg_to_cv2(msg, "rgb8")))
        elif topic == topic_config['img_left']:
            img_left_msgs.append((ts, bridge.imgmsg_to_cv2(msg, "rgb8")))
        elif topic == topic_config['img_right']:
            img_right_msgs.append((ts, bridge.imgmsg_to_cv2(msg, "rgb8")))
        elif topic == topic_config['puppet_left']:
            state_left_msgs.append((ts, np.array(msg.position)))
        elif topic == topic_config['puppet_right']:
            state_right_msgs.append((ts, np.array(msg.position)))
    
    bag.close()
    
    # 时间同步 - 简单策略：使用 img_high 的时间戳作为参考
    def find_nearest(msgs, target_ts):
        idx = np.argmin([abs(m[0] - target_ts) for m in msgs])
        return msgs[idx][1]
    
    images_high = []
    images_left = []
    images_right = []
    states = []
    
    for ts, img in img_high_msgs:
        images_high.append(img)
        images_left.append(find_nearest(img_left_msgs, ts))
        images_right.append(find_nearest(img_right_msgs, ts))
        
        state_l = find_nearest(state_left_msgs, ts)
        state_r = find_nearest(state_right_msgs, ts)
        states.append(np.concatenate([state_l, state_r]))
    
    states = np.array(states)
    # 计算动作（简单使用下一帧状态作为动作）
    actions = np.zeros_like(states)
    actions[:-1] = states[1:]
    actions[-1] = states[-1]
    
    return [{
        "images_high": images_high,
        "images_left": images_left,
        "images_right": images_right,
        "states": states,
        "actions": actions,
        "task": "bimanual manipulation task",
    }]


def convert_to_lerobot(
    episodes: list,
    repo_id: str,
    fps: int = 20,
    image_size: tuple = (224, 224),
    push_to_hub: bool = False,
):
    """
    将数据转换为 LeRobot 格式
    
    Args:
        episodes: 数据列表，每个元素是一个 episode 的数据字典
        repo_id: HuggingFace repo ID
        fps: 帧率
        image_size: 图像尺寸 (H, W)
        push_to_hub: 是否上传到 HuggingFace Hub
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        raise ImportError("请安装 lerobot: pip install lerobot")
    
    # 定义 features
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": ["left_j0", "left_j1", "left_j2", "left_j3", "left_j4", "left_j5", "left_grip",
                     "right_j0", "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_grip"],
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": ["left_j0", "left_j1", "left_j2", "left_j3", "left_j4", "left_j5", "left_grip",
                     "right_j0", "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_grip"],
        },
        "observation.images.cam_high": {
            "dtype": "video",
            "shape": (3, image_size[0], image_size[1]),
            "names": ["channel", "height", "width"],
        },
        "observation.images.cam_left_wrist": {
            "dtype": "video",
            "shape": (3, image_size[0], image_size[1]),
            "names": ["channel", "height", "width"],
        },
        "observation.images.cam_right_wrist": {
            "dtype": "video",
            "shape": (3, image_size[0], image_size[1]),
            "names": ["channel", "height", "width"],
        },
    }
    
    print(f">>> 创建 LeRobot 数据集: {repo_id}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        robot_type="bimanual_arm",
    )
    
    # 图像预处理
    def process_img(img, size):
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((size[1], size[0]), Image.BILINEAR)
        return np.array(pil_img).transpose(2, 0, 1)  # HWC -> CHW
    
    # 转换每个 episode
    total_frames = 0
    for ep_idx, ep in enumerate(tqdm(episodes, desc="转换 episodes")):
        n_frames = len(ep["states"])
        
        for i in range(n_frames):
            frame_data = {
                "observation.state": ep["states"][i].astype(np.float32),
                "action": ep["actions"][i].astype(np.float32),
                "observation.images.cam_high": process_img(ep["images_high"][i], image_size),
                "observation.images.cam_left_wrist": process_img(ep["images_left"][i], image_size),
                "observation.images.cam_right_wrist": process_img(ep["images_right"][i], image_size),
                "task": ep["task"],
            }
            dataset.add_frame(frame_data)
        
        dataset.save_episode()
        total_frames += n_frames
        print(f"  Episode {ep_idx}: {n_frames} frames")
    
    # 完成
    print(f">>> 正在完成数据集...")
    dataset.finalize()
    
    if push_to_hub:
        print(f">>> 上传到 HuggingFace Hub...")
        dataset.push_to_hub()
        print(f">>> 完成: https://huggingface.co/datasets/{repo_id}")
    
    print(f"\n{'='*50}")
    print(f"LeRobot 数据集创建完成!")
    print(f"  - Repo ID: {repo_id}")
    print(f"  - Episodes: {len(episodes)}")
    print(f"  - Total Frames: {total_frames}")
    print(f"  - FPS: {fps}")
    print(f"  - 本地路径: {dataset.root}")
    print(f"\n可视化方式:")
    print(f"  1. 本地: lerobot-dataset-viz --repo-id {repo_id} --episode-index 0")
    print(f"  2. 在线: https://huggingface.co/spaces/lerobot/visualize_dataset")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description='将数据转换为 LeRobot 格式')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据目录 (用于 load_your_data)')
    parser.add_argument('--pkl_path', type=str, default=None,
                        help='PKL 文件路径 (用于 load_from_pkl)')
    parser.add_argument('--bag_path', type=str, default=None,
                        help='ROS bag 文件路径 (用于 load_from_rosbag)')
    parser.add_argument('--repo_id', type=str, required=True,
                        help='HuggingFace 数据集 repo ID')
    parser.add_argument('--fps', type=int, default=20, help='帧率')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224],
                        help='图像尺寸 H W')
    parser.add_argument('--push_to_hub', action='store_true',
                        help='上传到 HuggingFace Hub')
    args = parser.parse_args()
    
    # 根据参数选择数据加载方式
    if args.pkl_path:
        print(f">>> 从 PKL 加载: {args.pkl_path}")
        episodes = load_from_pkl(args.pkl_path)
    elif args.bag_path:
        print(f">>> 从 ROS bag 加载: {args.bag_path}")
        episodes = load_from_rosbag(args.bag_path)
    else:
        print(f">>> 从目录加载: {args.data_dir}")
        episodes = load_your_data(args.data_dir)
    
    print(f">>> 加载了 {len(episodes)} 个 episodes")
    
    convert_to_lerobot(
        episodes=episodes,
        repo_id=args.repo_id,
        fps=args.fps,
        image_size=tuple(args.image_size),
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
