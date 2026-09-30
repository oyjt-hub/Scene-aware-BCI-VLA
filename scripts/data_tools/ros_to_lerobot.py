#!/usr/bin/env python3
"""
ROS OpenPI 推理节点 + LeRobot 数据集录制器 (交互版)

功能:
1. 执行推理控制逻辑
2. 归零后等待用户按 Enter 开始录制
3. Ctrl+C 结束当前 episode
4. 记录:
   - observation.state: 当前机器人实际状态
   - action.raw: 模型直接输出的动作 (前处理)
   - action: 实际发送给机器人的动作 (后处理，经过 ensemble + gripper scaling)

使用方法:
1. 安装: pip install lerobot
2. 运行: python ros_to_lerobot_interactive.py --repo_id ${HF_USER}/my_dataset

交互流程:
1. 启动后自动归零
2. 按 Enter 开始录制 Episode
3. 按 Ctrl+C 结束当前 Episode
4. 按 Enter 继续下一个 Episode，或输入 'q' 退出并保存
"""

import rospy
import threading
import time
import argparse
import signal
import sys
import numpy as np
from collections import deque
from pathlib import Path
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Header, Bool
from cv_bridge import CvBridge

from openpi_client import image_tools, websocket_client_policy

# ================= 配置区域 =================
SERVER_HOST = "172.19.1.40"
SERVER_PORT = 9000
CTRL_FREQ = 20  # 20Hz

# myprompt = "picking up a paper cup with the left hand and a bottle of nongfu spring mineral water with the right hand, pouring the water into the paper cup, and then placing them back on the table after completing."
myprompt = "pick up the cherry and put it into the white bowl."
GRIPPER_SCALE = 0.1

HOME_POSE_LEFT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08]
HOME_POSE_RIGHT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08]

TOPIC_CONFIG = {
    'img_high': '/camera_f/color/image_raw',
    'img_left': '/camera_l/color/image_raw',
    'img_right': '/camera_r/color/image_raw',
    'puppet_left': '/puppet/joint_left',
    'puppet_right': '/puppet/joint_right',
    'cmd_left': '/master/joint_left',
    'cmd_right': '/master/joint_right',
    'enable': '/enable_flag'
}


# ================= 全局状态 =================
class RecordingState:
    """录制状态管理"""
    def __init__(self):
        self.recording = False
        self.should_stop_episode = False
        self.should_quit = False
        self.lock = threading.Lock()
        
    def start_recording(self):
        with self.lock:
            self.recording = True
            self.should_stop_episode = False
            
    def stop_episode(self):
        with self.lock:
            self.should_stop_episode = True
            
    def is_recording(self):
        with self.lock:
            return self.recording and not self.should_stop_episode
    
    def end_recording(self):
        with self.lock:
            self.recording = False
            self.should_stop_episode = False


recording_state = RecordingState()


def signal_handler(sig, frame):
    """处理 Ctrl+C"""
    if recording_state.recording:
        print("\n>>> [Ctrl+C] 停止当前 Episode...")
        recording_state.stop_episode()
    else:
        print("\n>>> [Ctrl+C] 退出程序...")
        recording_state.should_quit = True
        sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


# ================= LeRobot 数据集录制器 =================
class LeRobotRecorder:
    """
    将 ROS 数据录制为 LeRobot 格式
    
    记录的数据:
    - observation.state: 机器人当前实际状态 (14,)
    - action.raw: 模型原始输出 (14,) - 未经后处理
    - action: 实际发送的动作 (14,) - 经过 ensemble + gripper scaling
    - observation.images.*: 三个相机的图像
    """
    
    def __init__(self, repo_id: str, fps: int = 20, task: str = None, 
                 image_size: tuple = (224, 224)):
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            self.LeRobotDataset = LeRobotDataset
        except ImportError:
            raise ImportError("请安装 lerobot: pip install lerobot")
        
        self.repo_id = repo_id
        self.fps = fps
        self.task = task or myprompt
        self.image_size = image_size
        self.episode_count = 0
        self.frame_count = 0
        self.dataset = None
        
        # 定义 features - 包含 raw action 和 processed action
        self.features = {
            # 观测：机器人当前实际状态
            "observation.state": {
                "dtype": "float32",
                "shape": (14,),
                "names": {
                    "motors": ["left_j0", "left_j1", "left_j2", "left_j3", 
                              "left_j4", "left_j5", "left_gripper",
                              "right_j0", "right_j1", "right_j2", "right_j3",
                              "right_j4", "right_j5", "right_gripper"]
                },
            },
            # 模型原始输出 (未经后处理)
            "action.raw": {
                "dtype": "float32",
                "shape": (14,),
                "names": {
                    "motors": ["left_j0", "left_j1", "left_j2", "left_j3",
                              "left_j4", "left_j5", "left_gripper",
                              "right_j0", "right_j1", "right_j2", "right_j3",
                              "right_j4", "right_j5", "right_gripper"]
                },
            },
            # 实际发送给机器人的动作 (经过后处理)
            "action": {
                "dtype": "float32",
                "shape": (14,),
                "names": {
                    "motors": ["left_j0", "left_j1", "left_j2", "left_j3",
                              "left_j4", "left_j5", "left_gripper",
                              "right_j0", "right_j1", "right_j2", "right_j3",
                              "right_j4", "right_j5", "right_gripper"]
                },
            },
            # 图像观测
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
        
        self._create_dataset()
        
    def _create_dataset(self):
        """创建数据集"""
        print(f">>> 创建 LeRobot 数据集: {self.repo_id}")
        self.dataset = self.LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            features=self.features,
            robot_type="bimanual_arm",
        )
        print(f">>> 数据集初始化完成")
        print(f"    - 本地路径: {self.dataset.root}")
        
    def start_episode(self):
        """开始新的 episode"""
        self.frame_count = 0
        print(f"\n>>> ▶️  开始录制 Episode {self.episode_count}")
        print(f"    按 Ctrl+C 结束当前 Episode")
        
    def add_frame(self, img_high: np.ndarray, img_left: np.ndarray, 
                  img_right: np.ndarray, state: np.ndarray, 
                  action_raw: np.ndarray, action_processed: np.ndarray):
        """
        添加一帧数据
        
        Args:
            img_high: 顶部相机图像 (H, W, C) uint8 RGB
            img_left: 左腕相机图像 (H, W, C) uint8 RGB
            img_right: 右腕相机图像 (H, W, C) uint8 RGB
            state: 机器人当前实际状态 (14,) float32
            action_raw: 模型原始输出 (14,) float32 - 未经后处理
            action_processed: 实际发送的动作 (14,) float32 - 经过后处理
        """
        from PIL import Image as PILImage
        
        def process_img(img):
            pil_img = PILImage.fromarray(img)
            pil_img = pil_img.resize((self.image_size[1], self.image_size[0]), 
                                     PILImage.BILINEAR)
            return np.array(pil_img).transpose(2, 0, 1)  # HWC -> CHW
        
        frame_data = {
            "observation.state": state.astype(np.float32),
            "action.raw": action_raw.astype(np.float32),
            "action": action_processed.astype(np.float32),
            "observation.images.cam_high": process_img(img_high),
            "observation.images.cam_left_wrist": process_img(img_left),
            "observation.images.cam_right_wrist": process_img(img_right),
            "task": self.task,
        }
        
        self.dataset.add_frame(frame_data)
        self.frame_count += 1
        
        # 每 100 帧打印一次进度
        if self.frame_count % 100 == 0:
            print(f"    已录制 {self.frame_count} 帧 ({self.frame_count / self.fps:.1f}s)")
        
    def end_episode(self):
        """结束当前 episode"""
        if self.frame_count == 0:
            print(f">>> ⚠️  Episode {self.episode_count} 无数据，跳过")
            return False
            
        self.dataset.save_episode()
        print(f">>> ⏹️  Episode {self.episode_count} 录制完成")
        print(f"    - 帧数: {self.frame_count}")
        print(f"    - 时长: {self.frame_count / self.fps:.1f}s")
        self.episode_count += 1
        return True
        
    def finalize(self, push_to_hub: bool = False):
        """完成数据集录制"""
        print(f"\n>>> 正在完成数据集...")
        self.dataset.finalize()
        
        if push_to_hub:
            print(f">>> 正在上传到 HuggingFace Hub...")
            self.dataset.push_to_hub()
            print(f">>> ✅ 上传完成: https://huggingface.co/datasets/{self.repo_id}")
        
        print(f"\n{'='*60}")
        print(f"📦 LeRobot 数据集录制完成!")
        print(f"{'='*60}")
        print(f"  Repo ID:    {self.repo_id}")
        print(f"  Episodes:   {self.episode_count}")
        print(f"  FPS:        {self.fps}")
        print(f"  本地路径:   {self.dataset.root}")
        print(f"\n📊 可视化方式:")
        print(f"  本地: lerobot-dataset-viz --repo-id {self.repo_id} --episode-index 0")
        print(f"  在线: https://huggingface.co/spaces/lerobot/visualize_dataset")
        print(f"\n📝 数据字段说明:")
        print(f"  observation.state  - 机器人当前实际状态")
        print(f"  action.raw         - 模型原始输出 (未经后处理)")
        print(f"  action             - 实际发送给机器人的动作 (经过后处理)")
        print(f"{'='*60}")


# ================= 时序聚合器 =================
class TemporalEnsemble:
    def __init__(self, max_len=1):
        self.lock = threading.Lock()
        self.max_len = max_len
        self.action_sum = np.zeros((max_len, 14), dtype=np.float32)
        self.count = np.zeros((max_len, 14), dtype=np.float32)
        self.gripper_buffer = np.zeros((max_len, 2), dtype=np.float32)
        self.gripper_valid = np.zeros((max_len, 1), dtype=bool)
        self.last_action = np.zeros(14)
        self.last_action[:7] = HOME_POSE_LEFT
        self.last_action[7:] = HOME_POSE_RIGHT
        
        # 保存最近的 raw action (模型直接输出)
        self.last_raw_action = np.zeros(14)

    def update(self, new_actions):
        new_actions = np.array(new_actions, dtype=np.float32)
        L = min(len(new_actions), self.max_len)
        with self.lock:
            self.action_sum[:L] += new_actions[:L]
            self.count[:L] += 1.0
            grippers = new_actions[:L][:, [6, 13]]
            self.gripper_buffer[:L] = grippers
            self.gripper_valid[:L] = True
            # 保存第一个 action 作为 raw action
            if L > 0:
                self.last_raw_action = new_actions[0].copy()

    def get_action(self):
        """返回 (processed_action, raw_action)"""
        with self.lock:
            if self.count[0, 0] == 0:
                return self.last_action.copy(), self.last_raw_action.copy()
                
            avg_action = self.action_sum[0] / self.count[0]
            
            if self.gripper_valid[0]:
                raw_left_grip = self.gripper_buffer[0, 0]
                raw_right_grip = self.gripper_buffer[0, 1]
            else:
                raw_left_grip = avg_action[6]
                raw_right_grip = avg_action[13]
                
            final_action = avg_action.copy()
            raw_left_grip = np.clip(raw_left_grip, 0.0, 1.0)
            raw_right_grip = np.clip(raw_right_grip, 0.0, 1.0)
            final_action[6] = raw_left_grip * GRIPPER_SCALE
            final_action[13] = raw_right_grip * GRIPPER_SCALE
            
            # shift buffer
            self.action_sum[:-1] = self.action_sum[1:]
            self.action_sum[-1] = 0
            self.count[:-1] = self.count[1:]
            self.count[-1] = 0
            self.gripper_buffer[:-1] = self.gripper_buffer[1:]
            self.gripper_buffer[-1] = 0
            self.gripper_valid[:-1] = self.gripper_valid[1:]
            self.gripper_valid[-1] = False
            
            self.last_action = final_action
            
            return final_action, self.last_raw_action.copy()


# ================= ROS 操作器 =================
class RosOperator:
    def __init__(self):
        self.bridge = CvBridge()
        self.img_high_deque = deque(maxlen=10)
        self.img_left_deque = deque(maxlen=10)
        self.img_right_deque = deque(maxlen=10)
        self.puppet_left_deque = deque(maxlen=10)
        self.puppet_right_deque = deque(maxlen=10)
        self.init_subscribers()
        self.pub_left = rospy.Publisher(TOPIC_CONFIG['cmd_left'], JointState, queue_size=1)
        self.pub_right = rospy.Publisher(TOPIC_CONFIG['cmd_right'], JointState, queue_size=1)
        self.pub_enable = rospy.Publisher(TOPIC_CONFIG['enable'], Bool, queue_size=1)
        print(">>> RosOperator 初始化完成", flush=True)

    def init_subscribers(self):
        rospy.Subscriber(TOPIC_CONFIG['img_high'], Image, 
                        lambda m: self.img_high_deque.append(m), tcp_nodelay=True)
        rospy.Subscriber(TOPIC_CONFIG['img_left'], Image, 
                        lambda m: self.img_left_deque.append(m), tcp_nodelay=True)
        rospy.Subscriber(TOPIC_CONFIG['img_right'], Image, 
                        lambda m: self.img_right_deque.append(m), tcp_nodelay=True)
        rospy.Subscriber(TOPIC_CONFIG['puppet_left'], JointState, 
                        lambda m: self.puppet_left_deque.append(m), tcp_nodelay=True)
        rospy.Subscriber(TOPIC_CONFIG['puppet_right'], JointState, 
                        lambda m: self.puppet_right_deque.append(m), tcp_nodelay=True)

    def get_frame(self):
        """获取同步的观测数据"""
        if len(self.img_high_deque) == 0 or len(self.puppet_left_deque) == 0:
            return None
        try:
            timestamps = [
                self.img_high_deque[-1].header.stamp.to_sec(),
                self.img_left_deque[-1].header.stamp.to_sec(),
                self.img_right_deque[-1].header.stamp.to_sec(),
                self.puppet_left_deque[-1].header.stamp.to_sec(),
                self.puppet_right_deque[-1].header.stamp.to_sec()
            ]
            frame_time = min(timestamps)
            
            def sync_deque(d, t):
                while len(d) > 0 and d[0].header.stamp.to_sec() < t - 0.05:
                    d.popleft()
                return len(d) > 0
                
            if not (sync_deque(self.img_high_deque, frame_time) and 
                    sync_deque(self.img_left_deque, frame_time) and 
                    sync_deque(self.img_right_deque, frame_time) and 
                    sync_deque(self.puppet_left_deque, frame_time) and 
                    sync_deque(self.puppet_right_deque, frame_time)):
                return None
                
            img_high = self.bridge.imgmsg_to_cv2(self.img_high_deque[0], "rgb8")
            img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque[0], "rgb8")
            img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque[0], "rgb8")
            qpos_left = np.array(self.puppet_left_deque[0].position)
            qpos_right = np.array(self.puppet_right_deque[0].position)
            return (img_high, img_left, img_right, qpos_left, qpos_right)
        except Exception as e:
            return None
    
    def get_current_state(self):
        """获取当前机器人状态"""
        if len(self.puppet_left_deque) == 0 or len(self.puppet_right_deque) == 0:
            return None
        qpos_left = np.array(self.puppet_left_deque[-1].position)
        qpos_right = np.array(self.puppet_right_deque[-1].position)
        return np.concatenate([qpos_left, qpos_right])

    def puppet_arm_publish(self, left_action, right_action):
        msg_l = JointState()
        msg_l.header.stamp = rospy.Time.now()
        msg_l.position = left_action
        self.pub_left.publish(msg_l)
        msg_r = JointState()
        msg_r.header.stamp = rospy.Time.now()
        msg_r.position = right_action
        self.pub_right.publish(msg_r)
        
    def enable_robot(self):
        print(">>> 正在发送使能信号...", flush=True)
        msg = Bool()
        msg.data = True
        for _ in range(5):
            self.pub_enable.publish(msg)
            time.sleep(0.1)


def move_to_home_slowly(ros_operator):
    print(">>> ⚠️  准备回到初始姿态...", flush=True)
    time.sleep(1.0)
    if len(ros_operator.puppet_left_deque) == 0:
        print("    未收到机械臂状态，跳过归位", flush=True)
        return
    start_left = np.array(ros_operator.puppet_left_deque[-1].position)
    start_right = np.array(ros_operator.puppet_right_deque[-1].position)
    target_left = np.array(HOME_POSE_LEFT)
    target_right = np.array(HOME_POSE_RIGHT)
    duration = 5.0 
    steps = int(duration * 50)
    rate = rospy.Rate(50)
    traj_left = np.linspace(start_left, target_left, steps)
    traj_right = np.linspace(start_right, target_right, steps)
    for i in range(steps):
        if rospy.is_shutdown(): break
        ros_operator.puppet_arm_publish(traj_left[i], traj_right[i])
        rate.sleep()
    ros_operator.puppet_arm_publish(HOME_POSE_LEFT, HOME_POSE_RIGHT)
    print(">>> ✅ 初始姿态移动完成！", flush=True)


# ================= 数据缓冲区 =================
class FrameBuffer:
    """线程安全的帧数据缓冲区"""
    def __init__(self):
        self.lock = threading.Lock()
        self.data = None
        
    def put(self, img_high, img_left, img_right, state, action_raw, action_processed):
        with self.lock:
            self.data = {
                'img_high': img_high.copy(),
                'img_left': img_left.copy(),
                'img_right': img_right.copy(),
                'state': state.copy(),
                'action_raw': action_raw.copy(),
                'action_processed': action_processed.copy(),
            }
    
    def get(self):
        with self.lock:
            if self.data is None:
                return None
            data = self.data
            self.data = None
            return data


frame_buffer = FrameBuffer()


# ================= 推理线程 =================
def inference_process(ros_operator, client, ensemble):
    print(">>> 推理线程已启动", flush=True)
    
    def process_image(img_data):
        resized = image_tools.resize_with_pad(img_data, 224, 224)
        uint8_img = image_tools.convert_to_uint8(resized)
        return uint8_img.transpose(2, 0, 1)

    while not rospy.is_shutdown() and not recording_state.should_quit:
        result = ros_operator.get_frame()
        if result is None:
            time.sleep(0.01)
            continue
            
        (img_high, img_left, img_right, qpos_left, qpos_right) = result
        current_state = np.concatenate([qpos_left, qpos_right])
        
        # 归一化输入状态
        state_vec_norm = current_state.copy()
        state_vec_norm[6] = np.clip(current_state[6] / GRIPPER_SCALE, 0.0, 1.0)
        state_vec_norm[13] = np.clip(current_state[13] / GRIPPER_SCALE, 0.0, 1.0)

        obs = {
            "images": {
                "cam_high": process_image(img_high),
                "cam_left_wrist": process_image(img_left),
                "cam_right_wrist": process_image(img_right),
            },
            "state": state_vec_norm.tolist(),
            "prompt": myprompt,
        }
        
        try:
            t0 = time.time()
            resp = client.infer(obs)
            t_infer = time.time() - t0
            latency_steps = int(t_infer / (1.0 / CTRL_FREQ))

            if resp and "actions" in resp:
                actions = np.array(resp["actions"])
                
                if latency_steps < len(actions):
                    valid_actions = actions[latency_steps:]
                    ensemble.update(valid_actions)
                    
                    # 获取处理后的动作和原始动作
                    action_processed, action_raw = ensemble.get_action()
                    
                    # 放入缓冲区供录制使用
                    if recording_state.is_recording():
                        frame_buffer.put(
                            img_high, img_left, img_right,
                            current_state, action_raw, action_processed
                        )
                    
                    if np.random.rand() < 0.02:
                        print(f"    [Infer] {t_infer*1000:.1f}ms, lag={latency_steps}", flush=True)
                else:
                    print(f"    ⚠️ 推理过慢 ({t_infer:.3f}s)", flush=True)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"推理错误: {e}")
            time.sleep(0.5)


# ================= 主函数 =================
def main():
    parser = argparse.ArgumentParser(description='ROS OpenPI 推理 + LeRobot 交互式录制')
    parser.add_argument('--repo_id', type=str, default='user/my_bimanual_dataset',
                        help='HuggingFace 数据集 repo ID')
    parser.add_argument('--push_to_hub', action='store_true', help='完成后上传到 HF Hub')
    parser.add_argument('--task', type=str, default=None, help='任务描述')
    parser.add_argument('--no_record', action='store_true', help='不录制，仅推理')
    args = parser.parse_args()
    
    print("="*60)
    print("  ROS OpenPI 推理 + LeRobot 交互式录制")
    print("="*60)
    
    rospy.init_node('inference_lerobot_node', anonymous=True)
    ros_operator = RosOperator()
    ensemble = TemporalEnsemble(max_len=30)
    
    # 初始化录制器
    recorder = None
    if not args.no_record:
        recorder = LeRobotRecorder(
            repo_id=args.repo_id,
            fps=CTRL_FREQ,
            task=args.task or myprompt,
            image_size=(224, 224),
        )
    
    # 等待 ROS 数据
    print(">>> 等待 ROS 数据...")
    time.sleep(2.0)
    
    # 归零
    move_to_home_slowly(ros_operator)
    
    # 连接推理服务器
    try:
        print(">>> 正在启动服务器...", flush=True)
        client = websocket_client_policy.WebsocketClientPolicy(
            host=SERVER_HOST, port=SERVER_PORT)
        print(">>> ✅ 推理服务器连接成功", flush=True)
    except Exception as e:
        print(f">>> ❌ 服务器连接失败: {e}", flush=True)
        return

    # 启动推理线程
    inf_thread = threading.Thread(
        target=inference_process, 
        args=(ros_operator, client, ensemble))
    inf_thread.daemon = True
    inf_thread.start()
    
    rate = rospy.Rate(CTRL_FREQ)
    print(f"\n>>> 推理循环已启动 ({CTRL_FREQ}Hz)")
    
    if args.no_record:
        print(">>> 仅推理模式，不录制数据")
        print(">>> 按 Ctrl+C 退出")
        while not rospy.is_shutdown() and not recording_state.should_quit:
            action_processed, _ = ensemble.get_action()
            if action_processed is not None:
                ros_operator.puppet_arm_publish(action_processed[:7], action_processed[7:])
            rate.sleep()
        return
    
    # 交互式录制循环
    print("\n" + "="*60)
    print("  交互式录制模式")
    print("="*60)
    print("  操作说明:")
    print("    - 按 Enter 开始录制 Episode")
    print("    - 录制中按 Ctrl+C 结束当前 Episode")
    print("    - Episode 结束后按 Enter 继续下一个")
    print("    - 输入 'q' 或 'quit' 退出并保存")
    print("="*60)
    
    try:
        while not rospy.is_shutdown() and not recording_state.should_quit:
            # 等待用户开始
            print(f"\n>>> 准备录制 Episode {recorder.episode_count}")
            user_input = input(">>> 按 Enter 开始录制 (输入 'q' 退出): ").strip().lower()
            
            if user_input in ['q', 'quit', 'exit']:
                print(">>> 用户选择退出")
                break
            
            # 开始录制
            recording_state.start_recording()
            recorder.start_episode()
            
            # 录制循环
            while not rospy.is_shutdown() and recording_state.is_recording():
                # 获取并发送动作
                action_processed, action_raw = ensemble.get_action()
                if action_processed is not None:
                    ros_operator.puppet_arm_publish(action_processed[:7], action_processed[7:])
                
                # 从缓冲区获取数据并录制
                frame_data = frame_buffer.get()
                if frame_data is not None:
                    recorder.add_frame(
                        img_high=frame_data['img_high'],
                        img_left=frame_data['img_left'],
                        img_right=frame_data['img_right'],
                        state=frame_data['state'],
                        action_raw=frame_data['action_raw'],
                        action_processed=frame_data['action_processed'],
                    )
                
                rate.sleep()
            
            # 结束当前 episode
            recording_state.end_recording()
            recorder.end_episode()
            
            # 归零
            print(">>> 正在归零...")
            move_to_home_slowly(ros_operator)
            
    except Exception as e:
        print(f">>> 异常: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 完成并保存数据集
        if recorder and recorder.episode_count > 0:
            recorder.finalize(push_to_hub=args.push_to_hub)
        elif recorder:
            print(">>> 未录制任何 episode，不保存数据集")


if __name__ == "__main__":
    main()