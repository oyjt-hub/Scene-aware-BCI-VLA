import rospy
import threading
import time
import numpy as np
from collections import deque
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Header, Bool
from cv_bridge import CvBridge
from openpi_client import image_tools, websocket_client_policy

# ================= 配置区域 =================
SERVER_HOST = "172.19.1.40"
SERVER_PORT = 9000
CTRL_FREQ = 20 # 20Hz (0.05s per step)
myprompt="picking up a paper cup with the left hand and a bottle of nongfu spring mineral water with the right hand, pouring the water into the paper cup, and then placing them back on the table after completing."
# [关键参数] 夹爪缩放因子
# 物理最大开口约为 0.08米。模型输出 0~1。
# 最终输出 = 模型值 * GRIPPER_SCALE
GRIPPER_SCALE = 0.1 

# [初始姿态]
HOME_POSE_LEFT  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08]
HOME_POSE_RIGHT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08]

TOPIC_CONFIG = {
    'img_high':  '/camera_f/color/image_raw',
    'img_left':  '/camera_l/color/image_raw',
    'img_right': '/camera_r/color/image_raw',
    'puppet_left': '/puppet/joint_left',
    'puppet_right': '/puppet/joint_right',
    'cmd_left': '/master/joint_left',
    'cmd_right': '/master/joint_right',
    'enable': '/enable_flag'
}

# ================= 时序聚合器 (Temporal Ensemble) =================
class TemporalEnsemble:
    def __init__(self, max_len=1):
        self.lock = threading.Lock()
        self.max_len = max_len
        
        # 动作累加 buffer [T, 14]
        self.action_sum = np.zeros((max_len, 14), dtype=np.float32)
        # 计数 buffer [T, 14]
        self.count = np.zeros((max_len, 14), dtype=np.float32)
        
        # 夹爪专用 buffer (不进行平均，直接覆盖)
        self.gripper_buffer = np.zeros((max_len, 2), dtype=np.float32) # [Left, Right]
        self.gripper_valid = np.zeros((max_len, 1), dtype=bool)         # 标记是否有值

        # 初始化默认动作 (用于 buffer 为空时)
        self.last_action = np.zeros(14)
        self.last_action[:7] = HOME_POSE_LEFT
        self.last_action[7:] = HOME_POSE_RIGHT

    def update(self, new_actions):
        """
        接收新的推理结果块 [H, 14]
        """
        new_actions = np.array(new_actions, dtype=np.float32)
        L = min(len(new_actions), self.max_len)
        
        with self.lock:
            # === 1. 机械臂部分：执行累加平均 ===
            # 将新预测加到 buffer 中对应的位置
            self.action_sum[:L] += new_actions[:L]
            self.count[:L] += 1.0
            
            # === 2. 夹爪部分：执行最新覆盖 (No Smoothing) ===
            # 提取新动作中的夹爪值 (index 6 和 13)
            # 这种策略保证夹爪总是执行"最新"的意图，而不会被旧的预测拖慢
            grippers = new_actions[:L][:, [6, 13]] # Shape [L, 2]
            self.gripper_buffer[:L] = grippers
            self.gripper_valid[:L] = True

    def get_action(self):
        """
        取出当前时刻的动作，并让 Buffer 前进一步
        """
        with self.lock:
            # 如果当前没有任何累积动作（还没推理出来），维持上一次动作
            if self.count[0, 0] == 0:
                return self.last_action.copy()

            # 1. 计算平均动作 (仅用于机械臂)
            avg_action = self.action_sum[0] / self.count[0]
            
            # 2. 获取夹爪动作 (覆盖策略)
            # 如果有值则取 gripper_buffer，否则取 avg 的结果兜底
            if self.gripper_valid[0]:
                raw_left_grip = self.gripper_buffer[0, 0]
                raw_right_grip = self.gripper_buffer[0, 1]
            else:
                raw_left_grip = avg_action[6]
                raw_right_grip = avg_action[13]

            # 3. 构造最终动作
            final_action = avg_action.copy()

            # === 夹爪处理：直接线性缩放 (无二值化) ===
            # 安全截断到 0~1 之间，防止模型输出越界
            raw_left_grip = np.clip(raw_left_grip, 0.0, 1.0)
            raw_right_grip = np.clip(raw_right_grip, 0.0, 1.0)
            
            # 映射到物理范围 (0 ~ GRIPPER_SCALE)
            final_action[6] = raw_left_grip * GRIPPER_SCALE
            final_action[13] = raw_right_grip * GRIPPER_SCALE

            # 4. 更新 Buffer (向前移位)
            # 像传送带一样，把未来的动作移到现在
            self.action_sum[:-1] = self.action_sum[1:]
            self.action_sum[-1] = 0
            
            self.count[:-1] = self.count[1:]
            self.count[-1] = 0
            
            self.gripper_buffer[:-1] = self.gripper_buffer[1:]
            self.gripper_buffer[-1] = 0
            
            self.gripper_valid[:-1] = self.gripper_valid[1:]
            self.gripper_valid[-1] = False

            # 保存本次动作，防止下次 buffer 为空
            self.last_action = final_action
            
            return final_action

# ================= 核心类：RosOperator =================
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
        rospy.Subscriber(TOPIC_CONFIG['img_high'], Image, lambda m: self.img_high_deque.append(m), tcp_nodelay=True)
        rospy.Subscriber(TOPIC_CONFIG['img_left'], Image, lambda m: self.img_left_deque.append(m), tcp_nodelay=True)
        rospy.Subscriber(TOPIC_CONFIG['img_right'], Image, lambda m: self.img_right_deque.append(m), tcp_nodelay=True)
        rospy.Subscriber(TOPIC_CONFIG['puppet_left'], JointState, lambda m: self.puppet_left_deque.append(m), tcp_nodelay=True)
        rospy.Subscriber(TOPIC_CONFIG['puppet_right'], JointState, lambda m: self.puppet_right_deque.append(m), tcp_nodelay=True)

    def get_frame(self):
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
    print(">>> ⚠️  准备回到初始姿态 (Slow Mode) ...", flush=True)
    time.sleep(1.0)
    if len(ros_operator.puppet_left_deque) == 0:
        print("    未收到机械臂状态，跳过归位，直接开始...", flush=True)
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

# ================= 推理线程 =================
# ================= 推理线程 (已修复 latency_steps 未定义错误) =================
def inference_process(ros_operator, client, ensemble):
    print(">>> 推理线程已启动", flush=True)
    
    def process_image(img_data):
        resized = image_tools.resize_with_pad(img_data, 224, 224)
        uint8_img = image_tools.convert_to_uint8(resized)
        return uint8_img.transpose(2, 0, 1)

    while not rospy.is_shutdown():
        # 获取观测
        result = ros_operator.get_frame()
        if result is None:
            time.sleep(0.01)
            continue
            
        (img_high, img_left, img_right, qpos_left, qpos_right) = result
        state_vec = np.concatenate([qpos_left, qpos_right])
        
        # 归一化输入状态中的夹爪值
        state_vec[6]  = np.clip(state_vec[6] / GRIPPER_SCALE, 0.0, 1.0)
        state_vec[13] = np.clip(state_vec[13] / GRIPPER_SCALE, 0.0, 1.0)

        obs = {
            "images": {
                "cam_high": process_image(img_high),
                "cam_left_wrist": process_image(img_left),
                "cam_right_wrist": process_image(img_right),
            },
            "state": state_vec.tolist(),
            "prompt": myprompt,
        }
        
        try:
            # 1. 记录开始时间
            t0 = time.time()
            
            # 2. 执行推理
            resp = client.infer(obs)
            
            # 3. 计算推理耗时
            t_infer = time.time() - t0
            
            # === [关键修复] 计算延迟了多少个时间步 ===
            # 公式：延迟时间 / 每步时间(0.05s)
            # 如果不加这行，下面用到 latency_steps 时就会报错
            latency_steps = int(t_infer / (1.0 / CTRL_FREQ))
            # latency_steps = 0
            # ==========================================

            if resp and "actions" in resp:
                actions = np.array(resp["actions"])
                
                # 4. 延迟补偿策略：丢弃掉已经过期的动作
                # 例如：如果推理花了0.2秒（4步），那么动作的前4步已经是过去式了，不能执行
                if latency_steps < len(actions):
                    # 取出剩余的有效动作
                    valid_actions = actions[latency_steps:]
                    # 放入时间聚合器
                    ensemble.update(valid_actions)
                    
                    if np.random.rand() < 0.05:
                        print(f"Infer time: {t_infer*1000:.1f}ms, Lag steps: {latency_steps}", flush=True)
                else:
                    # 如果推理太慢，导致整个预测窗口都过期了，就跳过
                    print(f"⚠️ 推理过慢 ({t_infer:.3f}s)，动作已过期", flush=True)

        except Exception as e:
            # 打印详细错误栈以便调试
            import traceback
            traceback.print_exc()
            print(f"推理错误: {e}")
            time.sleep(0.5)
# ================= 主循环 =================
def main():
    rospy.init_node('inference_node', anonymous=True)
    ros_operator = RosOperator()
    
    # 替换回 Ensemble
    # max_len 设置大一点以防网络波动
    ensemble = TemporalEnsemble(max_len=30)
    
    # ros_operator.enable_robot()
    time.sleep(1.0)
    move_to_home_slowly(ros_operator)
    
    try:
        print(">>> 正在启动服务器...", flush=True)
        client = websocket_client_policy.WebsocketClientPolicy(host=SERVER_HOST, port=SERVER_PORT)
        print(">>> 服务器连接成功", flush=True)
    except Exception as e:
        print(f"服务器连接失败: {e}", flush=True)
        return

    inf_thread = threading.Thread(target=inference_process, 
                                  args=(ros_operator, client, ensemble))
    inf_thread.daemon = True
    inf_thread.start()
    
    rate = rospy.Rate(CTRL_FREQ)
    print(f">>> 开始 AI 控制循环 ({CTRL_FREQ}Hz)", flush=True)
    
    while not rospy.is_shutdown():
        # 每一步从 Ensemble 获取平滑后的动作
        action = ensemble.get_action()
        
        if action is not None:
            left_action = action[:7]
            right_action = action[7:]
            ros_operator.puppet_arm_publish(left_action, right_action)
            
        rate.sleep()

if __name__ == "__main__":
    main()