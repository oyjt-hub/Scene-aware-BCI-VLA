import time
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Any
import threading
import copy  # 新增: 用于深拷贝动作块

# --- OpenPi 与 SDK 导入 ---
try:
    from openpi_client import image_tools, websocket_client_policy
except ImportError:
    print("❌ 错误: 未找到 openpi_client，请检查安装。")
    exit(1)

try:
    from piper_sdk import C_PiperInterface_V2
except ImportError:
    C_PiperInterface_V2 = None
    print("⚠️ 警告: 未找到 piper_sdk，将在模拟模式下运行 (无硬件连接)。")

# ================= 配置区域 =================

# 策略服务器地址
SERVER_HOST = "172.19.1.40"  # 请修改为实际 IP
SERVER_PORT = 9000
# 任务指令  
TASK_INSTRUCTION = "pick up the capsule and put it into the blue bowl,then pick up the cherry and put it into the white bowl."

# 控制频率
CONTROL_RATE_HZ = 15
DT = 1.0 / CONTROL_RATE_HZ

# --- 新增：插值配置 ---
INTERPOLATION_STEPS = 3  # 在两个动作点之间插值多少步（例如4步，将15Hz提升至60Hz控制）
SUB_DT = DT / INTERPOLATION_STEPS  # 插值小步的执行间隔时间

# RHC 执行步数核心参数 (Action Chunking)
EXECUTION_STEPS = 80

# 关节限位 (弧度)
JOINT_LIMITS_RAD = np.array([[-2.61799, 2.61799],[0.0, 3.14159], [-2.96706, 0.0],[-1.74533, 1.74533],[-1.22173, 1.22173],[-2.0944, 2.0944],
])

# --- 新增: 平滑配置与多线程通信变量 ---
ACTION_EMA_ALPHA = 0.35  # 平滑系数 (0.0~1.0): 越小越平滑但有延迟，越大跟随越快但易跳变。推荐 0.2~0.5
latest_chunk_lock = threading.Lock()
latest_action_chunk = None
new_chunk_available = False


# ================= 辅助类: 多线程摄像头 =================
class CameraThread:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.frame = None
        self.ret = False
        self.running = True
        self._init_cap()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _init_cap(self):
        """初始化或重新初始化摄像头"""
        print(f"正在尝试连接摄像头 {self.name} : {self.path}")
        self.cap = cv2.VideoCapture(self.path)
        # --- 关键设置：强制使用 MJPG 格式 ---
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _update(self):
        consecutive_failures = 0
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.frame = frame
                    self.ret = True
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
            else:
                consecutive_failures += 1

            # 如果连续丢失超过 30 帧（约2秒），尝试重连
            if consecutive_failures > 30:
                print(f"检测到摄像头 {self.name} 掉线，正在尝试自动重连...")
                self.cap.release()
                time.sleep(1.0)
                self._init_cap()
                consecutive_failures = 0
            
            time.sleep(0.01)

    def get_frame(self):
        return self.frame

    def stop(self):
        self.running = False
        if hasattr(self, 'cap'):
            self.cap.release()


# ================= 新增: 独立异步推理线程 =================
class InferenceThread(threading.Thread):
    def __init__(self, client, cam_main, cam_l, cam_r, piper_left, piper_right):
        super().__init__(daemon=True)
        self.client = client
        self.cam_main = cam_main
        self.cam_l = cam_l
        self.cam_r = cam_r
        self.piper_left = piper_left
        self.piper_right = piper_right
        self.running = True

    def run(self):
        global latest_action_chunk, new_chunk_available
        while self.running:
            start_time = time.time()
            
            # A. 获取图像
            img = self.cam_main.get_frame()
            w_l = self.cam_l.get_frame()
            w_r = self.cam_r.get_frame()
                    
            if img is None or w_l is None or w_r is None:
                missing =[]
                if img is None: missing.append("Main")
                if w_l is None: missing.append("Left")
                if w_r is None: missing.append("Right")
                print(f"⚠️ [推理线程] 等待摄像头数据: {missing}")
                time.sleep(0.1)
                continue
        
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            w_l = cv2.cvtColor(w_l, cv2.COLOR_BGR2RGB)
            w_r = cv2.cvtColor(w_r, cv2.COLOR_BGR2RGB)

            # B. 读取状态
            curr_l_rad = get_current_joints_rad(self.piper_left)
            curr_r_rad = get_current_joints_rad(self.piper_right)
            
            PHYSICAL_MAX_MM = 70.0
            val_l_mm = get_gripper_value(self.piper_left)
            val_r_mm = get_gripper_value(self.piper_right)
            norm_grip_l = np.clip(val_l_mm / PHYSICAL_MAX_MM, 0.0, 1.0)
            norm_grip_r = np.clip(val_r_mm / PHYSICAL_MAX_MM, 0.0, 1.0)

            state_vec = np.zeros(14, dtype=np.float32)
            state_vec[0:6] = curr_l_rad
            state_vec[6] = norm_grip_l
            state_vec[7:13] = curr_r_rad
            state_vec[13] = norm_grip_r

            obs = {
                "images": {
                    "cam_high": process_image(img),
                    "cam_left_wrist": process_image(w_l),
                    "cam_right_wrist": process_image(w_r),
                },
                "state": state_vec.tolist(),
                "prompt": TASK_INSTRUCTION,
            }

            # C. 发送请求
            try:
                resp = self.client.infer(obs)
                if resp and "actions" in resp:
                    all_actions = np.array(resp["actions"])
                    if all_actions.ndim == 3:  # 降维处理
                        all_actions = all_actions[0]
                    
                    # 更新全局动作块
                    with latest_chunk_lock:
                        latest_action_chunk = all_actions
                        new_chunk_available = True
                    
                    horizon = all_actions.shape[0]
                    print(f"✅ [推理线程] 获取推理结果，已更新最新动作块 (共 {horizon} 步)")
                else:
                    print("❌ 服务器未返回 actions 字段")
            except Exception as e:
                print(f"⚠️ 推理请求异常: {e}")
            
            # 控制推理最大请求频率 (假设模型极限推理需约0.2秒以上，防止无意义的高频并发)
            elapsed = time.time() - start_time
            if elapsed < 0.2:  
                time.sleep(0.2 - elapsed)

    def stop(self):
        self.running = False


# ================= 辅助函数 =================

def clamp_rad(rad_targets: np.ndarray) -> np.ndarray:
    """限制关节角度在物理极限内"""
    return np.clip(rad_targets, JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])

def deg_to_sdk_unit(rad: np.ndarray) -> np.ndarray:
    """弧度 -> SDK单位 (0.001度)"""
    deg = np.rad2deg(rad)
    return (deg * 1000.0).astype(np.int32)

def get_current_joints_rad(piper: Any) -> np.ndarray:
    """获取当前关节角 (弧度)，兼容多种 SDK 返回格式"""
    if piper is None: return np.zeros(6)
    try:
        js_ret = piper.GetArmJointMsgs()
        if isinstance(js_ret, tuple) and len(js_ret) >= 3:
            js = js_ret[2]
        else:
            js = js_ret
            
        if hasattr(js, "joint_state") and js.joint_state:
            st = js.joint_state
            for names in [["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],["angle_1", "angle_2", "angle_3", "angle_4", "angle_5", "angle_6"]
            ]:
                try:
                    vals =[getattr(st, n) for n in names]
                    return np.deg2rad(np.array(vals, dtype=np.float64) / 1000.0)
                except AttributeError: continue
        return np.zeros(6)
    except Exception as e:
        print(f"获取关节数据异常: {e}")
        return np.zeros(6)

def get_gripper_value(piper):
    """获取夹爪值"""
    if piper is None: return 0.0
    try:
        gripper_msgs = piper.GetArmGripperMsgs()
        if hasattr(gripper_msgs, 'gripper_state') and gripper_msgs.gripper_state:
            raw_value = gripper_msgs.gripper_state.grippers_angle
            width_mm = raw_value / 1000.0
            return width_mm
    except Exception:
        pass
    return 0.0

def map_actions_to_joint_targets_dual(action_step, current_left_rad, current_right_rad):
    """将模型输出动作映射为机械臂控制指令"""
    action = np.array(action_step, dtype=np.float64)
    left_target_raw = action[0:6]
    left_grip_raw = float(action[6])
    right_target_raw = action[7:13]
    right_grip_raw = float(action[13])

    left_target = clamp_rad(left_target_raw)
    right_target = clamp_rad(right_target_raw)
    
    return left_target, right_target, left_grip_raw, right_grip_raw

def setup_piper(can_name: str):
    """初始化机械臂连接"""
    if C_PiperInterface_V2 is None: return None
    print(f"正在连接 {can_name} ...")
    try:
        piper = C_PiperInterface_V2(can_name=can_name, judge_flag=True, can_auto_init=True)
        piper.CreateCanBus(can_name=can_name, bustype="socketcan", expected_bitrate=1000000, judge_flag=False)
        piper.ConnectPort(can_init=False, piper_init=True, start_thread=True)
        piper.ModeCtrl(ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=30, is_mit_mode=0x00)
        return piper
    except Exception as e:
        print(f"❌ {can_name} 连接失败: {e}")
        return None

def process_image(img_data):
    """图像预处理 (符合 OpenPI 规范)"""
    resized = image_tools.resize_with_pad(img_data, 224, 224)
    uint8_img = image_tools.convert_to_uint8(resized)
    return uint8_img.transpose(2, 0, 1)

def save_log(log_data):
    filename = f"inference_log_{int(time.time())}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "lj1", "lj2", "lj3", "lj4", "lj5", "lj6", "lgrip", 
                         "rj1", "rj2", "rj3", "rj4", "rj5", "rj6", "rgrip"])
        writer.writerows(log_data)
    print(f"日志已保存至 {filename}")

# ================= 主程序 =================

def main():
    global latest_action_chunk, new_chunk_available

    # 1. 初始化硬件
    piper_left = setup_piper("can_left")
    piper_right = setup_piper("can_right")
    
    # 初始化多线程摄像头
    print("正在连接摄像头 (多线程)...")
    cam_main = CameraThread("/dev/v4l/by-path/pci-0000:00:14.0-usb-0:3.2:1.0-video-index0", "main")
    cam_wrist_left = CameraThread("/dev/v4l/by-path/pci-0000:00:14.0-usb-0:5.2:1.0-video-index0", "left")
    cam_wrist_right = CameraThread("/dev/v4l/by-path/pci-0000:00:14.0-usb-0:4.2:1.0-video-index0", "right")

    if piper_left and piper_right:
        print("等待控制器就绪...")
        time.sleep(2.0)
        print(">>> 正在使能机械臂 <<<")
        for _ in range(3):
            if piper_left:
                piper_left.EnableArm(motor_num=7, enable_flag=0x02)
                piper_left.GripperCtrl(0, 300, 0x01)
            if piper_right:
                piper_right.EnableArm(motor_num=7, enable_flag=0x02)
                piper_right.GripperCtrl(0, 300, 0x01)
            time.sleep(0.2)
        
        print(">>> 移动到初始位姿 <<<")
        home_pose = np.array([-0.0013, 0.0043, 0.0345, -0.0535, 0.0, -0.0020]) 
        start_l = get_current_joints_rad(piper_left)
        start_r = get_current_joints_rad(piper_right)
        
        for i in range(60):
            alpha = (i + 1) / 60
            curr_l = start_l * (1 - alpha) + home_pose * alpha
            curr_r = start_r * (1 - alpha) + home_pose * alpha
            if piper_left: piper_left.JointCtrl(*deg_to_sdk_unit(curr_l))
            if piper_right: piper_right.JointCtrl(*deg_to_sdk_unit(curr_r))
            piper_left.GripperCtrl(70000, 3000, 0x01)
            piper_right.GripperCtrl(70000, 3000, 0x01)
            time.sleep(0.05)
        print("✅ 归位完成")
        time.sleep(1.0)
    else:
        print("❌ 机械臂连接失败")
        return

    # 4. 连接服务器
    print(f"连接策略服务器: {SERVER_HOST}:{SERVER_PORT}")
    try:
        client = websocket_client_policy.WebsocketClientPolicy(host=SERVER_HOST, port=SERVER_PORT)
    except Exception as e:
        print(f"❌ 连接服务器失败: {e}")
        return

    # >>> 启动异步推理线程 <<<
    print("启动异步推理线程...")
    inference_worker = InferenceThread(client, cam_main, cam_wrist_left, cam_wrist_right, piper_left, piper_right)
    inference_worker.start()

    inference_log =[]
    
    print("🔄 等待模型首次输出...")
    while True:
        with latest_chunk_lock:
            if latest_action_chunk is not None:
                break
        time.sleep(0.1)
    
    print("▶️ 开始连续执行动作...")

    # --- EMA 平滑预备：获取当前物理位置作为平滑的起点 ---
    smoothed_target_l = get_current_joints_rad(piper_left)
    smoothed_target_r = get_current_joints_rad(piper_right)
    smoothed_grip_l = get_gripper_value(piper_left) / 70.0
    smoothed_grip_r = get_gripper_value(piper_right) / 70.0

    current_chunk = None
    step_idx = 0
    steps_to_run = 0

    try:
        while True:
            # 保证主控制循环频率始终稳定
            loop_start = time.perf_counter()

            # --- 1. 检查推理线程是否送来了新的动作块 ---
            with latest_chunk_lock:
                if new_chunk_available:
                    current_chunk = copy.deepcopy(latest_action_chunk)
                    new_chunk_available = False
                    step_idx = 0  # 重置执行步数
                    steps_to_run = min(EXECUTION_STEPS, current_chunk.shape[0])
                    print(f"\n⚡ 主控制循环接管新动作块，准备执行 {steps_to_run} 步...")

            # --- 2. 连续执行动作 ---
            if current_chunk is not None:
                # 即使当前 chunk 跑完了，在拿到新 chunk 前也维持在最后一步（safe_idx保障数组不越界）
                safe_idx = min(step_idx, current_chunk.shape[0] - 1)
                action_step = current_chunk[safe_idx]
                
                # 解析原始目标点
                target_l, target_r, grip_l, grip_r = map_actions_to_joint_targets_dual(
                    action_step, smoothed_target_l, smoothed_target_r
                )

                # >>> 核心改变：EMA 低通滤波消除新老 Chunk 切换带来的瞬间跳变 <<<
                smoothed_target_l = ACTION_EMA_ALPHA * target_l + (1 - ACTION_EMA_ALPHA) * smoothed_target_l
                smoothed_target_r = ACTION_EMA_ALPHA * target_r + (1 - ACTION_EMA_ALPHA) * smoothed_target_r
                smoothed_grip_l = ACTION_EMA_ALPHA * grip_l + (1 - ACTION_EMA_ALPHA) * smoothed_grip_l
                smoothed_grip_r = ACTION_EMA_ALPHA * grip_r + (1 - ACTION_EMA_ALPHA) * smoothed_grip_r

                # --- 硬件插值逻辑 (使用前一次发送给物理硬件的位置做起点插值) ---
                prev_l_interp = get_current_joints_rad(piper_left)
                prev_r_interp = get_current_joints_rad(piper_right)
                prev_gl_interp = get_gripper_value(piper_left) / 70.0
                prev_gr_interp = get_gripper_value(piper_right) / 70.0

                for k in range(INTERPOLATION_STEPS):
                    sub_start = time.perf_counter()
                    
                    alpha = (k + 1) / INTERPOLATION_STEPS
                    
                    # 硬件级小步线性插值
                    interp_l = prev_l_interp * (1 - alpha) + smoothed_target_l * alpha
                    interp_r = prev_r_interp * (1 - alpha) + smoothed_target_r * alpha
                    interp_gl = prev_gl_interp * (1 - alpha) + smoothed_grip_l * alpha
                    interp_gr = prev_gr_interp * (1 - alpha) + smoothed_grip_r * alpha

                    if piper_left and piper_right:
                        piper_left.JointCtrl(*deg_to_sdk_unit(interp_l))
                        piper_right.JointCtrl(*deg_to_sdk_unit(interp_r))
                        
                        g_val_l = int(np.clip(interp_gl * 100000, 0, 70000))
                        g_val_r = int(np.clip(interp_gr * 100000, 0, 70000))
                        try:
                            piper_left.GripperCtrl(g_val_l, 3000, 0x01)
                            piper_right.GripperCtrl(g_val_r, 3000, 0x01)
                        except: pass

                    # 插值子步频率控制
                    elapsed_sub = time.perf_counter() - sub_start
                    if elapsed_sub < SUB_DT:
                        time.sleep(SUB_DT - elapsed_sub)
                
                inference_log.append([time.time(), *smoothed_target_l, smoothed_grip_l, *smoothed_target_r, smoothed_grip_r])
                
                # 保留原有打日志风格，为避免跑完后持续刷屏，加入判断
                if step_idx < steps_to_run:
                    print(f"--- 步数 {step_idx+1}/{steps_to_run} (已完成插值运行) ---")
                    step_idx += 1

            # 严格按照15Hz控制大循环耗时
            elapsed_loop = time.perf_counter() - loop_start
            if elapsed_loop < DT:
                time.sleep(DT - elapsed_loop)

    except KeyboardInterrupt:
        print("\n🛑 用户停止")
    finally:
        print("正在关闭...")
        # 新增关闭推理线程
        if 'inference_worker' in locals():
            inference_worker.stop()
            inference_worker.join(timeout=1.0)
            
        save_log(inference_log)
        cam_main.stop()
        cam_wrist_left.stop()
        cam_wrist_right.stop()
        if piper_left: 
            try: piper_left.DisconnectPort() 
            except: pass
        if piper_right: 
            try: piper_right.DisconnectPort()
            except: pass

if __name__ == "__main__":
    main()