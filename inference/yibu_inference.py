import time
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Any
import threading
import copy  

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
    print("⚠️ 警告: 未找到 piper_sdk。")

# ================= 配置区域 =================

SERVER_HOST = "172.19.5.252"  
SERVER_PORT = 9000
TASK_INSTRUCTION = "fold the cloth."
# TASK_INSTRUCTION = "pick up the cherry and put it into the bowl."
# TASK_INSTRUCTION = "pick up the brunch of grapes and put it into the bowl."


CONTROL_RATE_HZ = 30
DT = 1.0 / CONTROL_RATE_HZ

INTERPOLATION_STEPS = 3
SUB_DT = DT / INTERPOLATION_STEPS  

# RHC 滚动执行与多线程协同参数
EXECUTION_STEPS = 80
ACTION_HORIZON = 50    # 🌟 恢复到70：必须执行到 Chunk 的后半段，否则永远触发不了夹爪闭合！
PREFETCH_STEPS = 5      # 🌟 提前8帧拍照推理，掩盖网络延迟

LATENCY_STEPS = 7       # 补偿硬件延迟，防止换块时向后回溯
BLEND_STEPS = 10         # 仅对关节进行 6 帧的平滑过渡

JOINT_LIMITS_RAD = np.array([[-2.61799, 2.61799],[0.0, 3.14159],[-2.96706, 0.0],[-1.74533, 1.74533],[-1.22173, 1.22173],[-2.0944, 2.0944],
])

ACTION_EMA_ALPHA = 0.35  
latest_chunk_lock = threading.Lock()
latest_action_chunk = None
new_chunk_available = False

infer_trigger_event = threading.Event()

# ================= 绝对时间轴同步 =================
absolute_step_counter = 0  
latest_chunk_capture_step = 0  

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
        print(f"正在尝试连接摄像头 {self.name} : {self.path}")
        self.cap = cv2.VideoCapture(self.path)
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

# ================= 独立异步推理线程 =================
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
        global latest_action_chunk, new_chunk_available, latest_chunk_capture_step
        while self.running:
            infer_trigger_event.wait()
            infer_trigger_event.clear()
            
            capture_step = absolute_step_counter
            
            img = self.cam_main.get_frame()
            w_l = self.cam_l.get_frame()
            w_r = self.cam_r.get_frame()
                    
            if img is None or w_l is None or w_r is None:
                time.sleep(0.1)
                infer_trigger_event.set() 
                continue
        
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            w_l = cv2.cvtColor(w_l, cv2.COLOR_BGR2RGB)
            w_r = cv2.cvtColor(w_r, cv2.COLOR_BGR2RGB)

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

            try:
                resp = self.client.infer(obs)
                if resp and "actions" in resp:
                    all_actions = np.array(resp["actions"])
                    if all_actions.ndim == 3:  
                        all_actions = all_actions[0]
                    
                    with latest_chunk_lock:
                        latest_action_chunk = all_actions
                        latest_chunk_capture_step = capture_step 
                        new_chunk_available = True
                    
                    horizon = all_actions.shape[0]
                    print(f"✅[推理] 响应就绪 (共 {horizon} 步) | 对应物理帧: {capture_step}")
                else:
                    print("❌ 服务器未返回 actions 字段")
                    infer_trigger_event.set() 
            except Exception as e:
                print(f"⚠️ 推理请求异常: {e}")
                time.sleep(0.2)
                infer_trigger_event.set() 

    def stop(self):
        self.running = False
        infer_trigger_event.set()

# ================= 辅助函数 =================

def clamp_rad(rad_targets: np.ndarray) -> np.ndarray:
    return np.clip(rad_targets, JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])

def deg_to_sdk_unit(rad: np.ndarray) -> np.ndarray:
    deg = np.rad2deg(rad)
    return (deg * 1000.0).astype(np.int32)

def get_current_joints_rad(piper: Any) -> np.ndarray:
    if piper is None: return np.zeros(6)
    try:
        js_ret = piper.GetArmJointMsgs()
        if isinstance(js_ret, tuple) and len(js_ret) >= 3:
            js = js_ret[2]
        else:
            js = js_ret
            
        if hasattr(js, "joint_state") and js.joint_state:
            st = js.joint_state
            for names in [["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],["angle_1", "angle_2", "angle_3", "angle_4", "angle_5", "angle_6"]]:
                try:
                    vals =[getattr(st, n) for n in names]
                    return np.deg2rad(np.array(vals, dtype=np.float64) / 1000.0)
                except AttributeError: continue
        return np.zeros(6)
    except Exception as e:
        return np.zeros(6)

def get_gripper_value(piper):
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
    action = np.array(action_step, dtype=np.float64)
    left_target_raw = action[0:6]
    left_grip_raw = float(action[6])
    right_target_raw = action[7:13]
    right_grip_raw = float(action[13])

    left_target = clamp_rad(left_target_raw)
    right_target = clamp_rad(right_target_raw)
    
    return left_target, right_target, left_grip_raw, right_grip_raw

def setup_piper(can_name: str):
    if C_PiperInterface_V2 is None: return None
    print(f"正在连接 {can_name} ...")
    try:
        piper = C_PiperInterface_V2(can_name=can_name, judge_flag=True, can_auto_init=True)
        piper.CreateCanBus(can_name=can_name, bustype="socketcan", expected_bitrate=1000000, judge_flag=False)
        piper.ConnectPort(can_init=False, piper_init=True, start_thread=True)
        return piper
    except Exception as e:
        print(f"❌ {can_name} 连接失败: {e}")
        return None

def process_image(img_data):
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
    global latest_action_chunk, new_chunk_available, absolute_step_counter

    piper_left = setup_piper("can_left")
    piper_right = setup_piper("can_right")
    
    print("正在连接摄像头 (多线程)...")
    cam_main = CameraThread("/dev/v4l/by-path/pci-0000:00:14.0-usb-0:3.2:1.0-video-index0", "main")
    cam_wrist_left = CameraThread("/dev/v4l/by-path/pci-0000:00:14.0-usb-0:5.2:1.0-video-index0", "left")
    cam_wrist_right = CameraThread("/dev/v4l/by-path/pci-0000:00:14.0-usb-0:4.2:1.0-video-index0", "right")
    
    if piper_left and piper_right:
        print(">>> 正在使能机械臂 <<<")
        for _ in range(3):
            if piper_left:
                piper_left.EnableArm(motor_num=7, enable_flag=0x02)
                piper_left.GripperCtrl(0, 300, 0x01)
            if piper_right:
                piper_right.EnableArm(motor_num=7, enable_flag=0x02)
                piper_right.GripperCtrl(0, 300, 0x01)
            time.sleep(0.1)  
        time.sleep(0.5)
        start_l = get_current_joints_rad(piper_left)
        start_r = get_current_joints_rad(piper_right)
        
        retry_count = 0
        while (np.all(start_l == 0) or np.all(start_r == 0)):
            print("⚠️ 尚未收到机械臂真实回传数据，正在重试...")
            time.sleep(0.2)
            start_l = get_current_joints_rad(piper_left)
            start_r = get_current_joints_rad(piper_right)
            retry_count += 1
        print(">>> 移动到初始位姿 <<<")
        if piper_left:
            piper_left.ModeCtrl(ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=30, is_mit_mode=0x00)
        if piper_right:
            piper_right.ModeCtrl(ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=30, is_mit_mode=0x00)
        time.sleep(0.1)
        home_pose = np.array([0, 0.0043, 0.0345, -0.0535, 0.0, -0.0020]) 
        
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
        time.sleep(0.2)
    else:
        print("❌ 机械臂连接失败")
        return

    print(f"连接策略服务器: {SERVER_HOST}:{SERVER_PORT}")
    try:
        client = websocket_client_policy.WebsocketClientPolicy(host=SERVER_HOST, port=SERVER_PORT)
    except Exception as e:
        print(f"❌ 连接服务器失败: {e}")
        return
    print("启动异步推理线程...")
    inference_worker = InferenceThread(client, cam_main, cam_wrist_left, cam_wrist_right, piper_left, piper_right)
    inference_worker.start()

    inference_log =[]
    
    print("🔄 等待模型输出...")
    infer_trigger_event.set()
    
    while True:
        with latest_chunk_lock:
            if latest_action_chunk is not None:
                break
        time.sleep(0.1)
    
    print("▶️ 开始连续执行动作...")

    smoothed_target_l = get_current_joints_rad(piper_left)
    smoothed_target_r = get_current_joints_rad(piper_right)
    smoothed_grip_l = get_gripper_value(piper_left) / 70.0
    smoothed_grip_r = get_gripper_value(piper_right) / 70.0

    last_sent_grip_l = smoothed_grip_l
    last_sent_grip_r = smoothed_grip_r

    current_chunk = None
    step_idx = ACTION_HORIZON 

    old_chunk = None
    old_step_idx = 0
    blend_counter = 0

    try:
        while True:
            loop_start = time.perf_counter()

            # --- 1. 检查是否接收新 Chunk ---
            with latest_chunk_lock:
                if new_chunk_available and step_idx >= ACTION_HORIZON:
                    old_chunk = copy.deepcopy(current_chunk) if current_chunk is not None else None
                    old_step_idx = step_idx
                    
                    current_chunk = copy.deepcopy(latest_action_chunk)
                    new_chunk_available = False
                    
                    elapsed_steps = absolute_step_counter - latest_chunk_capture_step
                    # 补偿硬件延迟，对齐时间轴
                    step_idx = max(0, elapsed_steps + LATENCY_STEPS)  
                    
                    blend_counter = BLEND_STEPS  
                    
                    print(f"\n⚡ 稳健过渡完成！当前物理进度 {absolute_step_counter}，自动跳过新动作前 {step_idx} 帧，开启平滑融合。")

            # --- 2. 连续执行动作 ---
            if current_chunk is not None:
                if step_idx == ACTION_HORIZON - PREFETCH_STEPS or \
                   (step_idx >= ACTION_HORIZON and not new_chunk_available and step_idx % 5 == 0):
                    print(f"   [同步] 触发抓图请求... (此时物理帧: {absolute_step_counter})")
                    infer_trigger_event.set()

                safe_idx = min(step_idx, current_chunk.shape[0] - 1)
                action_step = current_chunk[safe_idx]
                
                target_l, target_r, grip_l, grip_r = map_actions_to_joint_targets_dual(
                    action_step, smoothed_target_l, smoothed_target_r
                )

                # ================= 🌟 轨迹融合：仅针对关节，绝对放过夹爪！ =================
                if blend_counter > 0 and old_chunk is not None:
                    safe_old_idx = min(old_step_idx, old_chunk.shape[0] - 1)
                    old_action_step = old_chunk[safe_old_idx]
                    
                    # 提取旧动作的关节坐标，直接无视旧动作的夹爪值
                    old_t_l, old_t_r, _, _ = map_actions_to_joint_targets_dual(
                        old_action_step, smoothed_target_l, smoothed_target_r
                    )
                    
                    w = 1.0 - (blend_counter / float(BLEND_STEPS))
                    
                    # 平滑位移
                    target_l = old_t_l * (1 - w) + target_l * w
                    target_r = old_t_r * (1 - w) + target_r * w
                    
                    # 注意：不修改 grip_l 和 grip_r，让夹爪果断执行新Chunk的指令
                    
                    old_step_idx += 1
                    blend_counter -= 1

                # ================= a. 关节与夹爪的 EMA 平滑 =================
                smoothed_target_l = ACTION_EMA_ALPHA * target_l + (1 - ACTION_EMA_ALPHA) * smoothed_target_l
                smoothed_target_r = ACTION_EMA_ALPHA * target_r + (1 - ACTION_EMA_ALPHA) * smoothed_target_r
                
                # 🌟 减弱夹爪的滤波：让夹爪响应更迅猛 (改为 0.7 甚至更高，0.2太软会导致永远夹不紧)
                GRIPPER_EMA_ALPHA = 0.7
                smoothed_grip_l = GRIPPER_EMA_ALPHA * grip_l + (1 - GRIPPER_EMA_ALPHA) * smoothed_grip_l
                smoothed_grip_r = GRIPPER_EMA_ALPHA * grip_r + (1 - GRIPPER_EMA_ALPHA) * smoothed_grip_r

                # ================= b. 夹爪专属：死区滤波与降频发送 =================
                GRIPPER_DEADBAND = 0.015  
                
                if abs(smoothed_grip_l - last_sent_grip_l) > GRIPPER_DEADBAND:
                    val_l = int(np.clip(smoothed_grip_l * 70000, 0, 70000))
                    if piper_left: 
                        piper_left.GripperCtrl(val_l, 3000, 0x01)
                    last_sent_grip_l = smoothed_grip_l

                if abs(smoothed_grip_r - last_sent_grip_r) > GRIPPER_DEADBAND:
                    val_r = int(np.clip(smoothed_grip_r * 70000, 0, 70000))
                    if piper_right: 
                        piper_right.GripperCtrl(val_r, 3000, 0x01)
                    last_sent_grip_r = smoothed_grip_r

                # ================= c. 硬件插值逻辑 (仅限机械臂关节) =================
                prev_l_interp = get_current_joints_rad(piper_left)
                prev_r_interp = get_current_joints_rad(piper_right)

                for k in range(INTERPOLATION_STEPS):
                    sub_start = time.perf_counter()
                    alpha = (k + 1) / INTERPOLATION_STEPS
                    
                    interp_l = prev_l_interp * (1 - alpha) + smoothed_target_l * alpha
                    interp_r = prev_r_interp * (1 - alpha) + smoothed_target_r * alpha

                    if piper_left and piper_right:
                        piper_left.JointCtrl(*deg_to_sdk_unit(interp_l))
                        piper_right.JointCtrl(*deg_to_sdk_unit(interp_r))

                    elapsed_sub = time.perf_counter() - sub_start
                    if elapsed_sub < SUB_DT:
                        time.sleep(SUB_DT - elapsed_sub)
                
                inference_log.append([time.time(), *smoothed_target_l, smoothed_grip_l, *smoothed_target_r, smoothed_grip_r])
                
                if step_idx < ACTION_HORIZON:
                    print(f"--- 步数 {step_idx+1}/{ACTION_HORIZON} (总步数 {absolute_step_counter}) ---")
                
                step_idx += 1
                absolute_step_counter += 1

            elapsed_loop = time.perf_counter() - loop_start
            if elapsed_loop < DT:
                time.sleep(DT - elapsed_loop)

    except KeyboardInterrupt:
        print("\n🛑 用户停止")
    finally:
        print("正在关闭...")
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