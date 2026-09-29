import time
import csv
import os
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

# Receding-horizon and RTC timing parameters.
ACTION_HORIZON = int(os.environ.get("OPENPI_ACTION_HORIZON", "100"))
PREFETCH_STEPS = int(os.environ.get("OPENPI_PREFETCH_STEPS", "15"))
LATENCY_STEPS = int(os.environ.get("OPENPI_LATENCY_STEPS", "7"))
RTC_INITIAL_DELAY_STEPS = int(os.environ.get("OPENPI_RTC_INITIAL_DELAY_STEPS", "20"))
RTC_PREFIX_EXTRA_STEPS = int(os.environ.get("OPENPI_RTC_PREFIX_EXTRA_STEPS", "40"))
RTC_MAX_GUIDANCE_WEIGHT = float(os.environ.get("OPENPI_RTC_MAX_GUIDANCE_WEIGHT", "5"))
MAX_STALE_STEPS = int(os.environ.get("OPENPI_MAX_STALE_STEPS", "30"))

JOINT_LIMITS_RAD = np.array([[-2.61799, 2.61799],[0.0, 3.14159],[-2.96706, 0.0],[-1.74533, 1.74533],[-1.22173, 1.22173],[-2.0944, 2.0944],
])

latest_chunk_lock = threading.Lock()
latest_action_chunk = None
new_chunk_available = False
active_action_chunk = None
active_chunk_capture_step = 0

infer_trigger_event = threading.Event()

# ================= Absolute action timeline =================
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
        self._request_pending = threading.Event()
        self.action_horizon = ACTION_HORIZON
        self._latency_estimate_steps = min(
            ACTION_HORIZON, max(LATENCY_STEPS, RTC_INITIAL_DELAY_STEPS)
        )

    def request(self):
        if self.running and not self._request_pending.is_set():
            self._request_pending.set()
            infer_trigger_event.set()

    def _make_rtc_context(self, capture_step):
        with latest_chunk_lock:
            previous_chunk = copy.deepcopy(active_action_chunk)
            previous_capture_step = active_chunk_capture_step

        if previous_chunk is None:
            return None

        current_index = capture_step - previous_capture_step + LATENCY_STEPS
        current_index = int(np.clip(current_index, 0, previous_chunk.shape[0] - 1))
        aligned_chunk = previous_chunk[current_index:]
        if aligned_chunk.shape[0] < self.action_horizon:
            padding = np.repeat(
                aligned_chunk[-1:, :],
                self.action_horizon - aligned_chunk.shape[0],
                axis=0,
            )
            aligned_chunk = np.concatenate([aligned_chunk, padding], axis=0)
        else:
            aligned_chunk = aligned_chunk[:self.action_horizon]

        inference_delay = int(
            np.clip(self._latency_estimate_steps, 0, self.action_horizon)
        )
        prefix_horizon = min(
            self.action_horizon,
            inference_delay + max(0, RTC_PREFIX_EXTRA_STEPS),
        )
        return {
            "version": 1,
            "mode": "realtime",
            "prev_action_chunk": aligned_chunk.astype(np.float32, copy=False),
            "inference_delay": inference_delay,
            "prefix_attention_horizon": prefix_horizon,
            "prefix_attention_schedule": "exp",
            "max_guidance_weight": RTC_MAX_GUIDANCE_WEIGHT,
        }

    def run(self):
        global latest_action_chunk, new_chunk_available, latest_chunk_capture_step

        while self.running:
            infer_trigger_event.wait()
            infer_trigger_event.clear()
            if not self.running:
                break

            request_again = False
            try:
                capture_step = absolute_step_counter
                img = self.cam_main.get_frame()
                w_l = self.cam_l.get_frame()
                w_r = self.cam_r.get_frame()

                if img is None or w_l is None or w_r is None:
                    print("⚠️ 摄像头图像尚未就绪，稍后重试。")
                    time.sleep(0.1)
                    request_again = True
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    w_l = cv2.cvtColor(w_l, cv2.COLOR_BGR2RGB)
                    w_r = cv2.cvtColor(w_r, cv2.COLOR_BGR2RGB)

                    curr_l_rad = get_current_joints_rad(self.piper_left)
                    curr_r_rad = get_current_joints_rad(self.piper_right)

                    val_l_mm = get_gripper_value(self.piper_left)
                    val_r_mm = get_gripper_value(self.piper_right)
                    # Aloha transforms represent gripper width in decimeters.
                    norm_grip_l = np.clip(val_l_mm / 100.0, 0.0, 0.7)
                    norm_grip_r = np.clip(val_r_mm / 100.0, 0.0, 0.7)

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
                    rtc_context = self._make_rtc_context(capture_step)
                    if rtc_context is not None:
                        obs["rtc"] = rtc_context

                    request_start = time.monotonic()
                    resp = self.client.infer(obs)
                    round_trip_seconds = time.monotonic() - request_start
                    observed_delay = int(
                        np.ceil(round_trip_seconds * CONTROL_RATE_HZ)
                    ) + LATENCY_STEPS
                    self._latency_estimate_steps = int(
                        np.clip(
                            round(
                                0.5 * self._latency_estimate_steps
                                + 0.5 * observed_delay
                            ),
                            LATENCY_STEPS,
                            self.action_horizon,
                        )
                    )

                    if not resp or "actions" not in resp:
                        raise RuntimeError("策略服务器未返回 actions 字段。")
                    all_actions = np.asarray(resp["actions"], dtype=np.float32)
                    if all_actions.ndim == 3 and all_actions.shape[0] == 1:
                        all_actions = all_actions[0]
                    expected_shape = (self.action_horizon, 14)
                    if all_actions.shape != expected_shape:
                        raise RuntimeError(
                            f"动作块形状应为 {expected_shape}，实际为 {all_actions.shape}。"
                        )
                    if not np.isfinite(all_actions).all():
                        raise RuntimeError("策略服务器返回了非有限动作值。")
                    all_actions[:, 0:6] = np.clip(
                        all_actions[:, 0:6],
                        JOINT_LIMITS_RAD[:, 0],
                        JOINT_LIMITS_RAD[:, 1],
                    )
                    all_actions[:, 6] = np.clip(all_actions[:, 6], 0.0, 0.7)
                    all_actions[:, 7:13] = np.clip(
                        all_actions[:, 7:13],
                        JOINT_LIMITS_RAD[:, 0],
                        JOINT_LIMITS_RAD[:, 1],
                    )
                    all_actions[:, 13] = np.clip(all_actions[:, 13], 0.0, 0.7)

                    with latest_chunk_lock:
                        latest_action_chunk = all_actions
                        latest_chunk_capture_step = capture_step
                        new_chunk_available = True

                    used_delay = rtc_context["inference_delay"] if rtc_context else 0
                    print(
                        f"✅[RTC推理] 收到 {all_actions.shape[0]} 步动作，"
                        f"请求帧 {capture_step}，往返 {round_trip_seconds:.2f}s，"
                        f"本次前缀延迟 {used_delay} 步。"
                    )
            except Exception as e:
                print(f"⚠️ 推理请求异常: {e}")
                time.sleep(0.2)
                request_again = True
            finally:
                self._request_pending.clear()

            if request_again and self.running:
                self.request()

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
    global ACTION_HORIZON, active_action_chunk, active_chunk_capture_step

    enable_robot = os.environ.get("OPENPI_ENABLE_ROBOT") == "1"
    dry_run = os.environ.get("OPENPI_DRY_RUN") == "1"
    if not enable_robot and not dry_run:
        raise SystemExit(
            "硬件输出默认锁定。先用 OPENPI_DRY_RUN=1 检查服务端；确认急停与工作区安全后，"
            "再设置 OPENPI_ENABLE_ROBOT=1 启动。"
        )

    print(f"连接策略服务器: {SERVER_HOST}:{SERVER_PORT}")
    try:
        client = websocket_client_policy.WebsocketClientPolicy(
            host=SERVER_HOST, port=SERVER_PORT
        )
        server_metadata = client.get_server_metadata()
    except Exception as e:
        raise SystemExit(f"策略服务器连接失败，机械臂未使能: {e}")

    rtc_metadata = server_metadata.get("openpi_rtc", {})
    if rtc_metadata.get("version") != 1 or "realtime" not in rtc_metadata.get("modes", []):
        raise SystemExit(
            "策略服务器没有声明 OpenPI RTC v1 支持；为避免静默退回普通分块推理，机械臂未使能。"
        )

    ACTION_HORIZON = int(server_metadata.get("action_horizon", ACTION_HORIZON))
    action_dim = int(server_metadata.get("action_dim", 14))
    if ACTION_HORIZON <= 0 or action_dim != 14:
        raise SystemExit(
            f"服务器策略形状不匹配：action_horizon={ACTION_HORIZON}, action_dim={action_dim}；"
            "双 Piper 控制要求 action_dim=14。"
        )
    if PREFETCH_STEPS <= 0 or PREFETCH_STEPS >= ACTION_HORIZON:
        raise SystemExit(
            f"OPENPI_PREFETCH_STEPS 必须在 1 到 {ACTION_HORIZON - 1} 之间。"
        )

    if dry_run:
        print(
            f"✅ 干运行通过：服务端声明 RTC v1，动作形状为 "
            f"{ACTION_HORIZON}×{action_dim}；没有连接或使能机械臂。"
        )
        client._ws.close()
        return

    piper_left = setup_piper("can_left")
    piper_right = setup_piper("can_right")
    
    print("正在连接摄像头 (多线程)...")
    cam_main = CameraThread("/dev/v4l/by-path/pci-0000:00:14.0-usb-0:6.2:1.0-video-index0", "main")
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

    print("启动异步推理线程...")
    inference_worker = InferenceThread(client, cam_main, cam_wrist_left, cam_wrist_right, piper_left, piper_right)
    inference_worker.start()

    inference_log =[]
    
    print("🔄 等待模型输出...")
    inference_worker.request()

    while True:
        with latest_chunk_lock:
            if latest_action_chunk is not None:
                break
        time.sleep(0.1)
    
    print("▶️ 开始连续执行动作...")

    smoothed_target_l = get_current_joints_rad(piper_left)
    smoothed_target_r = get_current_joints_rad(piper_right)
    smoothed_grip_l = get_gripper_value(piper_left) / 100.0
    smoothed_grip_r = get_gripper_value(piper_right) / 100.0

    current_chunk = None
    step_idx = ACTION_HORIZON 

    try:
        while True:
            loop_start = time.perf_counter()

            # --- 1. 检查是否接收新 Chunk ---
            with latest_chunk_lock:
                if new_chunk_available and (current_chunk is None or step_idx >= ACTION_HORIZON):
                    had_previous_chunk = current_chunk is not None
                    elapsed_steps = absolute_step_counter - latest_chunk_capture_step
                    new_step_idx = max(0, elapsed_steps + LATENCY_STEPS) if had_previous_chunk else 0
                    if had_previous_chunk and new_step_idx >= latest_action_chunk.shape[0]:
                        print(
                            "❌ RTC动作块到达时已过期；停止发送机械臂动作，请检查推理延迟。"
                        )
                        break
                    current_chunk = copy.deepcopy(latest_action_chunk)
                    new_chunk_available = False
                    step_idx = new_step_idx
                    active_action_chunk = copy.deepcopy(current_chunk)
                    active_chunk_capture_step = latest_chunk_capture_step

                    print(
                        f"\n⚡ RTC动作块切换：物理帧 {absolute_step_counter}，"
                        f"新块时间对齐索引 {step_idx}。"
                    )

            # --- 2. 连续执行动作 ---
            if current_chunk is not None:
                if step_idx >= ACTION_HORIZON + MAX_STALE_STEPS and not new_chunk_available:
                    print(
                        f"❌ 连续 {MAX_STALE_STEPS} 步未收到新动作；停止发送动作并退出控制循环。"
                    )
                    break

                if step_idx == ACTION_HORIZON - PREFETCH_STEPS or \
                   (step_idx >= ACTION_HORIZON and not new_chunk_available and step_idx % 5 == 0):
                    print(f"   [RTC同步] 触发抓图请求... (物理帧: {absolute_step_counter})")
                    inference_worker.request()

                safe_idx = min(step_idx, current_chunk.shape[0] - 1)
                action_step = current_chunk[safe_idx]
                
                target_l, target_r, grip_l, grip_r = map_actions_to_joint_targets_dual(
                    action_step, smoothed_target_l, smoothed_target_r
                )

                # RTC conditions the next chunk on these executed setpoints, so
                # avoid post-hoc EMA/blending that would change the planned prefix.
                smoothed_target_l = target_l
                smoothed_target_r = target_r
                smoothed_grip_l = float(np.clip(grip_l, 0.0, 1.0))
                smoothed_grip_r = float(np.clip(grip_r, 0.0, 1.0))

                # Send the policy gripper setpoints directly so later RTC requests
                # condition on the same action plan the controller is executing.
                val_l = int(np.clip(smoothed_grip_l * 100000, 0, 70000))
                val_r = int(np.clip(smoothed_grip_r * 100000, 0, 70000))
                if piper_left:
                    piper_left.GripperCtrl(val_l, 3000, 0x01)
                if piper_right:
                    piper_right.GripperCtrl(val_r, 3000, 0x01)

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