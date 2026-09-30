import time
import csv
import numpy as np
import cv2
import threading
import copy  
import requests
import os
from typing import Any

# ================= 🌟 解决内网服务器被代理拦截 (HTTP 403) =================
# 强制让访问本地和 VLA 服务器的请求不走代理
os.environ["no_proxy"] = "localhost,127.0.0.1,172.19.0.36"

# ================= 🌟 导入 Gemini 大模型库 =================
import google.generativeai as genai

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
    print("⚠️ 警告: 未找到 piper_sdk")

# ================= 配置区域 =================
SERVER_HOST = "172.19.0.36"  # 你的 4090 策略服务器 IP
SERVER_PORT = 9000

# 本地视觉服务器 (server_api.py) 的 URL
VISION_SERVER_URL = "http://172.19.0.36:8000"

# ================= Gemini 大模型配置 =================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")  # 请填入你的 API KEY
genai.configure(api_key=GEMINI_API_KEY)

# ================= 全局控制变量 =================
TASK_INSTRUCTION = "" 
CONTROL_RATE_HZ = 30
DT = 1.0 / CONTROL_RATE_HZ

INTERPOLATION_STEPS = 3
SUB_DT = DT / INTERPOLATION_STEPS  

# RHC 滚动执行与多线程协同参数
EXECUTION_STEPS = 80
ACTION_HORIZON = 50
PREFETCH_STEPS = 5

JOINT_LIMITS_RAD = np.array([[-2.61799, 2.61799],[0.0, 3.14159],[-2.96706, 0.0],[-1.74533, 1.74533],[-1.22173, 1.22173],[-2.0944, 2.0944]])

ACTION_EMA_ALPHA = 0.35  
latest_chunk_lock = threading.Lock()
latest_action_chunk = None
new_chunk_available = False

infer_trigger_event = threading.Event()

# 绝对时间轴同步
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
        
        if self.name == "main":
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
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


# ================= 独立异步推理线程 (VLA) =================
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


# ================= 机械臂辅助函数 =================
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
        piper.ModeCtrl(ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=30, is_mit_mode=0x00)
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


# ================== 🌟 Gemini 意图推断逻辑 ==================
def infer_intent_with_gemini(all_detected_objects, selected_target):
    """调用 Gemini 大模型推断真实意图"""
    print(f"\n🧠 [LLM 意图推理] 正在思考... (桌面物品: {all_detected_objects} | 用户焦点: {selected_target})")
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = fprompt = f"""
        You are an intelligent robotic assistant reasoning about human intent. 
        Context: A paralyzed user is controlling a dual-arm robot via a Brain-Computer Interface. 
        
        The camera detects the following objects on the table: {all_detected_objects}. 
        The user is focusing their BCI selection on this specific target: "{selected_target}". 
        
        Task: Based on common sense, infer the most logical assistive action the user wants the robot to perform with the selected target, considering the other objects available.
        For example:
        - If there is a "bowl" and the user selects "apple", they likely want to put the apple in the bowl.
        - If there is a "cup" and the user selects "bottle", they likely want to pour water into the cup.
        - If the user selects "cloth", "towel", or "clothes", they likely want to fold it (e.g., "fold the cloth").
        - If they just select a rigid object with no obvious container, they might just want to pick it up.
        
        Output requirement: Return ONLY a short, direct English command string suitable for a Vision-Language-Action (VLA) robot model. Do not include any explanations, reasoning, or quotation marks.
        """
        response = model.generate_content(prompt)
        inferred_command = response.text.strip().strip('"').strip("'")
        print(f"💡 [LLM 意图推理] 推理完成！大模型生成的 VLA 指令为: -> {inferred_command} <-")
        return inferred_command
    except Exception as e:
        print(f"❌ [LLM 意图推理] 调用 Gemini 失败: {e}")
        fallback_command = f"pick up the {selected_target}."
        print(f"⚠️ [LLM 意图推理] 触发保底策略: -> {fallback_command} <-")
        return fallback_command


# ================== 🌟 核心：双机协同 BCI 交互逻辑与计时 ==================
def bci_selection_phase(cam_main):
    """Linux 端：触发场景更新，等待 Windows 端的脑电解码结果，并记录各个模块的耗时"""
    global TASK_INSTRUCTION
    latency_log = {}
    
    # 🌟 1. 强制等待摄像头预热，并增加重试机制
    print("📸 正在等待主摄像头画面...")
    raw_frame = None
    for _ in range(10): # 最多等 5 秒
        time.sleep(0.5)
        raw_frame = cam_main.get_frame()
        if raw_frame is not None:
            break
            
    if raw_frame is None:
        print("❌ 致命错误: 无法获取主摄像头画面！请检查 USB 连接或分辨率设置。")
        TASK_INSTRUCTION = "" # 置空，让主程序直接退出
        return latency_log

    print("📡 正在将当前场景发送至视觉服务器...")
    t_vision_start = time.time()  # 开始视觉计时
    
    _, img_encoded = cv2.imencode('.jpg', raw_frame)
    files = {'image_file': ('main_cam.jpg', img_encoded.tobytes(), 'image/jpeg')}
    
    VISION_MODE = "phrase"  
    PROMPT_TEXT = "cloth" 
    # PROMPT_TEXT = "bowl  , grapes ,cherry ,starfruit" 
    data = {'mode': VISION_MODE, 'prompt': PROMPT_TEXT, 'threshold': '0.4'}
    
    try:
        response = requests.post(f"{VISION_SERVER_URL}/update_scene", files=files, data=data, timeout=30)
        res_json = response.json()
        if res_json.get("status") != "success":
            print(f"❌ 视觉服务器处理失败: {res_json.get('message')}")
            TASK_INSTRUCTION = ""
            return latency_log
    except Exception as e:
        print(f"❌ 请求视觉服务器异常: {e}")
        TASK_INSTRUCTION = ""
        return latency_log
        
    latency_log['T_Vision'] = time.time() - t_vision_start

    # 3. 阻塞等待 Windows 端传回脑电解码结果
    print("⏳ 等待 Windows 客户端进行 BCI 闪烁与解码...")
    t_bci_start = time.time()  # 开始 BCI 计时
    selected_target = None
    
    wait_start = time.time()
    while time.time() - wait_start < 999:
        try:
            resp = requests.get(f"{VISION_SERVER_URL}/get_bci_result", timeout=5).json()
            if resp["status"] == "success":
                selected_target = resp["selected_target"]
                break
        except:
            pass
        time.sleep(0.5)
        
    if not selected_target:
        print("❌ 等待 Windows 脑电结果超时！")
        TASK_INSTRUCTION = ""
        return latency_log
        
    latency_log['T_BCI'] = time.time() - t_bci_start
    print(f"🎯 收到 Windows 端解码结果: {selected_target}")

    # 4. 获取当前桌面的所有物品标签，供 Gemini 参考
    try:
        scene_data = requests.get(f"{VISION_SERVER_URL}/get_bci_scene", timeout=5).json()
        all_labels = scene_data.get("labels", [])
    except:
        all_labels = [selected_target]

    # 5. 移交大模型进行意图消歧
    t_llm_start = time.time()  # 开始 LLM 计时
    TASK_INSTRUCTION = infer_intent_with_gemini(all_detected_objects=all_labels, selected_target=selected_target)
    latency_log['T_LLM'] = time.time() - t_llm_start
    
    print(f"✨ 意图确认！正式传给 VLA 的指令: -> \"{TASK_INSTRUCTION}\" <-")
    return latency_log

# ================= 主程序 =================
def main():
    global latest_action_chunk, new_chunk_available, absolute_step_counter

    piper_left = setup_piper("can_left")
    piper_right = setup_piper("can_right")
    
    print("正在连接摄像头 (多线程)...")
    cam_main = CameraThread("/dev/v4l/by-path/pci-0000:00:14.0-usb-0:3.2:1.0-video-index0", "main")
    cam_wrist_left = CameraThread("/dev/v4l/by-path/pci-0000:00:14.0-usb-0:5.2:1.0-video-index0", "left")
    cam_wrist_right = CameraThread("/dev/v4l/by-path/pci-0000:00:14.0-usb-0:4.2:1.0-video-index0", "right")

    # 🌟 在机械臂启动前，挂起等待 BCI 交互完成，并接收前置耗时日志
    latency_log = bci_selection_phase(cam_main)
    
    if not TASK_INSTRUCTION:
        print("未生成有效的控制指令，系统关闭。")
        cam_main.stop()
        cam_wrist_left.stop()
        cam_wrist_right.stop()
        return
    
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
        home_pose = np.array([0, 0.0043, 0.0345, -0.0535, 0.0, -0.0020]) 
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
    
    print("🔄 触发第一次拍摄，等待模型首次输出...")
    infer_trigger_event.set()
    
    while True:
        with latest_chunk_lock:
            if latest_action_chunk is not None:
                break
        time.sleep(0.1)
    
    print("▶️ 开始连续执行动作...")
    t_robot_start = time.time()  # 🌟 开始记录机器人物理执行耗时

    smoothed_target_l = get_current_joints_rad(piper_left)
    smoothed_target_r = get_current_joints_rad(piper_right)
    smoothed_grip_l = get_gripper_value(piper_left) / 70.0
    smoothed_grip_r = get_gripper_value(piper_right) / 70.0

    last_sent_grip_l = smoothed_grip_l
    last_sent_grip_r = smoothed_grip_r

    current_chunk = None
    step_idx = ACTION_HORIZON 

    try:
        while True:
            loop_start = time.perf_counter()

            with latest_chunk_lock:
                if new_chunk_available and step_idx >= ACTION_HORIZON:
                    current_chunk = copy.deepcopy(latest_action_chunk)
                    new_chunk_available = False
                    
                    elapsed_steps = absolute_step_counter - latest_chunk_capture_step
                    step_idx = max(0, elapsed_steps)  
                    
                    print(f"\n⚡ 稳健过渡完成！当前物理总进度 {absolute_step_counter}，已为您自动跳过新动作块前 {step_idx} 帧历史动作。")

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

                smoothed_target_l = ACTION_EMA_ALPHA * target_l + (1 - ACTION_EMA_ALPHA) * smoothed_target_l
                smoothed_target_r = ACTION_EMA_ALPHA * target_r + (1 - ACTION_EMA_ALPHA) * smoothed_target_r
                
                GRIPPER_EMA_ALPHA = 0.2
                smoothed_grip_l = GRIPPER_EMA_ALPHA * grip_l + (1 - GRIPPER_EMA_ALPHA) * smoothed_grip_l
                smoothed_grip_r = GRIPPER_EMA_ALPHA * grip_r + (1 - GRIPPER_EMA_ALPHA) * smoothed_grip_r

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
        print("\n🛑 实验结束 (Ctrl+C触发)")
        if 't_robot_start' in locals():
            latency_log['T_Robot'] = time.time() - t_robot_start
            
    finally:
        print("正在关闭并保存系统耗时...")
        
        # ================= 🌟 打印并保存耗时拆解数据 =================
        if 'T_Robot' in latency_log:
            print("\n" + "="*50)
            print("📊 [实验数据] 系统耗时拆解 (Latency Breakdown):")
            print(f"  1. 视觉感知与传输 (T_Vision): {latency_log.get('T_Vision', 0):.2f} 秒")
            print(f"  2. 脑机交互与解码 (T_BCI)   : {latency_log.get('T_BCI', 0):.2f} 秒")
            print(f"  3. 大模型意图推理 (T_LLM)   : {latency_log.get('T_LLM', 0):.2f} 秒")
            print(f"  4. 机器人物理执行 (T_Robot) : {latency_log.get('T_Robot', 0):.2f} 秒")
            t_total = latency_log.get('T_Vision', 0) + latency_log.get('T_BCI', 0) + latency_log.get('T_LLM', 0) + latency_log.get('T_Robot', 0)
            print(f"  >> 论文核心数据 T_total     : {t_total:.2f} 秒")
            print("="*50 + "\n")
            
            try:
                with open("latency_results.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    if f.tell() == 0:
                        writer.writerow(["T_Vision", "T_BCI", "T_LLM", "T_Robot", "T_total"])
                    writer.writerow([
                        f"{latency_log.get('T_Vision', 0):.2f}", 
                        f"{latency_log.get('T_BCI', 0):.2f}", 
                        f"{latency_log.get('T_LLM', 0):.2f}", 
                        f"{latency_log.get('T_Robot', 0):.2f}",
                        f"{t_total:.2f}"
                    ])
                print("💾 耗时数据已自动追加至 latency_results.csv")
            except Exception as e:
                print(f"⚠️ 保存耗时数据失败: {e}")
        # ============================================================

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