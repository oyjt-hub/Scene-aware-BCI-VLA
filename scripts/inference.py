import time
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Any
import threading
import copy  
import requests
import base64
import json
import os
# ================= 🌟 New: Import Gemini LLM library =================
import google.generativeai as genai
# ======================================================================
LOG_DIR = "./csv_logs/starfruit"
# --- OpenPi and SDK imports ---
try:
    from openpi_client import image_tools, websocket_client_policy
except ImportError:
    print("❌ Error: openpi_client not found, please check the installation.")
    exit(1)

try:
    from piper_sdk import C_PiperInterface_V2
except ImportError:
    C_PiperInterface_V2 = None
    print("⚠️ Warning: piper_sdk not found")

# ================= Configuration =================
SERVER_HOST = "172.19.5.252"  # Your 4090 server IP
SERVER_PORT = 9000

# SAM2 server URL (Port 8000)
SAM2_SERVER_URL = f"http://{SERVER_HOST}:8000/update_scene"
# Candidate objects for BCI interface to identify and apply flashing masks
CANDIDATE_OBJECTS = " cherry, bowl, grapes, starfruit" 

# ================= 🌟 New: Ablation experiment switch =================
# True:  With LLM (Semantic Intent Synthesis, synthesize full action instructions via LLM)
# False: Without LLM (Direct Target Grounding, directly pass target object noun to VLA model)
USE_LLM = True
# ======================================================================

# ================= 🌟 New: Gemini LLM Configuration =================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")  # Please enter your API KEY here
genai.configure(api_key=GEMINI_API_KEY)
# If you cannot connect to Google directly, uncomment the lines below and set your proxy
# import os
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"
# =====================================================================

# Clear the static instruction, dynamically assign after stage 1 BCI selection
TASK_INSTRUCTION = "" 
CONTROL_RATE_HZ = 30
DT = 1.0 / CONTROL_RATE_HZ

INTERPOLATION_STEPS = 3
SUB_DT = DT / INTERPOLATION_STEPS  

# RHC rolling execution and multi-threading parameters
EXECUTION_STEPS = 80
ACTION_HORIZON = 50     # 🌟 Restored to 50: Must execute into the latter half of the chunk, otherwise gripper closure will never trigger!
PREFETCH_STEPS = 8      # 🌟 Prefetch 8 frames ahead for inference to mask network latency

LATENCY_STEPS = 3       # 🌟 Smooth addition: Compensate for hardware latency, preventing backtracking when switching chunks
BLEND_STEPS = 6         # 🌟 Smooth addition: Smooth transition over 6 frames for joints only

JOINT_LIMITS_RAD = np.array([[-2.61799, 2.61799],[0.0, 3.14159],[-2.96706, 0.0],[-1.74533, 1.74533],[-1.22173, 1.22173],[-2.0944, 2.0944],
])

ACTION_EMA_ALPHA = 0.35  
latest_chunk_lock = threading.Lock()
latest_action_chunk = None
new_chunk_available = False

infer_trigger_event = threading.Event()

# ================= Absolute Timeline Synchronization =================
absolute_step_counter = 0  
latest_chunk_capture_step = 0  
latest_infer_latency_ms = 0.0

# ================= Helper Class: Multi-threaded Camera =================
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
        print(f"Attempting to connect to camera {self.name} : {self.path}")
        self.cap = cv2.VideoCapture(self.path)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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
                print(f"Camera {self.name} disconnected, attempting to reconnect...")
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


# ================= Independent Async Inference Thread =================
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
                infer_start = time.perf_counter()

                resp = self.client.infer(obs)

                infer_end = time.perf_counter()

                global latest_infer_latency_ms
                latest_infer_latency_ms = (infer_end - infer_start) * 1000.0
                if resp and "actions" in resp:
                    all_actions = np.array(resp["actions"])
                    if all_actions.ndim == 3:  
                        all_actions = all_actions[0]
                    
                    with latest_chunk_lock:
                        latest_action_chunk = all_actions
                        latest_chunk_capture_step = capture_step 
                        new_chunk_available = True
                    
                    horizon = all_actions.shape[0]
                    print(f"✅ [Inference] Response ready ({horizon} steps) | Physical frame: {capture_step}")
                else:
                    print("❌ Server did not return 'actions' field")
                    infer_trigger_event.set() 
            except Exception as e:
                print(f"⚠️ Inference request exception: {e}")
                time.sleep(0.2)
                infer_trigger_event.set() 

    def stop(self):
        self.running = False
        infer_trigger_event.set()


# ================= Helper Functions =================

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
        print(f"Exception getting joint data: {e}")
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
    print(f"Connecting to {can_name} ...")
    try:
        piper = C_PiperInterface_V2(can_name=can_name, judge_flag=True, can_auto_init=True)
        piper.CreateCanBus(can_name=can_name, bustype="socketcan", expected_bitrate=1000000, judge_flag=False)
        piper.ConnectPort(can_init=False, piper_init=True, start_thread=True)
        
        return piper
    except Exception as e:
        print(f"❌ Failed to connect to {can_name}: {e}")
        return None

def process_image(img_data):
    resized = image_tools.resize_with_pad(img_data, 224, 224)
    uint8_img = image_tools.convert_to_uint8(resized)
    return uint8_img.transpose(2, 0, 1)

def save_log(log_data):
    # 🌟 Uniformly create LOG_DIR and build path here to avoid undefined CSV_LOG_DIR crash
    os.makedirs(LOG_DIR, exist_ok=True)
    filename = os.path.join(LOG_DIR, f"{int(time.time())}.csv")

    header = [
        "timestamp",
        "step",

        # Raw Prediction
        "pred_lj1","pred_lj2","pred_lj3","pred_lj4","pred_lj5","pred_lj6",
        "pred_lgrip",
        "pred_rj1","pred_rj2","pred_rj3","pred_rj4","pred_rj5","pred_rj6",
        "pred_rgrip",

        # EMA
        "ema_lj1","ema_lj2","ema_lj3","ema_lj4","ema_lj5","ema_lj6",
        "ema_lgrip",
        "ema_rj1","ema_rj2","ema_rj3","ema_rj4","ema_rj5","ema_rj6",
        "ema_rgrip",

        # Real Feedback
        "real_lj1","real_lj2","real_lj3","real_lj4","real_lj5","real_lj6",
        "real_lgrip",
        "real_rj1","real_rj2","real_rj3","real_rj4","real_rj5","real_rj6",
        "real_rgrip",

        # Tracking Error
        "err_lj1","err_lj2","err_lj3","err_lj4","err_lj5","err_lj6",
        "err_rj1","err_rj2","err_rj3","err_rj4","err_rj5","err_rj6",

        "infer_latency_ms",
        "control_period_ms"
    ]

    with open(filename,"w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(header)
        writer.writerows(log_data)

    print("Log saved to:", filename)

# ================== 🌟 Core: Gemini Intent Inference Logic ==================
def infer_intent_with_gemini(all_detected_objects, selected_target):
    """
    Infer human intent via Gemini LLM based on all detected objects and the user's selected target.
    Supports USE_LLM switch for ablation study.
    """
    # 🔬 Check if LLM is enabled (Ablation baseline: Direct Target Grounding)
    if not USE_LLM:
        print(f"\n🔬 [Ablation Study - Without LLM] LLM semantic reasoning disabled (Direct Target Grounding).")
        print(f"   Directly passing target object as VLA Prompt: -> {selected_target} <-")
        # Return object name directly without template completion for VLA model
        return selected_target

    # 🧠 With LLM branch (Semantic Intent Synthesis)
    print(f"\n🧠 [LLM Intent Inference] Thinking... (Table objects: {all_detected_objects} | User focus: {selected_target})")
    
    try:
        # Use the lightweight and fast gemini-2.5-flash model
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # prompt = f"""
        # You are an intelligent robotic assistant reasoning about human intent. 
        # Context: A paralyzed user is controlling a robot arm via a Brain-Computer Interface. 
        
        # The camera detects the following objects on the table: {all_detected_objects}. 
        # The user is focusing their BCI selection on this specific target: "{selected_target}". 
        
        # Task: Based on common sense, infer the most logical assistive action the user wants the robot to perform with the selected target, considering the other objects available.
        # For example:
        # - If there is a "bowl" and the user selects an object like "cherry", they likely want to put the cherry in the bowl.
        # - If there is a "cup" and the user selects "bottle", they likely want to pour water into the cup.
        # - If they just select an object with no obvious container, they might just want to pick it up.
        # - if the user selects a cloth, just fold it.
        
        # Output requirement: Return ONLY a short, direct English command string suitable for a Vision-Language-Action (VLA) robot model. Do not include any explanations, reasoning, or quotation marks.
        # """
        prompt = f"""
        # You are an intelligent robotic assistant reasoning about human intent. 
        # Context: A paralyzed user is controlling a robot arm via a Brain-Computer Interface. 
        
        # The camera detects the following objects on the table: {all_detected_objects}. 
        # The user is focusing their BCI selection on this specific target: "{selected_target}". 
        Task: Based on common sense, infer the most logical assistive action the user wants the robot to perform with the selected target, considering the other objects available.
        Return ONLY a short, direct English command string suitable for a Vision-Language-Action (VLA) robot model , choose form the following examples:        
        # pick up the starfruit and put it into the bowl.
        # Pick up the bunch of grapes and put it into the bowl.
        # pick up the cherry and put it into the bowl.
        # fold the cloth.
 
        """  
        response = model.generate_content(prompt)
        inferred_command = response.text.strip().strip('"').strip("'")
        
        print(f"💡 [LLM Intent Inference] Inference complete! Generated VLA instruction: -> {inferred_command} <-")
        return inferred_command
        
    except Exception as e:
        print(f"❌ [LLM Intent Inference] Gemini call failed: {e}")
        # Fallback strategy (network failure, etc.)
        fallback_command = f"pick up the {selected_target}."
        print(f"⚠️ [LLM Intent Inference] Triggering fallback strategy: -> {fallback_command} <-")
        return fallback_command
# ======================================================================


# ================== 🌟 BCI Interface Generation & Intent Selection Logic ==================

def get_bci_ui_from_server(main_cam_frame):
    """
    Upload main camera image to the 4090 server, wait for scene parsing,
    then fetch labels and masks for BCI UI display.
    """
    print("📡 Uploading image to 4090 server for scene parsing...")

    # JPEG encoding
    success, img_encoded = cv2.imencode(".jpg", main_cam_frame)
    if not success:
        print("❌ Image encoding failed")
        return [], []

    files = {
        "image_file": (
            "main_cam.jpg",
            img_encoded.tobytes(),
            "image/jpeg",
        )
    }

    # Consistent with server_api.py
    # data = {
    #     "mode": "od",
    #     "prompt": "",
    #     "threshold": "0.5"
    # }
    data = {
        "mode": "cloth",
        # "prompt": "starfruit, bowl, grapes, cherry ",
        "prompt": "cloth ",
        "threshold": "0.5"
    }

    try:
        # ==========================
        # Step 1: Upload image
        # ==========================
        response = requests.post(
            SAM2_SERVER_URL,
            files=files,
            data=data,
            timeout=30
        )

        if response.status_code != 200:
            print(f"❌ update_scene request failed: {response.status_code}")
            return [], []

        result = response.json()
        if result.get("status") != "success":
            print("❌ update_scene returned failure")
            return [], []

        # ==========================
        # Step 2: Poll for parsing result (blocking until completion)
        # ==========================
        scene_url = f"http://{SERVER_HOST}:8000/get_bci_scene"
        
        print("⏳ Waiting for 4090 server to process scene...")
        
        retry_count = 0
        while retry_count < 10:  # Wait up to 10 seconds
            resp = requests.get(scene_url, timeout=5)
            if resp.status_code == 200:
                scene = resp.json()
                if scene.get("status") == "success":
                    labels = scene.get("labels", [])
                    if len(labels) > 0:
                        masks = []
                        for b64 in scene.get("masks", []):
                            img_data = base64.b64decode(b64)
                            nparr = np.frombuffer(img_data, np.uint8)
                            mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                            masks.append(mask)
                        print(f"✅ Successfully retrieved {len(labels)} objects: {labels}")
                        return labels, masks
                    else:
                        print("⚠️ Server returned success, but no objects were detected in the scene")
                        return [], []
                else:
                    # Not ready yet, wait before querying again
                    time.sleep(1.0)
                    retry_count += 1
                    continue
            else:
                print(f"❌ get_bci_scene request failed: {resp.status_code}")
                return [], []

        print("❌ Scene data retrieval timed out...")
        return [], []

    except Exception as e:
        print(f"❌ Communication with Vision Server failed: {e}")
        return [], []


def pad_to_fullscreen(frame, screen_w=1920, screen_h=1080):
    """Resize image and center on black canvas"""
    h, w = frame.shape[:2]
    scale = min(screen_w / w, screen_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    x_offset = (screen_w - new_w) // 2
    y_offset = (screen_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas, scale, x_offset, y_offset, new_w, new_h

def bci_selection_phase(cam_main):
    """
    Stage 1:
        1. Loop uploading current scene until objects are detected (prevent premature execution)
        2. Wait for Windows BCI to finish selection
        3. Get selected_target
        4. Infer task via Gemini
    """
    global TASK_INSTRUCTION

    print("\n==============================")
    print("🧠 Stage 1 : BCI Selection")
    print("==============================")

    labels = []
    
    # ==========================
    # 🌟 Fix 1: Loop until targets are found, keep robotic arms strictly stationary!
    # ==========================
    while True:
        time.sleep(1.0) # Wait for camera to stabilize
        raw_frame = cam_main.get_frame()
        
        if raw_frame is None:
            print("❌ Main camera frame is empty, retrying...")
            continue
            
        labels, _ = get_bci_ui_from_server(raw_frame)
        
        if len(labels) > 0:
            print("✅ Scene uploaded successfully and objects detected!")
            break
        else:
            print("⚠️ No valid target detected, arm locked, retrying in 1s...")
            time.sleep(1.0)

    # ==========================
    # Step 2: Wait for Windows BCI result
    # ==========================
    print("\n⌛ Waiting for Windows BCI decoding result...")
    result_url = f"http://{SERVER_HOST}:8000/get_bci_result"
    selected_target = None

    while True:
        try:
            # 🌟 Fix 2: Add timeout and reduce request rate to prevent overloading server
            response = requests.get(result_url, timeout=5)
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success" and result.get("selected_target"):
                    selected_target = result["selected_target"]
                    break
        except Exception as e:
            print(f"⚠️ Waiting for BCI result... ({e})")
        time.sleep(1.0) 

    print(f"🧠 BCI Selection result: {selected_target}")

    # ==========================
    # Step 3: Gemini inference (supports USE_LLM ablation)
    # ==========================
    TASK_INSTRUCTION = infer_intent_with_gemini(
        all_detected_objects=labels,
        selected_target=selected_target,
    )

    print("\n==============================")
    print("✨ Final VLA Instruction:")
    print(TASK_INSTRUCTION)
    print("==============================\n")

# ================= Main Program =================

def main():
    global latest_action_chunk, new_chunk_available, absolute_step_counter

    piper_left = setup_piper("can_left")
    piper_right = setup_piper("can_right")
    
    print("Connecting cameras (multi-threaded)...")
    cam_main = CameraThread("/dev/v4l/by-path/pci-0000:00:14.0-usb-0:3.2:1.0-video-index0", "main")
    cam_wrist_left = CameraThread("/dev/v4l/by-path/pci-0000:00:14.0-usb-0:5.2:1.0-video-index0", "left")
    cam_wrist_right = CameraThread("/dev/v4l/by-path/pci-0000:00:14.0-usb-0:4.2:1.0-video-index0", "right")

    # 🌟=== Core insertion: Execute BCI selection before starting arms ===
    bci_selection_phase(cam_main)
    
    if not TASK_INSTRUCTION:
        print("No valid control instruction generated, shutting down.")
        cam_main.stop()
        cam_wrist_left.stop()
        cam_wrist_right.stop()
        return
    # ==========================================
    
    if piper_left and piper_right:
        print("Waiting for controller to be ready...")
        time.sleep(2.0)
        print(">>> Enabling robotic arms <<<")
        for _ in range(3):
            if piper_left:
                piper_left.EnableArm(motor_num=7, enable_flag=0x02)
                piper_left.GripperCtrl(0, 300, 0x01)
            if piper_right:
                piper_right.EnableArm(motor_num=7, enable_flag=0x02)
                piper_right.GripperCtrl(0, 300, 0x01)
            time.sleep(0.2)
        piper_left.ModeCtrl(ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=30, is_mit_mode=0x00)
        piper_right.ModeCtrl(ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=30, is_mit_mode=0x00)
        print(">>> Moving to initial home pose <<<")
        home_pose = np.array([0, 0.0043, 0.0345, -0.0535, 0.0, -0.0020]) 
        start_l = get_current_joints_rad(piper_left)
        start_r = get_current_joints_rad(piper_right)
        
        # 🌟 Smooth addition: Wait for valid physical feedback to avoid zero-joint crash or jerk
        retry_count = 0
        while (np.all(start_l == 0) or np.all(start_r == 0)) and retry_count < 20:
            print("⚠️ No valid feedback from arms yet, retrying...")
            time.sleep(0.2)
            start_l = get_current_joints_rad(piper_left)
            start_r = get_current_joints_rad(piper_right)
            retry_count += 1
        
        for i in range(60):
            alpha = (i + 1) / 60
            curr_l = start_l * (1 - alpha) + home_pose * alpha
            curr_r = start_r * (1 - alpha) + home_pose * alpha
            if piper_left: piper_left.JointCtrl(*deg_to_sdk_unit(curr_l))
            if piper_right: piper_right.JointCtrl(*deg_to_sdk_unit(curr_r))
            piper_left.GripperCtrl(70000, 3000, 0x01)
            piper_right.GripperCtrl(70000, 3000, 0x01)
            time.sleep(0.05)
        print("✅ Home pose reached")
        time.sleep(1.0)
    else:
        print("❌ Failed to connect to robotic arms")
        return

    print(f"Connecting to policy server: {SERVER_HOST}:{SERVER_PORT}")
    try:
        client = websocket_client_policy.WebsocketClientPolicy(host=SERVER_HOST, port=SERVER_PORT)
    except Exception as e:
        print(f"❌ Failed to connect to server: {e}")
        return

    print("Starting async inference thread...")
    inference_worker = InferenceThread(client, cam_main, cam_wrist_left, cam_wrist_right, piper_left, piper_right)
    inference_worker.start()

    inference_log =[]
    
    print("🔄 Triggering first capture, waiting for initial model output...")
    infer_trigger_event.set()
    
    while True:
        with latest_chunk_lock:
            if latest_action_chunk is not None:
                break
        time.sleep(0.1)
    
    print("▶️ Starting continuous action execution...")

    # Smoothing start initialization
    smoothed_target_l = get_current_joints_rad(piper_left)
    smoothed_target_r = get_current_joints_rad(piper_right)
    smoothed_grip_l = get_gripper_value(piper_left) / 70.0
    smoothed_grip_r = get_gripper_value(piper_right) / 70.0

    # State tracking for gripper deadband filtering
    last_sent_grip_l = smoothed_grip_l
    last_sent_grip_r = smoothed_grip_r

    current_chunk = None
    step_idx = ACTION_HORIZON 
    
    control_period_ms = 0.0  # 🌟 Fix: Assign initial value to local variable to prevent first-frame log error

    # 🌟 Smooth addition: Historical state variables for trajectory blending
    old_chunk = None
    old_step_idx = 0
    blend_counter = 0

    try:
        while True:
            loop_start = time.perf_counter()

            # --- 1. Check if new chunk is available ---
            with latest_chunk_lock:
                if new_chunk_available and step_idx >= ACTION_HORIZON:
                    # 🌟 Smooth addition: Record previous action chunk before replacement
                    old_chunk = copy.deepcopy(current_chunk) if current_chunk is not None else None
                    old_step_idx = step_idx
                    
                    current_chunk = copy.deepcopy(latest_action_chunk)
                    new_chunk_available = False
                    
                    elapsed_steps = absolute_step_counter - latest_chunk_capture_step
                    # 🌟 Smooth addition: Introduce LATENCY_STEPS compensation to avoid backtracking
                    step_idx = max(0, elapsed_steps + LATENCY_STEPS)  
                    
                    # 🌟 Smooth addition: Start smooth blend counter
                    blend_counter = BLEND_STEPS  
                    
                    print(f"\n⚡ Smooth transition done! Total physical step: {absolute_step_counter}, skipped first {step_idx} frames of new chunk, blending enabled.")

            # --- 2. Continuous action execution ---
            if current_chunk is not None:
                if step_idx == ACTION_HORIZON - PREFETCH_STEPS or \
                   (step_idx >= ACTION_HORIZON and not new_chunk_available and step_idx % 5 == 0):
                    print(f"   [Sync] Triggering image capture... (Physical frame: {absolute_step_counter})")
                    infer_trigger_event.set()

                safe_idx = min(step_idx, current_chunk.shape[0] - 1)
                action_step = current_chunk[safe_idx]
                
                target_l, target_r, grip_l, grip_r = map_actions_to_joint_targets_dual(
                    action_step, smoothed_target_l, smoothed_target_r
                )

                # ================= 🌟 Smooth addition: Trajectory blending (joints only, not grippers) =================
                if blend_counter > 0 and old_chunk is not None:
                    safe_old_idx = min(old_step_idx, old_chunk.shape[0] - 1)
                    old_action_step = old_chunk[safe_old_idx]
                    
                    # Extract joint targets from old action, ignore old gripper values
                    old_t_l, old_t_r, _, _ = map_actions_to_joint_targets_dual(
                        old_action_step, smoothed_target_l, smoothed_target_r
                    )
                    
                    w = 1.0 - (blend_counter / float(BLEND_STEPS))
                    
                    # Smooth joint displacement only
                    target_l = old_t_l * (1 - w) + target_l * w
                    target_r = old_t_r * (1 - w) + target_r * w
                    
                    # Note: Never alter grip_l/grip_r of new chunk, ensuring fast and firm grasp!
                    
                    old_step_idx += 1
                    blend_counter -= 1

                # ================= a. Joint and Gripper EMA Smoothing =================
                smoothed_target_l = ACTION_EMA_ALPHA * target_l + (1 - ACTION_EMA_ALPHA) * smoothed_target_l
                smoothed_target_r = ACTION_EMA_ALPHA * target_r + (1 - ACTION_EMA_ALPHA) * smoothed_target_r
                
                # 🌟 Smooth addition: Reduce gripper smoothing for faster response (set to 0.7 to avoid loose grasp)
                GRIPPER_EMA_ALPHA = 0.7
                smoothed_grip_l = GRIPPER_EMA_ALPHA * grip_l + (1 - GRIPPER_EMA_ALPHA) * smoothed_grip_l
                smoothed_grip_r = GRIPPER_EMA_ALPHA * grip_r + (1 - GRIPPER_EMA_ALPHA) * smoothed_grip_r

                # ================= b. Gripper Deadband Filtering =================
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

                # ================= c. Hardware Interpolation Logic =================
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
                
                real_l = get_current_joints_rad(piper_left)
                real_r = get_current_joints_rad(piper_right)

                real_gl = get_gripper_value(piper_left) / 70.0
                real_gr = get_gripper_value(piper_right) / 70.0

                err_l = smoothed_target_l - real_l
                err_r = smoothed_target_r - real_r
                
                # 🌟 Preserve original CSV logging format without dimension changes
                inference_log.append([

                    time.time(),
                    absolute_step_counter,

                    # ---------------- Raw Prediction ----------------
                    *target_l,
                    grip_l,

                    *target_r,
                    grip_r,

                    # ---------------- EMA ----------------
                    *smoothed_target_l,
                    smoothed_grip_l,

                    *smoothed_target_r,
                    smoothed_grip_r,

                    # ---------------- Real ----------------
                    *real_l,
                    real_gl,

                    *real_r,
                    real_gr,

                    # ---------------- Error ----------------
                    *err_l,
                    *err_r,

                    latest_infer_latency_ms,

                    control_period_ms

                ])
                if step_idx < ACTION_HORIZON:
                    print(f"--- Step {step_idx+1}/{ACTION_HORIZON} (Total steps {absolute_step_counter}) ---")
                
                step_idx += 1
                absolute_step_counter += 1

            elapsed_loop = time.perf_counter() - loop_start
            control_period_ms = elapsed_loop * 1000.0
            if elapsed_loop < DT:
                time.sleep(DT - elapsed_loop)

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    finally:
        print("Shutting down...")
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