import time
import numpy as np
import csv
import sys
import os
from typing import Any

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

try:
    from piper_sdk import C_PiperInterface_V2
except ImportError:
    C_PiperInterface_V2 = None
    print("❌ 错误: 未找到 piper_sdk，请检查环境配置。")
    exit(1)

# ================= 手柄按键映射配置 =================
BTN_A = 0          # 切换手臂
BTN_RB = 7         # 结束计时并保存数据
BTN_LT = 8         # 夹爪闭合
BTN_RT = 9         # 夹爪张开
# ==================================================

INITIAL_POSE_L_DEG = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
INITIAL_POSE_R_DEG = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
INITIAL_GRIPPER = 35000 

JOINT_LIMITS_RAD = np.array([
    [-2.61799, 2.61799], [0.0, 3.14159], [-2.96706, 0.0],
    [-1.74533, 1.74533], [-1.22173, 1.22173], [-2.0944, 2.0944]
])

# ================= 🌟 灵敏度调优核心区 🌟 =================
MAX_STEP_RAD = np.deg2rad(0.6) 
DPAD_STEP_RAD = np.deg2rad(0.4) 
GRIP_STEP = 1000                
DEADZONE = 0.15 

def apply_deadzone_and_curve(value: float) -> float:
    """应用死区，并引入二次方非线性曲线 (让微操更精准)"""
    if abs(value) < DEADZONE: 
        return 0.0
    mapped_val = (abs(value) - DEADZONE) / (1.0 - DEADZONE)
    curve_val = mapped_val ** 2.0 
    return curve_val * np.sign(value)
# =========================================================

def clamp_rad(rad_targets: np.ndarray) -> np.ndarray:
    return np.clip(rad_targets, JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])

def deg_to_sdk_unit(rad: np.ndarray) -> np.ndarray:
    deg = np.rad2deg(rad)
    return (deg * 1000.0).astype(np.int32)

def get_current_joints_rad(piper: Any) -> np.ndarray:
    if piper is None: return np.zeros(6)
    try:
        js_ret = piper.GetArmJointMsgs()
        js = js_ret[2] if isinstance(js_ret, tuple) and len(js_ret) >= 3 else js_ret
        if hasattr(js, "joint_state") and js.joint_state:
            st = js.joint_state
            for names in [["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
                          ["angle_1", "angle_2", "angle_3", "angle_4", "angle_5", "angle_6"]]:
                try:
                    vals = [getattr(st, n) for n in names]
                    return np.deg2rad(np.array(vals, dtype=np.float64) / 1000.0)
                except AttributeError: continue
        return np.zeros(6)
    except Exception:
        return np.zeros(6)

def setup_piper(can_name: str):
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

def move_to_initial_pose(piper_left, piper_right):
    print(">>> 正在平滑移动到初始位姿... <<<")
    target_l_rad = np.deg2rad(INITIAL_POSE_L_DEG)
    target_r_rad = np.deg2rad(INITIAL_POSE_R_DEG)
    curr_l_rad = get_current_joints_rad(piper_left)
    curr_r_rad = get_current_joints_rad(piper_right)
    
    steps = 50
    delay = 0.04 
    for i in range(1, steps + 1):
        if piper_left:
            interp_l = curr_l_rad + (target_l_rad - curr_l_rad) * (i / steps)
            piper_left.JointCtrl(*deg_to_sdk_unit(interp_l))
        if piper_right:
            interp_r = curr_r_rad + (target_r_rad - curr_r_rad) * (i / steps)
            piper_right.JointCtrl(*deg_to_sdk_unit(interp_r))
        time.sleep(delay)
        
    if piper_left: piper_left.GripperCtrl(INITIAL_GRIPPER, 3000, 0x01)
    if piper_right: piper_right.GripperCtrl(INITIAL_GRIPPER, 3000, 0x01)
    time.sleep(0.5)
    return target_l_rad, target_r_rad

def main():
    pygame.init()
    pygame.joystick.init()
    
    if pygame.joystick.get_count() == 0:
        print("❌ 错误: 未检测到 Xbox 手柄！")
        return
    
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"🎮 成功连接手柄: {joystick.get_name()}")

    piper_left = setup_piper("can_left")
    piper_right = setup_piper("can_right")
    
    if not piper_left and not piper_right:
        print("❌ 错误: 左右机械臂均连接失败。")
        return

    print(">>> 正在使能机械臂 <<<")
    for _ in range(3):
        if piper_left: piper_left.EnableArm(motor_num=7, enable_flag=0x02)
        if piper_right: piper_right.EnableArm(motor_num=7, enable_flag=0x02)
        time.sleep(0.2)

    target_l, target_r = move_to_initial_pose(piper_left, piper_right)
    grip_l = INITIAL_GRIPPER if piper_left else 70000
    grip_r = INITIAL_GRIPPER if piper_right else 70000

    active_arm = "left"
    
    print("\n" + "="*60)
    print("🎮 [Xbox 手柄遥操模式 - 丝滑微操版] 启动！")
    print("  - LT 键 (8) : 夹爪闭合")
    print("  - RT 键 (9) : 夹爪张开")
    print("  - A 键 (0)  : 切换左右臂")
    print("  - RB 键 (7) : ✅ 完成任务并保存时间")
    print("  - 键盘 ESC  : 🛑 放弃任务并退出程序")
    print("="*60 + "\n")

    input("⚠️ 按下【回车键】开始测试！")
    start_time = time.time()

    clock = pygame.time.Clock()
    running = True

    while running:
        clock.tick(50)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print("\n🛑 收到键盘 ESC，安全退出程序。")
                    running = False
                
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == BTN_A:  
                    active_arm = "right" if active_arm == "left" else "left"
                    print(f"\r🔄 切换至控制: {active_arm.upper()} 手臂" + " "*20)
                
                # 💡 新增：按 RB 键结束计时并保存
                elif event.button == BTN_RB:  
                    total_time = time.time() - start_time
                    print(f"\n🎉 任务成功完成！总耗时: {total_time:.2f} 秒")
                    try:
                        with open("manual_baseline_results.csv", "a", newline="") as f:
                            writer = csv.writer(f)
                            if f.tell() == 0:
                                writer.writerow(["Manual_Completion_Time_Seconds"])
                            writer.writerow([f"{total_time:.2f}"])
                        print("💾 数据已成功追加保存至 manual_baseline_results.csv")
                    except Exception as e:
                        print(f"⚠️ 保存数据失败: {e}")
                    running = False

        if not running:
            break

        # --- 读取摇杆模拟量 (使用新的非线性曲线函数) ---
        ax_j1 = apply_deadzone_and_curve(joystick.get_axis(0))       
        ax_j2 = apply_deadzone_and_curve(-joystick.get_axis(1))      
        ax_j4 = apply_deadzone_and_curve(joystick.get_axis(2))       
        ax_j3 = apply_deadzone_and_curve(-joystick.get_axis(3))      

        # --- 读取十字键 ---
        hat = joystick.get_hat(0)
        hat_x, hat_y = hat[0], hat[1] 

        # --- 读取肩键/扳机键 (夹爪控制) ---
        lt_pressed = joystick.get_button(BTN_LT) 
        rt_pressed = joystick.get_button(BTN_RT) 

        # --- 计算目标角度 ---
        delta = np.zeros(6)
        delta[0] = ax_j1 * MAX_STEP_RAD
        delta[1] = ax_j2 * MAX_STEP_RAD
        delta[2] = ax_j3 * MAX_STEP_RAD
        delta[3] = ax_j4 * MAX_STEP_RAD
        delta[4] = hat_y * DPAD_STEP_RAD
        delta[5] = hat_x * DPAD_STEP_RAD

        # --- 发送控制指令 ---
        if active_arm == "left" and piper_left:
            if np.any(delta != 0):
                target_l = clamp_rad(target_l + delta)
                piper_left.JointCtrl(*deg_to_sdk_unit(target_l))
            if lt_pressed:
                grip_l = max(grip_l - GRIP_STEP, 0)
                piper_left.GripperCtrl(grip_l, 3000, 0x01)
            elif rt_pressed:
                grip_l = min(grip_l + GRIP_STEP, 70000)
                piper_left.GripperCtrl(grip_l, 3000, 0x01)

        elif active_arm == "right" and piper_right: 
            if np.any(delta != 0):
                target_r = clamp_rad(target_r + delta)
                piper_right.JointCtrl(*deg_to_sdk_unit(target_r))
            if lt_pressed:
                grip_r = max(grip_r - GRIP_STEP, 0)
                piper_right.GripperCtrl(grip_r, 3000, 0x01)
            elif rt_pressed:
                grip_r = min(grip_r + GRIP_STEP, 70000)
                piper_right.GripperCtrl(grip_r, 3000, 0x01)

    pygame.quit()
    if piper_left: 
        try: piper_left.DisconnectPort()
        except: pass
    if piper_right: 
        try: piper_right.DisconnectPort()
        except: pass

if __name__ == "__main__":
    main()