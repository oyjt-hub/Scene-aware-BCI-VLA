import time
import numpy as np
import csv
import sys
import select
import termios
import tty
from typing import Any

# --- 导入 SDK ---
try:
    from piper_sdk import C_PiperInterface_V2
except ImportError:
    C_PiperInterface_V2 = None
    print("❌ 错误: 未找到 piper_sdk，请检查环境配置。")
    exit(1)

# ================= 优化后的键盘映射配置 =================
# 格式: '按键': (关节索引, 运动方向)
JOINT_MAP = {
    # 左手：大臂控制 (决定空间位置)
    'a': (0, 1.0),  'd': (0, -1.0),  # 关节1：底座 左转/右转
    'w': (1, 1.0),  's': (1, -1.0),  # 关节2：肩部 抬起/压低
    'q': (2, 1.0),  'e': (2, -1.0),  # 关节3：肘部 伸展/收缩
    
    # 右手：手腕控制 (决定末端姿态)
    'j': (3, 1.0),  'l': (3, -1.0),  # 关节4：手腕 左偏/右偏
    'i': (4, 1.0),  'k': (4, -1.0),  # 关节5：手腕 上翻/下翻
    'u': (5, 1.0),  'o': (5, -1.0),  # 关节6：手腕 左旋/右旋
}

GRIP_OPEN_KEY = 'g'       # 夹爪张开
GRIP_CLOSE_KEY = 'f'      # 夹爪闭合
TOGGLE_ARM_KEY = '\t'     # Tab键：切换左右手臂
SUCCESS_KEY = ' '         # 空格键：完成任务并记录时间

# 角度单步变化量 (度数转弧度，约等于 3.0 度)
STEP_RAD = np.deg2rad(3.0) 

# ================= 初始位姿配置 =================
# 定义左右臂的初始关节角度 (单位: 度) [关节1, 关节2, 关节3, 关节4, 关节5, 关节6]
INITIAL_POSE_L_DEG = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
INITIAL_POSE_R_DEG = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# 初始夹爪状态 (0: 完全闭合, 70000: 完全张开)
INITIAL_GRIPPER = 70000 
# ===============================================

JOINT_LIMITS_RAD = np.array([
    [-2.61799, 2.61799], [0.0, 3.14159], [-2.96706, 0.0],
    [-1.74533, 1.74533], [-1.22173, 1.22173], [-2.0944, 2.0944]
])

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
    """平滑移动到初始位姿"""
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
    
    print(">>> 已到达初始位姿！ <<<")
    return target_l_rad, target_r_rad

def main():
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

    # 移动到初始位姿
    target_l, target_r = move_to_initial_pose(piper_left, piper_right)
    grip_l = INITIAL_GRIPPER if piper_left else 70000
    grip_r = INITIAL_GRIPPER if piper_right else 70000

    active_arm = "left"
    
    print("\n" + "="*60)
    print("🎮 [纯终端键盘控制模式 - 双手分区版] 启动成功！")
    print("【左手区 - 大臂空间移动】")
    print("  - [A] / [D] : 底座 左转 / 右转")
    print("  - [W] / [S] : 肩部 抬起 / 压低")
    print("  - [Q] / [E] : 肘部 伸展 / 收缩")
    print("【右手区 - 手腕姿态微调】")
    print("  - [J] / [L] : 手腕 左偏 / 右偏")
    print("  - [I] / [K] : 手腕 上翻 / 下翻")
    print("  - [U] / [O] : 手腕 左旋 / 右旋")
    print("【功能键】")
    print("  - [F] / [G] : 夹爪 闭合 / 张开")
    print("  - [Tab键]   : 切换控制 左手臂 / 右手臂")
    print("  - [空格键]  : 成功完成任务并保存时间")
    print("  - [ESC]     : 放弃任务并退出")
    print("="*60 + "\n")

    # 注意：这里需要先按一下回车键，程序才会往下走！
    input("⚠️ 准备好了吗？在桌上摆好物品，按下【回车键】开始实验并启动秒表！")
    start_time = time.time()
    print(f"\n✅ 计时开始！当前控制: {active_arm.upper()} 手臂 (直接在终端按键即可)")

    # --- 配置终端为非阻塞读取模式 ---
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    
    try:
        tty.setraw(fd) # 进入终端 raw 模式，按键无需回车直接捕获
        while True:
            # 每 0.03 秒检测一次是否有按键按下
            rlist, _, _ = select.select([sys.stdin], [], [], 0.03)
            if not rlist:
                continue
                
            key_char = sys.stdin.read(1).lower() # 读取按键并统一转为小写
            
            # 处理退出逻辑 (ESC 键 或 Ctrl+C)
            if key_char == '\x1b' or key_char == '\x03':  
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                print("\n🛑 用户取消，放弃记录数据。")
                break
                
            # 处理完成逻辑 (空格键)
            elif key_char == SUCCESS_KEY:  
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
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
                break
                
            # 处理切换手臂 (Tab键)
            elif key_char == TOGGLE_ARM_KEY:
                active_arm = "right" if active_arm == "left" else "left"
                # 在 raw 模式下打印需要加 \r 回车符
                sys.stdout.write(f"\r🔄 切换至控制: {active_arm.upper()} 手臂          \n\r")
                sys.stdout.flush()
                continue

            # 处理关节控制
            elif key_char in JOINT_MAP:
                joint_idx, direction = JOINT_MAP[key_char]
                
                if active_arm == "left" and piper_left:
                    target_l[joint_idx] = np.clip(target_l[joint_idx] + direction * STEP_RAD, 
                                                  JOINT_LIMITS_RAD[joint_idx, 0], JOINT_LIMITS_RAD[joint_idx, 1])
                    piper_left.JointCtrl(*deg_to_sdk_unit(target_l))
                elif active_arm == "right" and piper_right:
                    target_r[joint_idx] = np.clip(target_r[joint_idx] + direction * STEP_RAD, 
                                                  JOINT_LIMITS_RAD[joint_idx, 0], JOINT_LIMITS_RAD[joint_idx, 1])
                    piper_right.JointCtrl(*deg_to_sdk_unit(target_r))

            # 处理夹爪控制
            elif key_char == GRIP_OPEN_KEY:
                if active_arm == "left" and piper_left:
                    grip_l = min(grip_l + 10000, 70000)
                    piper_left.GripperCtrl(grip_l, 3000, 0x01)
                elif active_arm == "right" and piper_right:
                    grip_r = min(grip_r + 10000, 70000)
                    piper_right.GripperCtrl(grip_r, 3000, 0x01)
            elif key_char == GRIP_CLOSE_KEY:
                if active_arm == "left" and piper_left:
                    grip_l = max(grip_l - 10000, 0)
                    piper_left.GripperCtrl(grip_l, 3000, 0x01)
                elif active_arm == "right" and piper_right:
                    grip_r = max(grip_r - 10000, 0)
                    piper_right.GripperCtrl(grip_r, 3000, 0x01)

    finally:
        # 确保程序退出时恢复终端正常状态，防止终端卡死
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        
        if piper_left: 
            try: piper_left.DisconnectPort()
            except: pass
        if piper_right: 
            try: piper_right.DisconnectPort()
            except: pass

if __name__ == "__main__":
    main()