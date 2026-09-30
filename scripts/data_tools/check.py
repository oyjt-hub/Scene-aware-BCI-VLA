import os
import glob
import pandas as pd
import numpy as np

def analyze_total_durations(folder_path, timestamp_col='timestamp'):
    # 获取文件夹下所有的 csv 文件
    file_pattern = os.path.join(folder_path, "*.csv")
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"未在路径【{folder_path}】下找到任何符合条件的 CSV 文件。")
        return
    
    total_durations = []
    file_details = []
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)
            
            if timestamp_col in df.columns:
                # 用最后一行的时间戳减去第一行的时间戳，计算总持续时间（单位：秒）
                start_time = df[timestamp_col].min()
                end_time = df[timestamp_col].max()
                duration = end_time - start_time
                
                total_durations.append(duration)
                file_details.append({
                    'file_name': file_name,
                    'duration': duration
                })
            else:
                print(f"警告: 文件 {file_name} 中未找到时间戳列 '{timestamp_col}'")
        except Exception as e:
            print(f"读取文件 {file_name} 时出错: {e}")
            
    if not total_durations:
        print("未提取到任何有效的持续时间数据。")
        return
        
    # 计算这十几个 CSV 文件的平均总持续时间及 ± 标准差
    mean_duration = np.mean(total_durations)
    std_duration = np.std(total_durations)
    
    print("\n" + "="*60)
    print("【每个 CSV 文件的总运行持续时间 统计结果】")
    print(f"总文件数: {len(total_durations)} 个")
    print(f"平均总持续时间: {mean_duration:.3f} 秒 ± {std_duration:.3f} 秒")
    print("="*60)
    
    print("\n【各文件详细总时长】")
    for detail in file_details:
        print(f"- {detail['file_name']}: {detail['duration']:.3f} 秒")

# ==================== 使用说明 ====================
# 请将下面的路径修改为您存放这十几个 CSV 文件的实际文件夹路径
target_folder = "./csv"
analyze_total_durations(target_folder, timestamp_col='timestamp')