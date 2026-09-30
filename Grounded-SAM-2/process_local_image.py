import os
import json
import base64
import numpy as np
import cv2
from PIL import Image
import time

# 导入你本地的模型
import grounded_sam2_florence2_image_demo as demo

# ================= 配置输入输出与模式 =================
INPUT_IMAGE_PATH = "/home/oyjt/Grounded-SAM-2/pic/table.png"  # 替换成你实际的图片路径 (比如 table.png)
OUTPUT_JSON_PATH = "/home/oyjt/Grounded-SAM-2/outputs/scene_data.json"  # 输出包含精确 Mask 的 JSON

# 🌟 在这里控制你要使用的模式
MODE = "text"             # 可选: "od" (检测所有) 或 "text" (根据提示词检测)
PROMPT = "rack, wine bottle, stove, drawer, cream cheese, bowl, plate"         # 如果是 "text" 模式，这里填你要寻找的物体，比如 "drawer"
THRESHOLD = 0.4           # 置信度阈值
# ======================================================

def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(b1_area + b2_area - inter_area + 1e-6)

def generate_scene_data(image_path, mode="od", prompt="", threshold=0.4):
    print(f"📷 正在处理图片: {image_path}")
    image = Image.open(image_path).convert("RGB")
    
    # 1. 设置 Florence-2 模式
    if mode == "od":
        print(f"📡 启动 <OD> 全局目标检测 (阈值: {threshold})...")
        task_prompt = "<OD>"
        text_input = None  
    else:
        print(f"📡 启动 <CAPTION_TO_PHRASE_GROUNDING> (提示词: '{prompt}', 阈值: {threshold})...")
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        text_input = prompt

    results = demo.run_florence2(
        task_prompt=task_prompt, 
        text_input=text_input, 
        model=demo.florence2_model, 
        processor=demo.florence2_processor, 
        image=image
    )
    
    results = results[task_prompt]
    input_boxes = np.array(results.get("bboxes", []))
    class_names = results.get("labels", [])

    if len(input_boxes) == 0:
        print("⚠️ 未检测到任何目标，请尝试更换 prompt 或降低阈值。")
        return
    
    print(f"✅ 检测到 {len(class_names)} 个初步目标: {class_names}")

    # 2. 启动 SAM-2 提取精确 Mask
    print("🎭 启动 SAM-2 提取精确边缘 (Mask)...")
    demo.sam2_predictor.set_image(np.array(image))
    masks, scores, _ = demo.sam2_predictor.predict(
        point_coords=None, point_labels=None, box=input_boxes, multimask_output=False
    )

    # 3. 过滤超大目标和低置信度
    img_area = image.width * image.height
    MAX_AREA_RATIO = 0.85  
    valid_indices = []
    
    for i, score in enumerate(scores):
        box = input_boxes[i]
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        area_ratio = box_area / img_area
        
        if score > threshold and area_ratio <= MAX_AREA_RATIO:
            valid_indices.append(i)
        elif area_ratio > MAX_AREA_RATIO:
            print(f"⚠️ 过滤超大目标: '{class_names[i]}' (占比 {area_ratio*100:.1f}%)")

    if not valid_indices:
        print("⚠️ 检测到的目标都被过滤掉了 (可能占比过大或置信度过低)。")
        return

    masks = masks[valid_indices]
    input_boxes = input_boxes[valid_indices]
    class_names = [class_names[i] for i in valid_indices]
    scores = scores[valid_indices] 

    # 4. NMS 去重过滤
    sorted_indices = np.argsort(np.array(scores).flatten())[::-1]
    keep_indices = []
    for idx in sorted_indices:
        idx = int(idx)  
        if not any(compute_iou(input_boxes[idx], input_boxes[keep_idx]) > 0.85 for keep_idx in keep_indices):
            keep_indices.append(idx)

    final_masks = masks[keep_indices]
    final_boxes = input_boxes[keep_indices].tolist()
    final_labels = [class_names[i] for i in keep_indices]

    # 5. 将精确边缘编码为 Base64 以供 Pygame 解析
    print("🔄 正在将精确边缘编码为 Base64 格式...")
    mask_b64s = []
    for mask in final_masks:
        mask_np = np.array(mask)
        while mask_np.ndim > 2:
            mask_np = mask_np[0]
        binary_mask = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.uint8)
        binary_mask[mask_np > 0] = 255
        success, buffer = cv2.imencode('.png', binary_mask)
        mask_b64s.append(base64.b64encode(buffer).decode('utf-8'))

    # 构建带 Mask 的 JSON 结构
    scene_data = {
        "labels": final_labels,
        "bboxes": final_boxes,
        "masks": mask_b64s  # 🌟 这里带上了图片形状信息
    }
    
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(scene_data, f)
        
    print(f"✅ 处理完成！共提取并保存了 {len(final_labels)} 个目标。")
    print(f"📁 完美包含掩码数据的 JSON 已保存至: {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    generate_scene_data(INPUT_IMAGE_PATH, mode=MODE, prompt=PROMPT, threshold=THRESHOLD)