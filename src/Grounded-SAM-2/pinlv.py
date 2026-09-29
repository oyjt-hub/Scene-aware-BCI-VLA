import torch
import torchvision.models.detection as detection
import torchvision.transforms as T
import cv2
import numpy as np
import os

OUTPUT_DIR = "./cv_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 0. 自定义输入参数 (专门为你论文定制) =================
image_path = "/home/oyjt/Grounded-SAM-2/pic/task3_init.jpg"

# 1. 在这里输入你想要的 Phrase 提示词！(系统会按画面中物体从左到右的顺序赋予)
PHRASE_PROMPTS = ["grapes", "bowl", "cherry" ]

# 2. 定义对应的频率与超粗线条样式 (针对小图排版优化)
freq_configs = [
    {"hz": 8,  "spacing": 45, "thickness": 8, "color": (50, 205, 50)},    # 绿色，疏粗线
    {"hz": 10, "spacing": 35, "thickness": 8, "color": (0, 165, 255)},    # 橙色，中粗线
    {"hz": 12, "spacing": 20, "thickness": 8, "color": (255, 50, 50)},    # 蓝色，密粗线
]

# ================= 1. 读取和预处理图片 =================
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    print("⚠️ 未找到原图，生成一张测试图...")
    img_bgr = np.full((600, 800, 3), (240, 240, 240), dtype=np.uint8)
    cv2.circle(img_bgr, (200, 300), 70, (50, 50, 200), -1)   # 左边物体
    cv2.circle(img_bgr, (400, 300), 80, (200, 150, 50), -1)  # 中间物体
    cv2.circle(img_bgr, (650, 300), 60, (50, 200, 50), -1)   # 右边物体

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ================= 2. 提取完美 Mask (使用 Mask R-CNN) =================
print("⏳ 正在提取物体掩膜...")
weights = detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = detection.maskrcnn_resnet50_fpn(weights=weights)
model.eval()

transform = T.Compose([T.ToTensor()])
input_tensor = transform(img_rgb).unsqueeze(0)

with torch.no_grad():
    predictions = model(input_tensor)[0]

# 提取置信度高的物体
score_threshold = 0.5
mask_indices = predictions['scores'] > score_threshold
boxes = predictions['boxes'][mask_indices].cpu().numpy()
masks = predictions['masks'][mask_indices].squeeze(1).cpu().numpy()

# ================= 3. 为图表可视化排序并绑定 Phrase =================
# 获取画面中最显著的 N 个物体 (N = 你的提示词数量)
num_targets = min(len(PHRASE_PROMPTS), len(boxes))
top_boxes = boxes[:num_targets]
top_masks = masks[:num_targets]

# 按中心点 X 坐标从左到右排序，确保 Prompt 能准确对上图中的物体
centers_x = [(box[0] + box[2]) / 2 for box in top_boxes]
sorted_indices = np.argsort(centers_x)

output_img = img_bgr.copy()
h, w = img_bgr.shape[:2]

# 为了让带颜色的超粗斜线更加凸显，把原图稍微调暗调灰一点点 (非常高级的学术图表技巧)
output_img = cv2.addWeighted(output_img, 0.6, np.zeros_like(output_img), 0.4, 0)

print(f"🎯 成功匹配 {num_targets} 个 Phrase 提示词，正在绘制超粗频率映射...")

# ================= 4. 绘制物体框、粗斜线与标签 =================
for idx, original_idx in enumerate(sorted_indices):
    box = top_boxes[original_idx].astype(int)
    mask = top_masks[original_idx] > 0.5
    
    phrase_name = PHRASE_PROMPTS[idx]
    config = freq_configs[idx]
    
    # 4.1 绘制包围盒 (粗边框)
    cv2.rectangle(output_img, (box[0], box[1]), (box[2], box[3]), config["color"], 4)
    
    # 4.2 绘制超粗代表频率的斜线 (Diagonal Hatching)
    line_mask = np.zeros((h, w, 3), dtype=np.uint8)
    spacing = config["spacing"]
    thickness = config["thickness"]
    
    # 画全图斜线 (加入 LINE_AA 抗锯齿，小图也能保持平滑)
    for c in range(-w, h, spacing):
        pt1 = (0, c)
        pt2 = (w, w + c)
        cv2.line(line_mask, pt1, pt2, config["color"], thickness, cv2.LINE_AA)
        
    # 4.3 仅在物体 Mask 内部保留斜线
    for c in range(3):
        line_mask[:, :, c] = np.where(mask, line_mask[:, :, c], 0)
        
    # 叠加到原图 (使用 1.0 的透明度，让颜色极其鲜艳醒目)
    line_pixels = np.any(line_mask > 0, axis=-1)
    output_img[line_pixels] = cv2.addWeighted(output_img, 0.0, line_mask, 1.0, 0)[line_pixels]
    
    # 4.4 绘制醒目的字号更大的标签文字
    label_text = f"[{phrase_name}] f={config['hz']}Hz"
    # 用稍微大一点的字体 (scale=0.9, thickness=2)
    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    
    # 标签背景框
    cv2.rectangle(output_img, (box[0], box[1] - text_h - 15), (box[0] + text_w, box[1]), config["color"], -1)
    # 白色高对比度文字 (使用 LINE_AA 抗锯齿)
    cv2.putText(output_img, label_text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

# ================= 5. 保存结果 =================
final_path = f"{OUTPUT_DIR}/5_Dynamic_Frequency_Mapping_Bold.jpg"
cv2.imwrite(final_path, output_img)
print(f"🎉 大功告成！支持自定义 Phrase 和超粗线条的特化图已保存至: {final_path}")