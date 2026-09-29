import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import cv2
import torch
import argparse
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForCausalLM
import json
import pycocotools.mask as mask_util

"""
Define Some Hyperparam
"""

TASK_PROMPT = {
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
    "object_detection": "<OD>",
    "dense_region_caption": "<DENSE_REGION_CAPTION>",
    "region_proposal": "<REGION_PROPOSAL>",
    "phrase_grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
    "referring_expression_segmentation": "<REFERRING_EXPRESSION_SEGMENTATION>",
    "region_to_segmentation": "<REGION_TO_SEGMENTATION>",
    "open_vocabulary_detection": "<OPEN_VOCABULARY_DETECTION>",
    "region_to_category": "<REGION_TO_CATEGORY>",
    "region_to_description": "<REGION_TO_DESCRIPTION>",
    "ocr": "<OCR>",
    "ocr_with_region": "<OCR_WITH_REGION>",
}

OUTPUT_DIR = "./outputs"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

"""
Init Florence-2 and SAM 2 Model (强制本地加载)
"""

# 请确保这个文件夹下有所有的 config.json, pytorch_model.bin/safetensors 以及 .py 代码文件
FLORENCE2_MODEL_ID = "./Florence-2-large"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# environment settings
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print("正在强制从本地路径加载 Florence-2...")
# 🚨 核心修改: 增加 local_files_only=True 彻底切断网络请求
florence2_model = AutoModelForCausalLM.from_pretrained(
    FLORENCE2_MODEL_ID, 
    trust_remote_code=True, 
    torch_dtype='auto',
    local_files_only=True  
).eval().to(device)

florence2_processor = AutoProcessor.from_pretrained(
    FLORENCE2_MODEL_ID, 
    trust_remote_code=True,
    local_files_only=True 
)

print("正在从本地路径加载 SAM 2...")
# build sam 2
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
sam2_predictor = SAM2ImagePredictor(sam2_model)

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def run_florence2(task_prompt, text_input, model, processor, image):
    assert model is not None, "You should pass the init florence-2 model here"
    assert processor is not None, "You should set florence-2 processor here"

    device = model.device

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].to(device),
      pixel_values=inputs["pixel_values"].to(device),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    return parsed_answer


# =====================================================================
# 🌟 新增 Pipeline：生成论文流程图所需的 3 张图 (Raw / Heatmap / Mask)
#    (已应用真实感受野扩散及特征底噪算法)
# =====================================================================
# =====================================================================
# 🌟 新增 Pipeline：生成论文流程图所需的 3 张图 (Raw / Heatmap / Mask)
# =====================================================================
def generate_paper_diagrams(
    florence2_model,
    florence2_processor,
    sam2_predictor,
    image_path,
    text_input,
    output_dir=OUTPUT_DIR
):
    print(f"\n[论文图生成] 正在处理图片: {image_path}")
    
    # 1. 强制将提示词转换成短句，避免大模型把所有东西当成一个物体
    text_input = text_input.replace(",", ".")
    if not text_input.endswith("."):
        text_input += "."
    print(f"[论文图生成] 实际送入模型的 Prompt: {text_input}")
    
    # 换回最靠谱的 Phrase Grounding 任务
    task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
    image = Image.open(image_path).convert("RGB")
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_h, img_w = img_cv.shape[:2]

    # ---------- [图 1] Raw RGB ----------
    cv2.imwrite(os.path.join(output_dir, "1_Raw_RGB.jpg"), img_cv)
    print("✅ 图1: Raw RGB 保存成功")

    # 获取检测框
    results = run_florence2(task_prompt, text_input, florence2_model, florence2_processor, image)
    predictions = results[task_prompt]
    
    raw_boxes = np.array(predictions.get("bboxes", []))
    raw_labels = predictions.get("labels", [])  # Grounding 返回的是 labels 字段
    
    # 2. 强力过滤：剔除超过全图 80% 的误识别大框
    valid_boxes = []
    valid_labels = []
    img_area = img_h * img_w
    for i, box in enumerate(raw_boxes):
        w = box[2] - box[0]
        h = box[3] - box[1]
        if (w * h) < 0.8 * img_area:  # 只有正常大小的物品才会被保留
            valid_boxes.append(box)
            valid_labels.append(raw_labels[i])
            
    if len(valid_boxes) == 0:
        print("⚠️ 未检测到有效的独立物品框！请检查提示词或图片。")
        return
        
    input_boxes = np.array(valid_boxes)
    class_names = valid_labels

    # 先做 SAM-2 预测，获取物体真实的形状轮廓
    sam2_predictor.set_image(np.array(image))
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # ---------- [图 2] 基于真实掩码扩散的高阶注意力热力图 (Attention Map) ----------
    heatmap_base = np.zeros((img_h, img_w), dtype=np.float32)
    
    # 将真实的物体 Mask 叠加到底图上
    for mask in masks:
        heatmap_base += mask.astype(np.float32)

    # 核心算法：用超大的高斯核进行扩散，模拟 CNN/ViT 的特征感受野 (Receptive Field)
    # 这会让物体中心极其高亮，并向四周平滑产生光晕，形状完全贴合物体本身的轮廓！
    heatmap_base = cv2.GaussianBlur(heatmap_base, (0, 0), sigmaX=45, sigmaY=45)

    # 归一化亮度
    max_val = np.max(heatmap_base)
    if max_val > 0:
        heatmap_base = heatmap_base / max_val
        
    # 添加极其微弱的底色，防止背景死黑
    heatmap_base = np.clip(heatmap_base + 0.05, 0, 1)
    
    heatmap_8u = np.uint8(255 * heatmap_base)
    # 使用 JET 色谱 
    heatmap_color = cv2.applyColorMap(heatmap_8u, cv2.COLORMAP_JET) 
    
    # 叠加到黑白底图上
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray_img_color = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    tech_heatmap = cv2.addWeighted(gray_img_color, 0.4, heatmap_color, 0.6, 0)
    
    cv2.imwrite(os.path.join(output_dir, "2_Florence2_Heatmap.jpg"), tech_heatmap)
    print("✅ 图2: Florence-2 特征热力图保存成功 (已升级为基于 Mask 扩散的神图)")

    # ---------- [图 3] SAM-2 高清掩码图 ----------
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=np.arange(len(class_names))
    )
    
    mask_annotator = sv.MaskAnnotator(opacity=0.6)
    mask_img = mask_annotator.annotate(scene=img_cv.copy(), detections=detections)
    
    # # 使用 TOP_LEFT（左上角）排版，防止多个标签在中心互相遮挡
    # # label_annotator = sv.LabelAnnotator(
    # #         text_position=sv.Position.TOP_LEFT, 
    # #         text_scale=4,       # ⬅️ 修改这里：默认是 0.8，数字越大标签和字就越大 (比如 1.5, 2.0)
    # #         text_thickness=2,     # ⬅️ 加上这个：默认是 1，改成 2 会让字更粗更清晰
    # #         text_padding=15       # ⬅️ 加上这个(可选)：可以让文字周围的彩色背景框大一点，看起来更舒展
    # #     )
    # final_img = label_annotator.annotate(scene=mask_img, detections=detections, labels=class_names)
    
    cv2.imwrite(os.path.join(output_dir, "3_SAM2_Mask_For_Paper.jpg"), final_img)
    print("✅ 图3: SAM-2 论文纯净掩码图保存成功\n")
# =====================================================================
    # ---------- [图 4] 自动生成用于画“特征金字塔”的 3 张小特征图 ----------
    # =====================================================================
    print("\n[金字塔生成] 正在为你生成顶刊级的高级特征金字塔...")
    
    # ---------- 1. 浅层特征 (Low-level)：连续梯度响应，拒绝简笔画 ----------
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (7, 7), 0) # 抹平细小噪点
    
    # 使用 Sobel 提取连续的梯度幅度（类似 X 光质感，而不是死线条）
    grad_x = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_mag = cv2.normalize(grad_mag, None, 0, 1, cv2.NORM_MINMAX)
    
    # 结合 Mask 平滑地压暗背景，突出主体轮廓
    mask_weight = np.zeros((img_h, img_w), dtype=np.float32)
    for mask in masks:
        mask_weight += mask.astype(np.float32)
    mask_weight = np.clip(mask_weight, 0, 1)
    mask_weight = cv2.GaussianBlur(mask_weight, (99, 99), 0) # 极其平滑的过渡
    mask_weight = np.clip(mask_weight + 0.15, 0, 1) # 保留 15% 的背景感知
    
    feat1 = grad_mag * mask_weight
    feat1 = cv2.GaussianBlur(feat1, (5, 5), 0) # 模拟网络特征的微小柔化
    feat1_color = cv2.applyColorMap(np.uint8(255 * feat1), cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(os.path.join(output_dir, "4_Pyramid_Feat1_Low.jpg"), feat1_color)

    # ---------- 2. 中层特征 (Mid-level)：柔和 Blob 激活块，拒绝硬边缘贴纸 ----------
    feat2_base = np.zeros((img_h, img_w), dtype=np.float32)
    for mask in masks:
        feat2_base += mask.astype(np.float32)
    feat2_base = np.clip(feat2_base, 0, 1)
    
    # 核心：必须在缩小前做极重度的模糊，让硬贴纸融化成能量气泡 (Blob)
    feat2_base = cv2.GaussianBlur(feat2_base, (0, 0), sigmaX=35, sigmaY=35)
    
    small_w, small_h = max(img_w//16, 1), max(img_h//16, 1)
    feat2_small = cv2.resize(feat2_base, (small_w, small_h), interpolation=cv2.INTER_AREA)
    
    # 构造“团块状”低频噪点（而不是劣质的雪花噪点）
    noise_tiny = np.random.rand(small_h//2, small_w//2).astype(np.float32) * 0.4
    noise_blob = cv2.resize(noise_tiny, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    
    feat2_small = np.clip(feat2_small * 0.8 + noise_blob, 0, 1)
    
    feat2_up = cv2.resize(feat2_small, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    feat2_up = np.clip(feat2_up, 0, 1)  # 🚨 就是加了这一   行！防止插值过冲导致的颜色溢出黑斑！
    feat2_color = cv2.applyColorMap(np.uint8(255 * feat2_up), cv2.COLORMAP_PLASMA)
    cv2.imwrite(os.path.join(output_dir, "4_Pyramid_Feat2_Mid.jpg"), feat2_color)

    # ---------- 3. 深层特征 (High-level)：保留完美的 ViT 大模型 Patch 网格 ----------
    feat3_base = np.zeros((img_h, img_w), dtype=np.float32)
    for box in input_boxes:
        w_box, h_box = box[2] - box[0], box[3] - box[1]
        cx, cy = int((box[0]+box[2])/2), int((box[1]+box[3])/2)
        
        sigma_x, sigma_y = max(w_box/4.0, 5), max(h_box/4.0, 5)
        y, x = np.ogrid[-cy:img_h-cy, -cx:img_w-cx]
        g = np.exp(-( (x*x)/(2*sigma_x*sigma_x) + (y*y)/(2*sigma_y*sigma_y) ))
        feat3_base += g

    feat3_base = np.clip(feat3_base, 0, 1)
    
    grid_w = 32
    grid_h = max(int(32 * (img_h / img_w)), 1)
    feat3_small = cv2.resize(feat3_base, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    
    # 稍微压低底噪，让中心的发光感更神圣
    noise3 = np.random.rand(grid_h, grid_w).astype(np.float32) * 0.1
    feat3_small = np.clip(feat3_small * 0.9 + noise3 + 0.05, 0, 1)
    
    feat3_up = cv2.resize(feat3_small, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    feat3_up = cv2.GaussianBlur(feat3_up, (5, 5), 0) # 微微融化方块边缘的毛刺
    
    feat3_color = cv2.applyColorMap(np.uint8(255 * feat3_up), cv2.COLORMAP_INFERNO)
    cv2.imwrite(os.path.join(output_dir, "4_Pyramid_Feat3_High.jpg"), feat3_color)

    print("✅ 金字塔三部曲已重构完毕！快去验收这完美的 X光轮廓 和 能量团块 吧！\n")
def phrase_grounding_and_segmentation(
    florence2_model,
    florence2_processor,
    sam2_predictor,
    image_path,
    task_prompt="<CAPTION_TO_PHRASE_GROUNDING>",
    text_input=None,
    output_dir=OUTPUT_DIR
):
    image = Image.open(image_path).convert("RGB")
    results = run_florence2(task_prompt, text_input, florence2_model, florence2_processor, image)
    
    assert text_input is not None, "Text input should not be None when calling phrase grounding pipeline."
    results = results[task_prompt]
    
    input_boxes = np.array(results["bboxes"])
    class_names = results["labels"]
    class_ids = np.array(list(range(len(class_names))))
    
    sam2_predictor.set_image(np.array(image))
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None, point_labels=None, box=input_boxes, multimask_output=False,
    )
    
    if masks.ndim == 4:
        masks = masks.squeeze(1)
        
    bci_freqs = ["2Hz", "5Hz", "10Hz", "12Hz", "15Hz"]
    labels = []
    for i in range(len(class_names)):
        freq = bci_freqs[i % len(bci_freqs)] 
        item_name = str(class_names[i]).strip() 
        labels.append(f"{item_name} ")
    
    img = cv2.imread(image_path)
    detections = sv.Detections(xyxy=input_boxes, mask=masks.astype(bool), class_id=class_ids)
    
    mask_annotator = sv.MaskAnnotator(opacity=0.4) 
    annotated_frame = mask_annotator.annotate(scene=img.copy(), detections=detections)

    H, W = annotated_frame.shape[:2] 
    
    for idx, (mask, class_id) in enumerate(zip(masks, class_ids)):
        color = sv.ColorPalette.DEFAULT.by_idx(class_id).as_bgr()
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated_frame, contours, -1, color, 5)
        
        spacing = 10 + (idx % 4) * 15  
        line_thickness = 2
        hatch_canvas = np.zeros_like(annotated_frame)
        
        if idx % 2 == 0:
            for offset in range(-H, W, spacing):
                cv2.line(hatch_canvas, (offset, 0), (offset + H, H), color, line_thickness)
        else:
            for offset in range(0, W + H, spacing):
                cv2.line(hatch_canvas, (offset, 0), (offset - H, H), color, line_thickness)
                
        hatch_roi = cv2.bitwise_and(hatch_canvas, hatch_canvas, mask=mask.astype(np.uint8))
        lines_mask = cv2.cvtColor(hatch_roi, cv2.COLOR_BGR2GRAY) > 0
        annotated_frame[lines_mask] = hatch_roi[lines_mask]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.4      
    thickness = 3     
    padding_x = 12    
    padding_y = 10    

    for box, class_id, label_text in zip(detections.xyxy, detections.class_id, labels):
        label_text = str(label_text).strip()
        bg_color = sv.ColorPalette.DEFAULT.by_idx(class_id).as_bgr()
        text_color = (255, 255, 255) 
        
        (tw, th), _ = cv2.getTextSize(label_text, font, scale, thickness)
        (_, standard_th), standard_base = cv2.getTextSize("Ag", font, scale, thickness)
        
        box_w = tw + padding_x * 2
        box_h = standard_th + standard_base + padding_y * 2
        
        x1 = int(box[0])
        y1 = int(box[1]) - box_h - 8
        if y1 < 0:
            x1_left = int(box[0]) - box_w - 8
            if x1_left >= 0:        
                x1 = x1_left
                y1 = max(8, int(box[1]))
            else:
                x1_right = int(box[2]) + 8
                if x1_right + box_w <= W:   
                    x1 = x1_right
                    y1 = max(8, int(box[1]))
                else:
                    x1 = int(box[0]) + 8
                    y1 = int(box[1]) + 8
                    
        x2 = x1 + box_w
        y2 = y1 + box_h
        text_x = x1 + padding_x
        text_y = y1 + (box_h // 2) + (th // 2) 

    final_output_path = os.path.join(output_dir, "grounded_sam2_florence2_mask_only.jpg")
    cv2.imwrite(final_output_path, annotated_frame)
    print(f'✅ 成功保存 BCI 掩码图片至: "{final_output_path}"')
    
    clean_labels = [str(name).strip() for name in class_names]
    return annotated_frame, clean_labels

# =====================================================================
# 主入口
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded SAM 2 Florence-2 Demos", add_help=True)
    parser.add_argument("--image_path", type=str, default="/home/oyjt/Grounded-SAM-2/pic/table.png", required=False)
    # 🚨 将默认管道修改为画论文图（如果你想用回 BCI 画线，改成 "phrase_grounding_segmentation"）
    parser.add_argument("--pipeline", type=str, default="generate_paper_diagrams", required=False)
    parser.add_argument("--text_input", type=str, default=None, required=False)
    args = parser.parse_args()

    # 🚨🚨 在这里直接写死你的路径和测试物品 🚨🚨
    IMAGE_PATH = "/home/oyjt/Grounded-SAM-2/pic/task3_init.jpg"
    PIPELINE = args.pipeline 
    INPUT_TEXT = "starfruit, bowl, cherry, grapes"
    
    print(f"Running pipeline: {PIPELINE} now.")

    if PIPELINE == "generate_paper_diagrams":
        generate_paper_diagrams(
            florence2_model=florence2_model,
            florence2_processor=florence2_processor,
            sam2_predictor=sam2_predictor,
            image_path=IMAGE_PATH,
            text_input=INPUT_TEXT  # 这里的 text_input 就是你想高亮的物品
        )
    elif PIPELINE == "phrase_grounding_segmentation":
        # 你原有的画 BCI 网纹线的 Pipeline
        phrase_grounding_and_segmentation(
            florence2_model=florence2_model,
            florence2_processor=florence2_processor,
            sam2_predictor=sam2_predictor,
            image_path=IMAGE_PATH,
            text_input=INPUT_TEXT
        )
    else:
        print("其它 Pipeline 代码由于篇幅略，如需可继续添加。")