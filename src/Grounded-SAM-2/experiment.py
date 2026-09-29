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
from utils.supervision_utils import CUSTOM_COLOR_MAP
import json
import pycocotools.mask as mask_util
"""
Define Some Hyperparam
"""

TASK_PROMPT = {
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "more_detailed_caption": "<MORE_DETAILED_CAPTION",
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
Init Florence-2 and SAM 2 Model
"""

# FLORENCE2_MODEL_ID = "./Florence-2-large-ft"
FLORENCE2_MODEL_ID = "./Florence-2-large"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# environment settings
# use bfloat16
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# build florence-2
florence2_model = AutoModelForCausalLM.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True, torch_dtype='auto').eval().to(device)
florence2_processor = AutoProcessor.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True)

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


"""
We support a set of pipelines built by Florence-2 + SAM 2
"""

"""
Pipeline-1: Object Detection + Segmentation
"""
def object_detection_and_segmentation(
    florence2_model,
    florence2_processor,
    sam2_predictor,
    image_path,
    task_prompt="<OD>",
    text_input=None,
    output_dir=OUTPUT_DIR
):
    assert text_input is None, "Text input should be None when calling object detection pipeline."
    # run florence-2 object detection in demo
    image = Image.open(image_path).convert("RGB")
    results = run_florence2(task_prompt, text_input, florence2_model, florence2_processor, image)
    
    """ Florence-2 Object Detection Output Format
    {'<OD>': 
        {
            'bboxes': 
                [
                    [33.599998474121094, 159.59999084472656, 596.7999877929688, 371.7599792480469], 
                    [454.0799865722656, 96.23999786376953, 580.7999877929688, 261.8399963378906], 
                    [224.95999145507812, 86.15999603271484, 333.7599792480469, 164.39999389648438], 
                    [449.5999755859375, 276.239990234375, 554.5599975585938, 370.3199768066406], 
                    [91.19999694824219, 280.0799865722656, 198.0800018310547, 370.3199768066406]
                ], 
            'labels': ['car', 'door', 'door', 'wheel', 'wheel']
        }
    }
    """
    results = results[task_prompt]
    # parse florence-2 detection results
    input_boxes = np.array(results["bboxes"])
    class_names = results["labels"]
    class_ids = np.array(list(range(len(class_names))))
    
    # predict mask with SAM 2
    sam2_predictor.set_image(np.array(image))
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    # specify labels
    labels = [
        f"{class_name}" for class_name in class_names
    ]
    
    # visualization results
    img = cv2.imread(image_path)
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids
    )
    
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    cv2.imwrite(os.path.join(output_dir, "grounded_sam2_florence2_det_annotated_image.jpg"), annotated_frame)
    
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(output_dir, "grounded_sam2_florence2_det_image_with_mask.jpg"), annotated_frame)

    print(f'Successfully save annotated image to "{output_dir}"')

"""
Pipeline 2: Dense Region Caption + Segmentation
"""
def dense_region_caption_and_segmentation(
    florence2_model,
    florence2_processor,
    sam2_predictor,
    image_path,
    task_prompt="<DENSE_REGION_CAPTION>",
    text_input=None,
    output_dir=OUTPUT_DIR
):
    assert text_input is None, "Text input should be None when calling dense region caption pipeline."
    # run florence-2 object detection in demo
    image = Image.open(image_path).convert("RGB")
    results = run_florence2(task_prompt, text_input, florence2_model, florence2_processor, image)
    
    """ Florence-2 Object Detection Output Format
    {'<DENSE_REGION_CAPTION>': 
        {
            'bboxes': 
                [
                    [33.599998474121094, 159.59999084472656, 596.7999877929688, 371.7599792480469], 
                    [454.0799865722656, 96.23999786376953, 580.7999877929688, 261.8399963378906], 
                    [224.95999145507812, 86.15999603271484, 333.7599792480469, 164.39999389648438], 
                    [449.5999755859375, 276.239990234375, 554.5599975585938, 370.3199768066406], 
                    [91.19999694824219, 280.0799865722656, 198.0800018310547, 370.3199768066406]
                ], 
            'labels': ['turquoise Volkswagen Beetle', 'wooden double doors with metal handles', 'wheel', 'wheel', 'door']
        }
    }
    """
    results = results[task_prompt]
    # parse florence-2 detection results
    input_boxes = np.array(results["bboxes"])
    class_names = results["labels"]
    class_ids = np.array(list(range(len(class_names))))
    
    # predict mask with SAM 2
    sam2_predictor.set_image(np.array(image))
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    # specify labels
    labels = [
        f"{class_name}" for class_name in class_names
    ]
    
    # visualization results
    img = cv2.imread(image_path)
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids
    )
    
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    cv2.imwrite(os.path.join(output_dir, "grounded_sam2_florence2_dense_region_cap_annotated_image.jpg"), annotated_frame)
    
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(output_dir, "grounded_sam2_florence2_dense_region_cap_image_with_mask.jpg"), annotated_frame)

    print(f'Successfully save annotated image to "{output_dir}"')


"""
Pipeline 3: Region Proposal + Segmentation
"""
def region_proposal_and_segmentation(
    florence2_model,
    florence2_processor,
    sam2_predictor,
    image_path,
    task_prompt="<REGION_PROPOSAL>",
    text_input=None,
    output_dir=OUTPUT_DIR
):
    assert text_input is None, "Text input should be None when calling region proposal pipeline."
    # run florence-2 object detection in demo
    image = Image.open(image_path).convert("RGB")
    results = run_florence2(task_prompt, text_input, florence2_model, florence2_processor, image)
    
    """ Florence-2 Object Detection Output Format
    {'<REGION_PROPOSAL>': 
        {
            'bboxes': 
                [
                    [33.599998474121094, 159.59999084472656, 596.7999877929688, 371.7599792480469], 
                    [454.0799865722656, 96.23999786376953, 580.7999877929688, 261.8399963378906], 
                    [224.95999145507812, 86.15999603271484, 333.7599792480469, 164.39999389648438], 
                    [449.5999755859375, 276.239990234375, 554.5599975585938, 370.3199768066406], 
                    [91.19999694824219, 280.0799865722656, 198.0800018310547, 370.3199768066406]
                ], 
            'labels': ['', '', '', '', '', '', '']
        }
    }
    """
    results = results[task_prompt]
    # parse florence-2 detection results
    input_boxes = np.array(results["bboxes"])
    class_names = results["labels"]
    class_ids = np.array(list(range(len(class_names))))
    
    # predict mask with SAM 2
    sam2_predictor.set_image(np.array(image))
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    # specify labels
    labels = [
        f"region_{idx}" for idx, class_name in enumerate(class_names)
    ]
    
    # visualization results
    img = cv2.imread(image_path)
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids
    )
    
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    cv2.imwrite(os.path.join(output_dir, "grounded_sam2_florence2_region_proposal.jpg"), annotated_frame)
    
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(output_dir, "grounded_sam2_florence2_region_proposal_with_mask.jpg"), annotated_frame)

    print(f'Successfully save annotated image to "{output_dir}"')


"""
Pipeline 4: Phrase Grounding + Segmentation
"""
def phrase_grounding_and_segmentation(
    florence2_model,
    florence2_processor,
    sam2_predictor,
    image_path,
    task_prompt="<CAPTION_TO_PHRASE_GROUNDING>",
    text_input=None,
    output_dir=OUTPUT_DIR
):
    # run florence-2 object detection in demo
    image = Image.open(image_path).convert("RGB")
    results = run_florence2(task_prompt, text_input, florence2_model, florence2_processor, image)
    
    assert text_input is not None, "Text input should not be None when calling phrase grounding pipeline."
    results = results[task_prompt]
    
    # parse florence-2 detection results
    input_boxes = np.array(results["bboxes"])
    class_names = results["labels"]
    class_ids = np.array(list(range(len(class_names))))
    
    # predict mask with SAM 2
    sam2_predictor.set_image(np.array(image))
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    
    if masks.ndim == 4:
        masks = masks.squeeze(1)
        
    # ================= 1. 准备标签内容 (物品名称 + 频率) =================
    bci_freqs = ["2Hz", "5Hz", "10Hz", "12Hz", "15Hz"]
    labels =[]
    for i in range(len(class_names)):
        freq = bci_freqs[i % len(bci_freqs)] 
        item_name = str(class_names[i]).strip() # 清除模型可能输出的多余空格
        labels.append(f"{item_name} ")
    
    # visualization results
    img = cv2.imread(image_path)
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids
    )
    
    # ================= 2. 绘制基础底色 (半透明掩码) =================
    mask_annotator = sv.MaskAnnotator(opacity=0.4) 
    annotated_frame = mask_annotator.annotate(scene=img.copy(), detections=detections)

    # ================= 3. 绘制高级 BCI 视觉元素 (斜线 + 轮廓) =================
    H, W = annotated_frame.shape[:2] 
    
    for idx, (mask, class_id) in enumerate(zip(masks, class_ids)):
        color = sv.ColorPalette.DEFAULT.by_idx(class_id).as_bgr()
        
        # --- a. 提取二值掩码的轮廓并画边缘线 ---
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated_frame, contours, -1, color, 5)
        
        # --- b. 在掩码内部画斜线（代表频率特征） ---
        spacing = 10 + (idx % 4) * 15  
        line_thickness = 2
        hatch_canvas = np.zeros_like(annotated_frame)
        
        # 交叉网纹/斜线区分不同目标
        if idx % 2 == 0:
            for offset in range(-H, W, spacing):
                cv2.line(hatch_canvas, (offset, 0), (offset + H, H), color, line_thickness)
        else:
            for offset in range(0, W + H, spacing):
                cv2.line(hatch_canvas, (offset, 0), (offset - H, H), color, line_thickness)
                
        # 提取斜线并叠加到原图
        hatch_roi = cv2.bitwise_and(hatch_canvas, hatch_canvas, mask=mask.astype(np.uint8))
        lines_mask = cv2.cvtColor(hatch_roi, cv2.COLOR_BGR2GRAY) > 0
        annotated_frame[lines_mask] = hatch_roi[lines_mask]

    # ================= 4. 自定义绘制：高度统一、防遮挡的完美居中标签 =================
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.4      # 字号大小
    thickness = 3     # 字体粗细
    padding_x = 12    # 左右留白
    padding_y = 10    # 上下留白

    for box, class_id, label_text in zip(detections.xyxy, detections.class_id, labels):
        # 再次确认剔除多余空格
        label_text = str(label_text).strip()
        
        bg_color = sv.ColorPalette.DEFAULT.by_idx(class_id).as_bgr()
        text_color = (255, 255, 255) 
        
        # 获取当前文字的真实宽度和视觉高度
        (tw, th), _ = cv2.getTextSize(label_text, font, scale, thickness)
        
        # 虚拟一个包含最高字母和最低字母的标准字符串 "Ag" (强制固定方块总高度)
        (_, standard_th), standard_base = cv2.getTextSize("Ag", font, scale, thickness)
        
        # 计算统一的方块宽高
        box_w = tw + padding_x * 2
        box_h = standard_th + standard_base + padding_y * 2
        
        # --- 动态防遮挡排版逻辑 ---
        # 首选方案：悬浮在物体上方，左侧对齐
        x1 = int(box[0])
        y1 = int(box[1]) - box_h - 8
        
        # 如果上方出界
        if y1 < 0:
            # 尝试方案 B：放在物体【左侧】
            x1_left = int(box[0]) - box_w - 8
            if x1_left >= 0:        # 左侧放得下
                x1 = x1_left
                y1 = max(8, int(box[1]))

            else:
                # 尝试方案 A：放在物体【右侧】
                x1_right = int(box[2]) + 8
                if x1_right + box_w <= W:   # 右侧放得下
                    x1 = x1_right
                    y1 = max(8, int(box[1]))
                else:
                    # 终极保底：贴在物体内部左上角
                    x1 = int(box[0]) + 8
                    y1 = int(box[1]) + 8
                    
        # 最终计算右下角终点
        x2 = x1 + box_w
        y2 = y1 + box_h
        
        # # 绘制彩色背景方框
        # cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), bg_color, -1)
        
        # 【视觉绝对居中算法】：中心 Y 坐标 + 文字视觉高度的一半
        text_x = x1 + padding_x
        text_y = y1 + (box_h // 2) + (th // 2) 
        
        # # 绘制抗锯齿文字
        # cv2.putText(annotated_frame, label_text, (text_x, text_y), font, scale, text_color, thickness, cv2.LINE_AA)

    # 保存最终完美的标注图片
    final_output_path = os.path.join(output_dir, "grounded_sam2_florence2_mask_only.jpg")
    cv2.imwrite(final_output_path, annotated_frame)
    print(f'✅ 成功保存带高级排版的图片至: "{final_output_path}"')
    
    clean_labels = [str(name).strip() for name in class_names]
    return annotated_frame, clean_labels
    # ================= 5. 保存 JSON 数据 (可选保留) =================
    mask_rles = [single_mask_to_rle(mask) for mask in masks]
    json_results = {
        "image_path": image_path,
        "pipeline": "phrase_grounding",
        "input_text": text_input,
        "annotations": []
    }
    for i in range(len(class_names)):
        box = input_boxes[i].tolist()
        json_results["annotations"].append({
            "class_name": class_names[i],
            "bbox": box,
            "segmentation": mask_rles[i],
            "score": float(scores[i]),
            "center": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        })

    json_filename = os.path.join(output_dir, "florence2_sam2_results.json")
    with open(json_filename, "w") as f:
        json.dump(json_results, f, indent=4)
    print(f'✅ 数据结果已保存至: {json_filename}')
#####################################################

"""
Pipeline 5: Referring Expression Segmentation

Note that Florence-2 directly support referring segmentation with polygon output format, which may be not that accurate, 
therefore we try to decode box from polygon and use SAM 2 for mask prediction
"""
def referring_expression_segmentation(
    florence2_model,
    florence2_processor,
    sam2_predictor,
    image_path,
    task_prompt="<REFERRING_EXPRESSION_SEGMENTATION>",
    text_input=None,
    output_dir=OUTPUT_DIR
):
    # run florence-2 object detection in demo
    image = Image.open(image_path).convert("RGB")
    results = run_florence2(task_prompt, text_input, florence2_model, florence2_processor, image)
    
    """ Florence-2 Object Detection Output Format
    {'<REFERRING_EXPRESSION_SEGMENTATION>': 
        {
            'polygons': [[[...]]]
            'labels': ['']
        }
    }
    """
    assert text_input is not None, "Text input should not be None when calling referring segmentation pipeline."
    results = results[task_prompt]
    # parse florence-2 detection results
    polygon_points = np.array(results["polygons"][0], dtype=np.int32).reshape(-1, 2)
    class_names = [text_input]
    class_ids = np.array(list(range(len(class_names))))
    
    # parse polygon format to mask
    img_width, img_height = image.size[0], image.size[1]
    florence2_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    if len(polygon_points) < 3:
        print("Invalid polygon:", polygon_points)
        exit()
    cv2.fillPoly(florence2_mask, [polygon_points], 1)
    if florence2_mask.ndim == 2:
        florence2_mask = florence2_mask[None]

    # compute bounding box based on polygon points
    x_min = np.min(polygon_points[:, 0])
    y_min = np.min(polygon_points[:, 1])
    x_max = np.max(polygon_points[:, 0])
    y_max = np.max(polygon_points[:, 1])

    input_boxes = np.array([[x_min, y_min, x_max, y_max]])

    # predict mask with SAM 2
    sam2_predictor.set_image(np.array(image))
    sam2_masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    
    if sam2_masks.ndim == 4:
        sam2_masks = sam2_masks.squeeze(1)
    
    # specify labels
    labels = [
        f"{class_name}" for class_name in class_names
    ]
    
    # visualization florence2 mask
    img = cv2.imread(image_path)
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=florence2_mask.astype(bool),
        class_id=class_ids
    )
    
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    cv2.imwrite(os.path.join(output_dir, "florence2_referring_segmentation_box.jpg"), annotated_frame)
    
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(output_dir, "florence2_referring_segmentation_box_with_mask.jpg"), annotated_frame)

    print(f'Successfully save florence-2 annotated image to "{output_dir}"')

    # visualize sam2 mask
    img = cv2.imread(image_path)
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=sam2_masks.astype(bool),
        class_id=class_ids
    )

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    cv2.imwrite(os.path.join(output_dir, "grounded_sam2_florence2_referring_box.jpg"), annotated_frame)
    
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(output_dir, "grounded_sam2_florence2_referring_box_with_sam2_mask.jpg"), annotated_frame)

    print(f'Successfully save sam2 annotated image to "{output_dir}"')


"""
Pipeline 6: Open-Vocabulary Detection + Segmentation
"""
def open_vocabulary_detection_and_segmentation(
    florence2_model,
    florence2_processor,
    sam2_predictor,
    image_path,
    task_prompt="<OPEN_VOCABULARY_DETECTION>",
    text_input=None,
    output_dir=OUTPUT_DIR
):
    # run florence-2 object detection in demo
    image = Image.open(image_path).convert("RGB")
    results = run_florence2(task_prompt, text_input, florence2_model, florence2_processor, image)
    
    """ Florence-2 Open-Vocabulary Detection Output Format
    {'<OPEN_VOCABULARY_DETECTION>': 
        {
            'bboxes': 
                [
                    [34.23999786376953, 159.1199951171875, 582.0800170898438, 374.6399841308594]
                ], 
            'bboxes_labels': ['A green car'],
            'polygons': [], 
            'polygons_labels': []
        }
    }
    """
    assert text_input is not None, "Text input should not be None when calling open-vocabulary detection pipeline."
    results = results[task_prompt]
    # parse florence-2 detection results
    input_boxes = np.array(results["bboxes"])
    print(results)
    class_names = results["bboxes_labels"]
    class_ids = np.array(list(range(len(class_names))))
    
    # predict mask with SAM 2
    sam2_predictor.set_image(np.array(image))
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    # specify labels
    labels = [
        f"{class_name}" for class_name in class_names
    ]
    
    # visualization results
    img = cv2.imread(image_path)
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids
    )
    
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    cv2.imwrite(os.path.join(output_dir, "grounded_sam2_florence2_open_vocabulary_detection.jpg"), annotated_frame)
    
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(output_dir, "grounded_sam2_florence2_open_vocabulary_detection_with_mask.jpg"), annotated_frame)

    print(f'Successfully save annotated image to "{output_dir}"')

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded SAM 2 Florence-2 Demos", add_help=True)
    parser.add_argument("--image_path", type=str, default="/home/oyjt/Grounded-SAM-2/pic/table.png", required=False, help="path to image file")
    # parser.add_argument("--pipeline", type=str, default="object_detection_segmentation", required=False, help="path to image file")
    parser.add_argument("--pipeline", type=str, default="phrase_grounding_segmentation", required=False, help="path to image file")
    parser.add_argument("--text_input", type=str, default=None, required=False, help="path to image file")
    args = parser.parse_args()

    # IMAGE_PATH = args.image_path
    IMAGE_PATH = "/home/oyjt/Grounded-SAM-2/pic/table2.jpg"
    PIPELINE = args.pipeline
    # INPUT_TEXT = args.text_input
    INPUT_TEXT = "starfruit, pen, capsule"
    print(f"Running pipeline: {PIPELINE} now.")

    if PIPELINE == "object_detection_segmentation":
        # pipeline-1: detection + segmentation
        object_detection_and_segmentation(
            florence2_model=florence2_model,
            florence2_processor=florence2_processor,
            sam2_predictor=sam2_predictor,
            image_path=IMAGE_PATH
        )
    elif PIPELINE == "dense_region_caption_segmentation":
        # pipeline-2: dense region caption + segmentation
        dense_region_caption_and_segmentation(
            florence2_model=florence2_model,
            florence2_processor=florence2_processor,
            sam2_predictor=sam2_predictor,
            image_path=IMAGE_PATH
        )
    elif PIPELINE == "region_proposal_segmentation":
        # pipeline-3: dense region caption + segmentation
        region_proposal_and_segmentation(
            florence2_model=florence2_model,
            florence2_processor=florence2_processor,
            sam2_predictor=sam2_predictor,
            image_path=IMAGE_PATH
        )
    elif PIPELINE == "phrase_grounding_segmentation":
        # pipeline-4: phrase grounding + segmentation
        phrase_grounding_and_segmentation(
            florence2_model=florence2_model,
            florence2_processor=florence2_processor,
            sam2_predictor=sam2_predictor,
            image_path=IMAGE_PATH,
            text_input=INPUT_TEXT
        )
    elif PIPELINE == "referring_expression_segmentation":
        # pipeline-5: referring segmentation + segmentation
        referring_expression_segmentation(
            florence2_model=florence2_model,
            florence2_processor=florence2_processor,
            sam2_predictor=sam2_predictor,
            image_path=IMAGE_PATH,
            text_input=INPUT_TEXT
        )
    elif PIPELINE == "open_vocabulary_detection_segmentation":
        # pipeline-6: open-vocabulary detection + segmentation
        open_vocabulary_detection_and_segmentation(
            florence2_model=florence2_model,
            florence2_processor=florence2_processor,
            sam2_predictor=sam2_predictor,
            image_path=IMAGE_PATH,
            text_input=INPUT_TEXT
        )
    else:
        raise NotImplementedError(f"Pipeline: {args.pipeline} is not implemented at this time")