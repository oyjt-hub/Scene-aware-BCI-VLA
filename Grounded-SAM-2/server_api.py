import os
import json
import base64
import shutil
import traceback
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
from typing import Optional
from pydantic import BaseModel
import time
import uuid
from datetime import datetime
import threading

import grounded_sam2_florence2_image_demo as demo


LOG_DIR = "./exp_logs/starfruit"
os.makedirs(LOG_DIR, exist_ok=True)
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CURRENT_TRIAL = None
TRIAL_COUNTER = 0 
lock = threading.Lock()

app = FastAPI()


@app.on_event("shutdown")
def shutdown_event():
    """
    Triggered when pressing Ctrl+C to shut down the server, used to rescue unsaved experiment data.
    """
    global CURRENT_TRIAL
    print("\n🛑 Received Ctrl+C interrupt signal, preparing to shut down Vision Server...")
    
    if CURRENT_TRIAL is not None:
        if CURRENT_TRIAL.get("end_time") is None:
            CURRENT_TRIAL["end_time"] = time.time()
            if CURRENT_TRIAL.get("start_time"):
                CURRENT_TRIAL["total_time"] = CURRENT_TRIAL["end_time"] - CURRENT_TRIAL["start_time"]
            
            CURRENT_TRIAL["success"] = False 
            CURRENT_TRIAL["interrupt_by_user"] = True 
            
            save_trial()
            print(f"💾 [Logger] Successfully rescued and saved interrupted experiment data: {CURRENT_TRIAL['trial_id']}.json")

def log_trial():
    global CURRENT_TRIAL
    if CURRENT_TRIAL is None:
        return
    path = os.path.join(LOG_DIR, f"{CURRENT_TRIAL['trial_id']}.json")
    with open(path, "w") as f:
        json.dump(CURRENT_TRIAL, f, indent=2)

def save_trial():
    global CURRENT_TRIAL
    if CURRENT_TRIAL is None:
        return
    # path = os.path.join(LOG_DIR, f"{CURRENT_TRIAL['trial_id']}.json")
    path = os.path.join(LOG_DIR, f"{CURRENT_TRIAL['trial_id']}.json")
    with open(path, "w") as f:
        json.dump(CURRENT_TRIAL, f, indent=2)

def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(b1_area + b2_area - inter_area + 1e-6)

def start_trial():
    global CURRENT_TRIAL, TRIAL_COUNTER  
    
    TRIAL_COUNTER += 1 
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
    custom_id = f"{TRIAL_COUNTER}-{time_str}" 
    
    CURRENT_TRIAL = {
        "trial_id": custom_id,  
        "timestamp": datetime.now().isoformat(),
        "start_time": time.time(),
        "mode": None,
        "prompt": None,
        "selected_target": None,
        "n_trigger": 0,
        "scene_update_time": None,
        "vla_time": None,
        "total_time": None,
        "success": None,
    }


GLOBAL_SCENE_DATA = {
    "status": "waiting",
    "image_b64": "",
    "labels": [],
    "bboxes": []
}
GLOBAL_BCI_RESULT = None

class BCIResult(BaseModel):
    selected_target: str



@app.post("/update_scene")
async def update_scene(
    image_file: UploadFile = File(...),
    mode: str = Form("od"),            
    prompt: Optional[str] = Form(""),  
    threshold: float = Form(0.4)
):
    global GLOBAL_SCENE_DATA, GLOBAL_BCI_RESULT
    GLOBAL_BCI_RESULT = None 
    start_trial()
    CURRENT_TRIAL["scene_update_time"] = time.time()
    CURRENT_TRIAL["mode"] = mode
    CURRENT_TRIAL["prompt"] = prompt
    temp_input_path = os.path.join(OUTPUT_DIR, "temp_bci_input.jpg")
    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(image_file.file, buffer)
        
    try:
        image = Image.open(temp_input_path).convert("RGB")
        
        if mode == "od":
            print(f"📡 [Vision Server] Starting <OD> global object detection (threshold: {threshold})...")
            task_prompt = "<OD>"
            text_input = None  
        else:
            print(f"📡 [Vision Server] Starting <CAPTION_TO_PHRASE_GROUNDING> (prompt: '{prompt}', threshold: {threshold})...")
            task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
            text_input = prompt

        t_vla = time.time()
        results = demo.run_florence2(
            task_prompt=task_prompt, 
            text_input=text_input, 
            model=demo.florence2_model, 
            processor=demo.florence2_processor, 
            image=image
        )
        CURRENT_TRIAL["vla_time"] = time.time() - t_vla
        
        results = results[task_prompt]
        input_boxes = np.array(results["bboxes"])
        class_names = results["labels"]
        
        if len(input_boxes) == 0:
            return JSONResponse(content={"status": "error", "message": "No objects found."})

        demo.sam2_predictor.set_image(np.array(image))
        masks, scores, _ = demo.sam2_predictor.predict(
            point_coords=None, point_labels=None, box=input_boxes, multimask_output=False
        )

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
                print(f"⚠️ Filtering oversized object: '{class_names[i]}' (ratio {area_ratio*100:.1f}%)")

        if not valid_indices:
            return JSONResponse(content={"status": "error", "message": "All objects filtered out."})
            
        masks = masks[valid_indices]
        input_boxes = input_boxes[valid_indices]
        class_names = [class_names[i] for i in valid_indices]
        scores = scores[valid_indices] 

        scores_1d = np.array(scores).flatten()
        sorted_indices = np.argsort(scores_1d)[::-1]
        
        keep_indices = []
        for idx in sorted_indices:
            idx = int(idx)  
            
            is_duplicate = False
            for keep_idx in keep_indices:
                box1 = np.array(input_boxes[idx]).flatten()
                box2 = np.array(input_boxes[keep_idx]).flatten()
                
                if compute_iou(box1, box2) > 0.85:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                keep_indices.append(idx)

        final_masks = masks[keep_indices]
        final_boxes = input_boxes[keep_indices].tolist()
        final_labels = [class_names[i] for i in keep_indices]

        with open(temp_input_path, "rb") as img_file:
            img_b64 = base64.b64encode(img_file.read()).decode('utf-8')

        mask_b64s = []
        for idx, mask in enumerate(final_masks):
            try:
                mask_np = np.array(mask)
                while mask_np.ndim > 2:
                    mask_np = mask_np[0]
                binary_mask = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.uint8)
                binary_mask[mask_np > 0] = 255
                binary_mask = np.ascontiguousarray(binary_mask)
                success, buffer = cv2.imencode('.png', binary_mask)
                if success:
                    mask_b64s.append(base64.b64encode(buffer).decode('utf-8'))
                else:
                    print(f"⚠️ Warning: Mask {idx} failed to encode as PNG, shape: {binary_mask.shape}")
            except Exception as e:
                print(f"❌ Mask {idx} processing exception: {e}")

        GLOBAL_SCENE_DATA = {
            "status": "success",
            "image_b64": img_b64,
            "labels": final_labels,
            "bboxes": final_boxes,
            "masks": mask_b64s  
        }
        
        print(f"✅ [Vision Server] Scene update successful, extracted {len(final_labels)} objects: {final_labels}")
        return JSONResponse(content={"status": "success"})
    
    except Exception as e:
        traceback.print_exc()
        if CURRENT_TRIAL:
            CURRENT_TRIAL["end_time"] = time.time()
            CURRENT_TRIAL["total_time"] = CURRENT_TRIAL["end_time"] - CURRENT_TRIAL["start_time"]
            CURRENT_TRIAL["success"] = False
            save_trial()
        return JSONResponse(content={"status": "error"})

@app.get("/get_bci_scene")
async def get_bci_scene():
    return JSONResponse(content=GLOBAL_SCENE_DATA)

@app.post("/submit_bci_result")
async def submit_bci_result(result: BCIResult):
    global GLOBAL_BCI_RESULT, CURRENT_TRIAL

    GLOBAL_BCI_RESULT = result.selected_target

    with lock:
        if CURRENT_TRIAL:
            CURRENT_TRIAL["selected_target"] = result.selected_target
            CURRENT_TRIAL["n_trigger"] += 1
            
            CURRENT_TRIAL["end_time"] = time.time()
            if CURRENT_TRIAL["start_time"]:
                CURRENT_TRIAL["total_time"] = CURRENT_TRIAL["end_time"] - CURRENT_TRIAL["start_time"]
            CURRENT_TRIAL["success"] = True 
            
            save_trial()
            print(f"📁 [Logger] Experiment data successfully saved to {LOG_DIR}/{CURRENT_TRIAL['trial_id']}.json")

    print(f"🧠 [BCI Result] Received EEG decoding result from Windows client: {GLOBAL_BCI_RESULT}")

    return JSONResponse(content={"status": "success"})

@app.get("/get_bci_result")
async def get_bci_result():
    if GLOBAL_BCI_RESULT is not None:
        return JSONResponse(content={"status": "success", "selected_target": GLOBAL_BCI_RESULT})
    else:
        return JSONResponse(content={"status": "waiting"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)