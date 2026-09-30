import cv2
import numpy as np
import os

OUTPUT_DIR = "./outputs"

# ================= 辅助绘画函数 =================
def draw_text(img, text, position, font_scale=1.0, color=(50, 50, 50), thickness=2, center=False):
    """学术感抗锯齿文字绘制"""
    font = cv2.FONT_HERSHEY_DUPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    if center:
        x -= text_size[0] // 2
    cv2.putText(img, text, (int(x), int(y)), font, font_scale, color, thickness, cv2.LINE_AA)

def draw_arrow(img, pt1, pt2, text="", text_offset=(0, -15)):
    """带文字的高级灰色数据流箭头"""
    color = (150, 140, 130) # 高级灰蓝色
    cv2.arrowedLine(img, pt1, pt2, color, 4, tipLength=0.03, line_type=cv2.LINE_AA)
    if text:
        cx, cy = (pt1[0]+pt2[0])//2, (pt1[1]+pt2[1])//2
        draw_text(img, text, (cx + text_offset[0], cy + text_offset[1]), font_scale=0.8, color=(80, 80, 100), thickness=2, center=True)

def overlay_image(bg, fg, x, y):
    """将子图贴到底图上，并加上高质感边框"""
    h, w = fg.shape[:2]
    bg[y:y+h, x:x+w] = fg
    cv2.rectangle(bg, (x, y), (x+w, y+h), (180, 180, 180), 3, cv2.LINE_AA)

def create_3d_layer(img, cx, cy, dw, dh):
    """将 2D 图片扭曲成 3D 菱形图层"""
    h, w = img.shape[:2]
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = np.float32([[cx, cy - dh], [cx + dw, cy], [cx, cy + dh], [cx - dw, cy]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (2800, 1200))
    mask = np.zeros((1200, 2800), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_pts.astype(int), 255)
    return warped, mask, dst_pts

# ================= 核心拼图主逻辑 =================
def build_pipeline_diagram():
    print("🚀 正在启动一键学术排版引擎...")
    
    # 1. 创建超高清浅色画布 (2800 x 1200)
    W, H = 2800, 1200
    canvas = np.full((H, W, 3), (250, 248, 245), dtype=np.uint8) # 学术暖灰底色
    
    # 2. 画大标题
    draw_text(canvas, "[Block B] Neural-Cognitive Visual Perception Hub", (80, 80), font_scale=1.6, color=(40, 40, 60), thickness=3)

    # 3. 绘制两个核心算法模块面板 (Panel)
    # Florence-2 Panel
    cv2.rectangle(canvas, (750, 180), (1900, 950), (240, 235, 230), -1)
    cv2.rectangle(canvas, (750, 180), (1900, 950), (200, 190, 180), 3, cv2.LINE_AA)
    draw_text(canvas, "Florence-2 (Semantic Grounding)", (1325, 230), font_scale=1.2, color=(60, 60, 120), center=True)

    # SAM-2 Panel
    cv2.rectangle(canvas, (1980, 180), (2700, 950), (235, 240, 230), -1)
    cv2.rectangle(canvas, (1980, 180), (2700, 950), (180, 200, 180), 3, cv2.LINE_AA)
    draw_text(canvas, "SAM-2 (Geometric Refinement)", (2340, 230), font_scale=1.2, color=(60, 120, 60), center=True)

    # 4. 加载图片并统一缩放尺寸
    try:
        img_rgb = cv2.imread(os.path.join(OUTPUT_DIR, "1_Raw_RGB.jpg"))
        img_heat = cv2.imread(os.path.join(OUTPUT_DIR, "2_Florence2_Heatmap.jpg"))
        img_mask = cv2.imread(os.path.join(OUTPUT_DIR, "3_SAM2_Mask_For_Paper.jpg"))
        feat1 = cv2.imread(os.path.join(OUTPUT_DIR, "4_Pyramid_Feat1_Low.jpg"))
        feat2 = cv2.imread(os.path.join(OUTPUT_DIR, "4_Pyramid_Feat2_Mid.jpg"))
        feat3 = cv2.imread(os.path.join(OUTPUT_DIR, "4_Pyramid_Feat3_High.jpg"))
        
        # 统一宽度为 500
        target_w = 500
        aspect = img_rgb.shape[0] / img_rgb.shape[1]
        target_h = int(target_w * aspect)
        
        img_rgb = cv2.resize(img_rgb, (target_w, target_h))
        img_heat = cv2.resize(img_heat, (target_w, target_h))
        img_mask = cv2.resize(img_mask, (target_w, target_h))
        
    except Exception as e:
        print("❌ 图片加载失败，请确保你已经生成了前 6 张图！", e)
        return

    # 5. [第一列] 粘贴 Raw RGB
    y_center = 500
    overlay_image(canvas, img_rgb, 100, y_center)
    draw_text(canvas, "Real-time RGB Observation", (350, y_center + target_h + 40), font_scale=0.9, center=True)

    # 6. [第二列] 绘制 3D 特征金字塔 (Florence-2 内部)
    pyr_cx = 1000  # 金字塔中心X
    pyr_dw, pyr_dh = 180, 90
    layers = [
        (feat1, 750, 1.0, "Low-level Features"),
        (feat2, 550, 0.8, "Mid-level Features"),
        (feat3, 350, 0.6, "High-level ViT Patches")
    ]
    
    for img, cy, scale, label in layers:
        w, h, pts = create_3d_layer(img, pyr_cx, cy, int(pyr_dw*scale), int(pyr_dh*scale))
        canvas[h > 0] = w[h > 0]
        cv2.polylines(canvas, [pts.astype(int)], True, (150, 150, 150), 2, cv2.LINE_AA)
        draw_text(canvas, label, (pyr_cx - 200, cy + 10), font_scale=0.7, color=(80, 80, 80))

    # 金字塔下方的描述词
    draw_text(canvas, "DaViT Backbone & Multi-scale Extraction", (1000, 880), font_scale=0.9, center=True, color=(100, 50, 50))

    # 7. [第三列] 粘贴 Heatmap (Florence-2 输出)
    overlay_image(canvas, img_heat, 1320, y_center)
    draw_text(canvas, "Seq2Seq Bbox Generation", (1570, y_center - 20), font_scale=0.9, center=True)
    draw_text(canvas, "[x_min, y_min, x_max, y_max]", (1570, y_center + target_h + 40), font_scale=0.9, center=True, color=(50, 50, 150))

    # 8. [第四列] 粘贴 SAM-2 Mask
    overlay_image(canvas, img_mask, 2100, y_center)
    draw_text(canvas, "Two-way Cross-Attention", (2350, y_center - 20), font_scale=0.9, center=True)
    draw_text(canvas, "Zero-shot Mask Decoder", (2350, y_center + target_h + 40), font_scale=0.9, center=True, color=(50, 100, 50))

    # 9. 连线与数据流（灵魂注入！）
    # RGB -> 金字塔
    draw_arrow(canvas, (620, y_center + 100), (800, y_center + 100), text="Global Features")
    
    # 金字塔 -> Heatmap
    draw_arrow(canvas, (1180, y_center + 100), (1300, y_center + 100), text="Reasoning")
    
    # Heatmap -> SAM-2 (传递框)
    draw_arrow(canvas, (1840, y_center + 100), (2080, y_center + 100), text="Spatial Prompt (Box)", text_offset=(0, -20))
    
    # RGB -> SAM-2 (底层长线特征传递，解耦设计的精髓！)
    # 画一条折线从 RGB 底部绕到 SAM-2 底部
    p1, p2, p3, p4 = (350, y_center + target_h + 80), (350, 1100), (2350, 1100), (2350, y_center + target_h + 80)
    cv2.polylines(canvas, [np.array([p1, p2, p3, p4])], False, (150, 140, 130), 4, cv2.LINE_AA)
    cv2.arrowedLine(canvas, p3, p4, (150, 140, 130), 4, tipLength=0.08, line_type=cv2.LINE_AA)
    draw_text(canvas, "Hiera ViT High-Res Features", (1350, 1080), font_scale=0.9, color=(80, 80, 120), center=True)

    # 10. 最终输出箭头 (流出 Block B)
    # 向上输出（给屏幕 BCI 闪烁用）
    draw_arrow(canvas, (2350, y_center - 80), (2350, 50), text="Mask-to-Frequency (SSVEP)", text_offset=(-220, 10))
    # 向右输出（给 VLA 执行动作）
    draw_arrow(canvas, (2620, y_center + 100), (2750, y_center + 100), text="Cropped Context", text_offset=(-60, -20))

    # 保存超级图表！
    final_path = os.path.join(OUTPUT_DIR, "6_Block_B_Final_Pipeline.jpg")
    cv2.imwrite(final_path, canvas)
    print(f"🎉 大功告成！完美架构图已生成在: {final_path}")
    print("💡 你现在只需把这张图直接插入论文，享受审稿人的惊叹吧！")

if __name__ == "__main__":
    build_pipeline_diagram()