import cv2
import numpy as np
import os

OUTPUT_DIR = "./outputs"

def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_len=15):
    """手写一个画虚线的函数，用于连接金字塔层级"""
    x1, y1 = pt1
    x2, y2 = pt2
    dist = np.hypot(x2 - x1, y2 - y1)
    if dist == 0: return
    dashes = int(dist / dash_len)
    for i in range(dashes):
        # 只画偶数段，形成虚线
        if i % 2 == 0:
            start_x = int(x1 + (x2 - x1) * i / dashes)
            start_y = int(y1 + (y2 - y1) * i / dashes)
            end_x = int(x1 + (x2 - x1) * (i + 1) / dashes)
            end_y = int(y1 + (y2 - y1) * (i + 1) / dashes)
            cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)

def create_3d_feature_pyramid():
    # 读取特征图
    img_low = cv2.imread(os.path.join(OUTPUT_DIR, "4_Pyramid_Feat1_Low.jpg"))
    img_mid = cv2.imread(os.path.join(OUTPUT_DIR, "4_Pyramid_Feat2_Mid.jpg"))
    img_high = cv2.imread(os.path.join(OUTPUT_DIR, "4_Pyramid_Feat3_High.jpg"))

    if img_low is None or img_mid is None or img_high is None:
        print("❌ 找不到特征图！请确认你已经成功运行了上一步的代码。")
        return

    # 【关键修改1】将读取的 3 通道 BGR 图像转换为 4 通道 BGRA 图像，赋予完全不透明的 Alpha 通道
    img_low = cv2.cvtColor(img_low, cv2.COLOR_BGR2BGRA)
    img_mid = cv2.cvtColor(img_mid, cv2.COLOR_BGR2BGRA)
    img_high = cv2.cvtColor(img_high, cv2.COLOR_BGR2BGRA)

    canvas_w, canvas_h = 1400, 1300
    
    # 【关键修改2】创建一个全透明画布：使用 np.zeros，形状为 4 通道，值全是 0 (即完全透明)
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

    # 定义 3 层的金字塔参数（自下而上变小，模拟感受野聚合）
    cx = 800  
    
    layers = [
        {"img": img_low,  "y": 800, "scale": 1.0},
        {"img": img_mid,  "y": 450, "scale": 0.75},
        {"img": img_high, "y": 150, "scale": 0.5}
    ]

    # 计算每一层的 3D 菱形顶点坐标
    for layer in layers:
        scale = layer["scale"]
        y = layer["y"]
        dw = int(400 * scale) 
        dh = int(200 * scale) 
        
        pts = np.float32([
            [cx, y],                # Top
            [cx + dw, y + dh],      # Right
            [cx, y + 2*dh],         # Bottom
            [cx - dw, y + dh]       # Left
        ])
        layer["pts"] = pts
        layer["dw"], layer["dh"] = dw, dh

    # ================= 1. 先画底层的连接虚线 =================
    # 【关键修改3】画笔颜色需要加上第 4 个参数 255 (代表完全不透明的线条)
    line_color = (180, 180, 180, 255) 
    for i in range(len(layers) - 1):
        pts_bottom = layers[i]["pts"]
        pts_top = layers[i+1]["pts"]
        for j in range(4):
            draw_dashed_line(canvas, tuple(pts_bottom[j]), tuple(pts_top[j]), line_color, thickness=2)
            
    # 画一条贯穿中心的向上主箭头 (颜色也要加上255)
    cv2.arrowedLine(canvas, (cx, 1150), (cx, 80), (150, 150, 150, 255), 3, tipLength=0.03, line_type=cv2.LINE_AA)

    # ================= 2. 将图片 3D 扭曲并贴到画布上 =================
    for layer in layers:
        img = layer["img"]
        pts = layer["pts"]
        
        h, w = img.shape[:2]
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src_pts, pts)
        
        # 将原图扭曲成 3D 菱形 (由于 img 已经是 4 通道，扭曲后依然是带透明通道的)
        warped = cv2.warpPerspective(img, M, (canvas_w, canvas_h))
        
        # 创建遮罩并合成到画布
        mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts.astype(int), 255)
        
        # 仅将有图像像素的区域贴合到全透明画布上
        canvas[mask > 0] = warped[mask > 0]
        
        # 给菱形加上一层深灰色的边框 (颜色带上 255 透明度参数)
        cv2.polylines(canvas, [pts.astype(int)], isClosed=True, color=(100, 100, 100, 255), thickness=2, lineType=cv2.LINE_AA)

    # 【关键修改4】必须保存为 .png 后缀，JPG 不支持透明底
    final_path = os.path.join(OUTPUT_DIR, "5_Feature_Pyramid_Final_Transparent.png")
    cv2.imwrite(final_path, canvas)
    print(f"🎉 搞定！【无背景/透明底】的 3D 特征金字塔已生成：{final_path}")
    print("💡 直接把这张 PNG 拖进 PPT，它将完美融入你 PPT 任意颜色的背景！")

if __name__ == "__main__":
    create_3d_feature_pyramid()