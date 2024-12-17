import cv2
import numpy as np
import torch
from ultralytics import YOLO
import streamlit as st
import os

# 定义字体和大小
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
text_color = (0, 255, 0)  # 绿色
text_position_offset = (10, 10)  # 文字相对于框左上角的偏移量

def convert_obb_to_corners(obb):
    """将 OBB (x, y, w, h, r) 转换为四个角的坐标"""
    if isinstance(obb, torch.Tensor):
        obb = obb.cpu().numpy()

    x, y, w, h, r = obb
    cos_r = np.cos(r)
    sin_r = np.sin(r)

    corners = np.array([
        [-w / 2, -h / 2],
        [w / 2, -h / 2],
        [w / 2, h / 2],
        [-w / 2, h / 2]
    ])

    rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    rotated_corners = np.dot(corners, rotation_matrix.T)
    translated_corners = rotated_corners + np.array([x, y])

    return np.round(translated_corners).astype(int).flatten()

def draw_detections(image, detections):
    """在图像上绘制检测结果"""
    for detection in detections:
        corners = detection['corners']
        conf = detection['confidence']
        # 绘制检测框
        cv2.polylines(image, [corners.reshape(-1, 2)], True, (0, 255, 0), 2)
        # 计算框的左上角位置
        left_top = (corners[0], corners[1])
        # 绘制文字
        text_position = (left_top[0] + text_position_offset[0], left_top[1] + text_position_offset[1])
        cv2.putText(image, f"{conf:.2f}", text_position, font, font_scale, text_color, font_thickness)
    return image

def process_image(image_path, output_container):
    """Process a single image and display the results in the specified container."""
    original_image = cv2.imread(image_path)
    output_container.image(original_image, caption='Selected Image.', use_container_width=True)
    output_container.write("Detecting objects...")

    # 加载预训练模型（如果尚未加载）
    global model
    if 'model' not in globals():
        model = YOLO('D:\code\python深度学习\SAR图像目标检测\_train10\weights\last.pt')

    # 进行预测
    results = model(image_path)

    detections = []
    for result in results:
        if not hasattr(result, 'obb') or result.obb is None:
            output_container.write("No OBB found.")
            continue

        obb_boxes = result.obb.xywhr
        total_conf = result.obb.conf

        if obb_boxes is not None and len(obb_boxes) > 0:
            for i, obb in enumerate(obb_boxes):
                conf = total_conf[i].cpu().numpy()
                corners = convert_obb_to_corners(obb)

                # 显示检测结果
                line = f"Corners: {corners.tolist()}, Confidence: {conf:.6f}"
                output_container.write(line)

                # 添加到检测列表以便绘制
                detections.append({'corners': corners, 'confidence': conf})

    # 如果有检测结果，则绘制它们
    if detections:
        annotated_image = draw_detections(original_image.copy(), detections)
        output_container.image(annotated_image, caption='Detected Image.', use_container_width=True)

def main():
    st.title("SAR image ship detection")

    # 创建用于放置按钮和输入控件的容器
    input_container = st.container()
    # 创建用于放置输出结果的容器
    output_container = st.container()

    with input_container:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        folder_path = st.text_input('Enter folder path to process all images:', '')

        if st.button('Process Single Image') and uploaded_file is not None:
            image_location = 'temp_image.jpg'
            with open(image_location, "wb") as f:
                f.write(uploaded_file.getbuffer())
            process_image(image_location, output_container)
            os.remove(image_location)

        if st.button('Process Folder Images') and folder_path:
            if not os.path.isdir(folder_path):
                output_container.error(f"The provided path '{folder_path}' is not a valid directory.")
            else:
                output_container.write("Processing images in the folder:")
                progress_bar = output_container.progress(0)
                image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                total_images = len(image_files)
                for index, image_name in enumerate(image_files, start=1):
                    image_path = os.path.join(folder_path, image_name)
                    output_container.write(f"Processing image: {image_name}")
                    process_image(image_path, output_container)
                    progress_bar.progress(index / total_images if total_images > 0 else 1)

if __name__ == '__main__':
    main()