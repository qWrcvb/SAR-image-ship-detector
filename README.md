2024秋SCNU AI项目
检测SAR图像中船只的项目

请注意在SAR_ship_detection.py中将以下代码段的路径修改成实际路径再运行：
   # 加载预训练模型（如果尚未加载）
    global model
    if 'model' not in globals():
        model = YOLO('D:\code\python深度学习\SAR图像目标检测\_train10\weights\last.pt')
