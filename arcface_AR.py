# ARface文件夹下，有120个人，每个人有26张图片, 像素是80*100
# 起名是AR{person_id}-{image_id}.tif
# personid是3位整数，AR001-1.tif~AR001-26.tif...AR120-1.tif~AR120-26.tif


# 提取张量torch和标签（彩色，通道是3）
# X.shape = (3120, 100, 80, 3)
# y.shape = (3120,)

import os
import cv2
import torch
import numpy as np
import insightface
from tqdm import tqdm

'''
CUDA_VISIBLE_DEVICES=1 python arcface.py
'''


# 数据集路径
dataset_path = "ARface"

# 加载数据（X 和 y 已经是 torch 张量）
X = torch.load("ARface/X.pth")  # torch.Size([3120, 3, 100, 80])
y = torch.load("ARface/y.pth")  # torch.Size([3120])

# 加载 ArcFace 模型（预训练模型）
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=1)  # ctx_id=0 表示使用 CPU，1 表示使用 GPU

# 提取特征
features = []

for i in tqdm(range(X.shape[0]), desc="Extracting features", ncols=100):
    # 获取图像数据
    image = X[i].numpy().transpose(1, 2, 0)  # 将形状从 (3, 100, 80) 转为 (100, 80, 3)

    # 将图像转换为 RGB 格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 使用 ArcFace 提取人脸特征
    faces = model.get(image_rgb)
    
    if len(faces) > 0:
        # 只取第一个检测到的人脸
        feature = faces[0].embedding
        features.append(feature)
    else:
        # 如果没有检测到人脸，填充一个零向量
        print(f"No face detected in image {i}")

# 将特征转为 PyTorch 张量并保存
features = torch.tensor(features)
torch.save(features, "ARface/features.pth")
print(feature.shape)

print(f"Feature extraction complete. Features saved to ARface/features.pth")
