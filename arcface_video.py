import os
import cv2
import torch
import numpy as np
import insightface
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据（X 和 y 已经是 torch 张量）
X = np.load("Vidface_three_dimension/X_train.npy")  # torch.Size([3120, 150, 150,3])
y_train = np.load("Vidface_three_dimension/y_train.npy")  # torch.Size([3120])

X_test = np.load("Vidface_three_dimension/X_test.npy")  # torch.Size([3120, 150, 150,3])
y_test = np.load("Vidface_three_dimension/y_test.npy")  # torch.Size([3120])

# 转到设备cuda上
X_train = torch.tensor(X)
y_train = torch.tensor(y_train)

X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)

# 移动到cuda上
X_train = X_train.to(device)
y_train = y_train.to(device)

X_test = X_test.to(device)
y_test = y_test.to(device)

# 加载 ArcFace 模型（预训练模型）
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=1)  # ctx_id=0 表示使用 CPU，1 表示使用 GPU

# 提取特征
features = []
y = []

X = X_train

X = X.to(device)

img_num = 0
for i in tqdm(range(X.shape[0]), desc="提取特征", ncols=100):
    # 获取图像数据

    image = X[i].cpu().numpy()  # 转回 numpy 数组
    # image = X[i]  # 直接使用numpy数组
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

    # 使用 ArcFace 提取人脸特征
    faces = model.get(image_rgb)
    
    if len(faces) > 0:
        # 只取第一个检测到的人脸
        feature = faces[0].embedding
        features.append(feature)
        y.append(y_test[i])
    else:
        # 如果没有检测到人脸，填充一个零向量
        print(f"第{i}张图像未检测到人脸")
        img_num += 1

print(f"检测到人脸共有 {X.shape[0] - img_num} / {X.shape[0]}")


# 将特征转为 PyTorch 张量并保存
features = torch.tensor(features)
y = torch.tensor(y)
# torch.save(features, "Vidface_three_dimension/arcface_F_test.pth")
# torch.save(y, "Vidface_three_dimension/arcface_y_test.pth")