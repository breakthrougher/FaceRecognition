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

# 数据集路径
dataset_path = "ARface"

# 初始化列表存储图像数据和标签
images = []
labels = []

# 遍历ARface文件夹
for file_name in os.listdir(dataset_path):
    if file_name.endswith(".tif"):
        # 提取person_id作为标签
        person_id = int(file_name.split("-")[0][2:])
        
        # 读取图像并转为彩色图
        img_path = os.path.join(dataset_path, file_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        print(img.shape)
        # 检查图像尺寸是否符合要求
        if img.shape == (100, 80, 3):
            images.append(img)
            labels.append(person_id)

# 转换为张量
X = torch.tensor(np.array(images), dtype=torch.float32)  # 图像数据
y = torch.tensor(np.array(labels), dtype=torch.int64)    # 标签

# 调整顺序 3120,3,100,80
X = X.permute(0, 3, 1, 2)
# 标签减1
y = y - 1

# 检查形状
print(f"X.shape: {X.shape}")
print(f"y.shape: {y.shape}")

# 保存到ARface下，保存为张量pth
torch.save(X, "ARface/X.pth")
torch.save(y, "ARface/y.pth")

# import torch
#
# a = torch.load('ARface/features.pth')
# print(a.shape)  # torch.Size([3120, 512])