import os
import cv2
import numpy as np

# 数据集路径
dataset_path = "train_samples"

X = []
y = []

# 遍历每个文件夹
for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (150, 150))
            # print(img.shape)
            X.append(img)
            y.append(label)

# 转换为numpy数组并保存
X = np.array(X)  # shape: (2400, 150, 150, 3)
y = np.array(y)  # shape: (2400,)

print(f"X.shape: {X.shape}")
print(f"y.shape: {y.shape}")

if not os.path.exists("Vidface_three_dimension"):
    os.makedirs("Vidface_three_dimension")

np.save("Vidface_three_dimension/X_test.npy", X)
np.save("Vidface_three_dimension/y_test.npy", y)

dataset_path = "test_samples"

X = []
y = []

# 遍历每个文件夹
for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (150, 150))
            # print(img.shape)
            X.append(img)
            y.append(label)

# 转换为numpy数组并保存
X = np.array(X)  # shape: (2400, 150, 150, 3)
y = np.array(y)  # shape: (2400,)

print(f"X.shape: {X.shape}")
print(f"y.shape: {y.shape}")

np.save("Vidface_three_dimension/X_train.npy", X)
np.save("Vidface_three_dimension/y_train.npy", y)