import torch
import torch.nn as n
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import numpy as np
import os
from sklearn.decomposition import PCA as Sklearn_PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# XGBClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据（X 和 y 已经是 torch 张量）
X_train = torch.load("Vidface_three_dimension/arcface_F_train.pth").to(device)  # torch.Size([2332, 512])
y_train = torch.load("Vidface_three_dimension/arcface_y_train.pth").to(device) # torch.Size([2332])

X_test = torch.load("Vidface_three_dimension/arcface_F_test.pth").to(device)  # torch.Size([5004, 512])
y_test = torch.load("Vidface_three_dimension/arcface_y_test.pth").to(device)  # torch.Size([5004])


# 变成nb
X_train = X_train.cpu().numpy()
y_train = y_train.cpu().numpy()
X_test = X_test.cpu().numpy()
y_test = y_test.cpu().numpy()

# PCA

component = 26

# sklearn库中的PCA进行特征提取
pca = Sklearn_PCA(n_components=component)


X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

################################### 分类器 ##################################
# # KNN训练和识别
# knn = KNeighborsClassifier()
# knn.fit(X_train, y_train)
# predictions = knn.predict(X_test)

# # 高斯朴素贝叶斯分类器
# nb = GaussianNB()
# nb.fit(X_train, y_train)
# predictions = nb.predict(X_test)

#支持向量机
# svm = SVC(kernel='rbf',gamma='scale')
# svm.fit(X_train,y_train)
# predictions = svm.predict(X_test)

# # # # 随机森林
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

# XGBoost
# xgb = XGBClassifier()
# y_test = y_test.astype(int)
# y_train = y_train.astype(int)
# xgb.fit(X_train, y_train)
# predictions = xgb.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy is {accuracy*100:.2f}")



# 可视化特征，t-SNE 降维, 二维可视化，不同的颜色代表不同的类别
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

X_embedded = TSNE(n_components=2).fit_transform(X_test)

plt.figure(figsize=(10, 10))

for i in range(120):
    plt.scatter(X_embedded[y_test == i, 0], X_embedded[y_test == i, 1], label=str(i))

# plt.legend()
# 保存图片
plt.savefig("arcface_tsne_test.png")