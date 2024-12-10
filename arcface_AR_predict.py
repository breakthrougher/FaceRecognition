import torch
import torch.nn as n
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
import numpy as np
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
X = torch.load("ARface/features.pth").to(device)
y = torch.load("ARface/y.pth").to(device)

# 转换为 NumPy 数组
X = X.cpu().numpy()  # 转回 CPU 并转换为 NumPy 数组
y = y.cpu().numpy()

# 三折交叉验证
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
accuracy_list = []

for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # PCA 降维
    pca = Sklearn_PCA(n_components=28)  # 选择降维后的维度
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # KNN分类器
    knn = KNeighborsClassifier()
    knn.fit(X_train_pca, y_train)
    y_pred = knn.predict(X_test_pca)

    # # 高斯朴素贝叶斯分类器
    # nb = GaussianNB()
    # nb.fit(X_train, y_train)
    # y_pred = nb.predict(X_test)

    #支持向量机
    # svm = SVC(kernel='rbf',gamma='scale')
    # svm.fit(X_train,y_train)
    # y_pred = svm.predict(X_test)

    # 随机森林
    #rf = RandomForestClassifier()
    #rf.fit(X_train, y_train)
    #y_pre = rf.predict(X_test)

    # XGBoost
    # xgb = XGBClassifier()
    # y_test = y_test.astype(int)
    # y_train = y_train.astype(int)
    # xgb.fit(X_train, y_train)
    # y_pred = xgb.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)
    
    print(f"Fold {fold + 1}, Accuracy: {accuracy * 100:.2f}%")

# 输出平均准确率
average_accuracy = np.mean(accuracy_list)
print(f"Average Accuracy: {average_accuracy * 100:.2f}%")