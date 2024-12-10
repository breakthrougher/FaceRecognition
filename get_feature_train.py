import cv2
import numpy as np
import os
from sklearn.decomposition import PCA as Sklearn_PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from tqdm import tqdm

component = 28
Neighbors = 6
scale = 1.1

# 伽马矫正
def gamma_correction(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# 数据格式为按列
def PCA(X_train, components):
    # Prepare
    col = X_train.shape[1]
    # axis=0 表示操作是针对行的方向，但实际上是在处理特征（列）
    Mean = np.mean(X_train, axis=1)
    # X = X_train - Mean
    # 减去每个特征的均值,使得每个特征的均值均为0
    X = X_train - Mean[:, np.newaxis]
    # Calculate eig_vectors, eig_values
    cov = np.dot(X, X.T) / col
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # 确保特征值、特征向量为实数
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # 对特征值、特征向量进行排序
    index = np.argsort(eigenvalues)[::-1]
    # 大小为样本数量*样本数量
    W = eigenvectors[:, index]

    # Build model
    model = {
        'mean': Mean,
        'W': W[:, :components],
        'components': components
    }

    return model


#########################################  主函数  #########################################

# 加载人脸检测模型
# face_mode = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
face_mode = cv2.CascadeClassifier("ckpt/haarcascade_frontalface_alt.xml")

# 训练集和测试集
train_path = "train_samples"
train_images = []
train_labels = []
train_num = 0

# 人脸检测
# 获取训练集总文件数
total_files = sum([len(files) for _, _, files in os.walk(train_path)])

# 添加进度条
for label in tqdm(os.listdir(train_path), desc="Processing labels", total=len(os.listdir(train_path))):
    label_path = os.path.join(train_path, label)
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        img = cv2.imread(img_path)

        if img is not None:
            # 转为灰度图
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_mode.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=6)

            # 二次探查
            if len(faces) == 0:
                blurred_image = cv2.GaussianBlur(gray_img, (3, 3), 0)
                faces = face_mode.detectMultiScale(blurred_image, scaleFactor=1.1, minNeighbors=6)
                # 三次探查
                if len(faces) == 0:
                    histed_image = cv2.equalizeHist(blurred_image)
                    faces = face_mode.detectMultiScale(histed_image, scaleFactor=1.1, minNeighbors=6)
                    # 四次探查
                    if len(faces) == 0:
                        gamma_image = gamma_correction(histed_image, 1.5)
                        faces = face_mode.detectMultiScale(gamma_image, scaleFactor=1.1, minNeighbors=6)

            for (x, y, w, h) in faces:
                face = gray_img[y:y+h, x:x+w]
                print(face.shape)
                face_sized = cv2.resize(face, (150, 150))
                train_images.append(face_sized)
                train_labels.append(label)

        train_num += 1

# 将训练数据转为numpy数组
X_train = np.array(train_images)
y_train = np.array(train_labels)

# print("Train samples shape:", X_train.shape)
# print("Train labels shape:", y_train.shape)

print(f"Detected Train faces num are: {X_train.shape[0]}/{train_num}")

# 保存到文件
np.save("data_cache/X_train.npy", X_train)
np.save("data_cache/y_train.npy", y_train)
