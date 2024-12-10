import cv2
import numpy as np
import os
from sklearn.decomposition import PCA as Sklearn_PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

component = 28
Neighbors = 6
scale = 1.1

# 伽马矫正
def gamma_correction(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def adaptive_median_filter(image, max_window_size=7):
    # 获取图像的大小
    rows, cols = image.shape
    # 创建一个输出图像
    output_image = np.copy(image)
    
    # 遍历图像的每一个像素点
    for i in range(rows):
        for j in range(cols):
            # 初始化窗口大小
            window_size = 3
            while window_size <= max_window_size:
                # 计算窗口的边界
                half_window = window_size // 2
                x1, x2 = max(i - half_window, 0), min(i + half_window + 1, rows)
                y1, y2 = max(j - half_window, 0), min(j + half_window + 1, cols)

                # 获取当前窗口的像素值
                window = image[x1:x2, y1:y2]

                # 计算当前窗口的中值
                median_val = np.median(window)
                # 当前像素值
                pixel_val = image[i, j]
                
                # 判断是否是噪声
                if median_val > pixel_val:
                    min_val = np.min(window)
                    max_val = np.max(window)
                else:
                    min_val = median_val
                    max_val = median_val
                
                if min_val < pixel_val < max_val:
                    # 如果当前像素值正常，则不进行更改
                    output_image[i, j] = pixel_val
                    break
                else:
                    # 如果当前像素值是噪声，则使用窗口的中值替换
                    output_image[i, j] = median_val
                    break
                
                # 如果窗口太大，继续增大窗口
                window_size += 2

    return output_image


def adaptive_gaussian_filter(image, max_window_size=7, k=2.0):
    """
    自适应高斯滤波
    :param image: 输入图像（灰度图）
    :param max_window_size: 滤波器的最大窗口大小
    :param k: 用于计算局部标准差的系数
    :return: 去噪后的图像
    """
    rows, cols = image.shape
    output_image = np.copy(image)
    
    # 遍历图像的每一个像素点
    for i in range(rows):
        for j in range(cols):
            # 初始化窗口大小
            window_size = 3
            while window_size <= max_window_size:
                # 计算窗口的边界
                half_window = window_size // 2
                x1, x2 = max(i - half_window, 0), min(i + half_window + 1, rows)
                y1, y2 = max(j - half_window, 0), min(j + half_window + 1, cols)
                
                # 获取当前窗口的像素值
                window = image[x1:x2, y1:y2]

                # 计算窗口内的局部标准差（σ）
                local_std = np.std(window)
                
                # 根据局部标准差调整高斯滤波器的标准差（σ）
                sigma = k * local_std
                
                # 使用高斯滤波器平滑窗口
                gaussian_kernel = cv2.getGaussianKernel(window_size, sigma)
                gaussian_kernel = gaussian_kernel @ gaussian_kernel.T  # 转为2D
                smoothed_window = cv2.filter2D(window, -1, gaussian_kernel)
                
                # 计算当前像素的新值，通常是窗口的中心值
                output_image[i, j] = smoothed_window[half_window, half_window]
                
                # 如果窗口已经足够大且标准差较大，则退出
                if window_size == max_window_size:
                    break
                
                # 增大窗口
                window_size += 2

    # 输出超参数
    print("sigma = ", sigma)
    print("window_size = ", window_size)

    return output_image

#########################################  主函数  #########################################

# 加载人脸检测模型
face_mode = cv2.CascadeClassifier("ckpt/haarcascade_frontalface_alt.xml")

img_path = "1.png"


img = cv2.imread(img_path)
# 灰度
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 对人脸增加难度：降低对比度、增加高斯噪声，增加椒盐噪声，然后可视化：

# 降低对比度
img = gamma_correction(img, 0.5)
# 直方图均衡化
# img = cv2.equalizeHist(img)


# 高斯噪声
gaussian_noise = np.zeros(img.shape, dtype=np.uint8)
cv2.randn(gaussian_noise, 0, 20)
img = cv2.add(img, gaussian_noise)

# 对img进行自适应高斯滤波
# img = adaptive_gaussian_filter(img)
# cv的
# img = cv2.GaussianBlur(img, (7, 7), 30)


# 增加椒盐噪声
noise = np.random.randint(0, 255, img.shape)
img[noise < 10] = 0
img[noise > 250] = 255

# 对img进行中值滤波
# img = cv2.medianBlur(img, 5)
# # img = adaptive_median_filter(img)

# img = cv2.medianBlur(img, 5)
# img = cv2.GaussianBlur(img, (7, 7), 30)
# # img = adaptive_gaussian_filter(img)
# # 伽马
# img = gamma_correction(img, 2)
# # img = cv2.equalizeHist(img)

#
# img = cv2.medianBlur(img, 5)
# img = cv2.GaussianBlur(img, (7, 7), 30)
# # img = adaptive_gaussian_filter(img)
# # 伽马
# img = gamma_correction(img, 2)
# # img = cv2.equalizeHist(img)


cv2.imshow("img", img)

# 保存图片
cv2.imwrite("1_pro.png", img)

cv2.waitKey(0)