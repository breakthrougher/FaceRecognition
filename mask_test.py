from sklearn.decomposition import PCA
import numpy as np
import cv2

# 示例函数：使用PCA对图像进行线性表示
def linear_representation(image, n_components=50):
    # 将图像展平
    image_flattened = image.flatten().reshape(1, -1)
    
    # 使用PCA来进行线性表示
    pca = PCA(n_components=n_components)
    pca.fit(image_flattened)
    
    # 获取线性表示
    image_reconstructed = pca.inverse_transform(pca.transform(image_flattened))
    
    # 计算重建误差（残差）
    residual = np.linalg.norm(image_flattened - image_reconstructed)
    
    return residual, image_reconstructed.reshape(image.shape)

# 示例：判断图像是否存在遮挡
def detect_occlusion(image, threshold=100):
    residual, reconstructed_image = linear_representation(image)
    print(f"残差值：{residual}")
    
    if residual > threshold:
        print("检测到遮挡")
    else:
        print("无遮挡")

# 示例图像处理
image = cv2.imread('image_path', cv2.IMREAD_GRAYSCALE)  # 加载灰度图像
detect_occlusion(image)
