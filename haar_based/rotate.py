import cv2
import numpy as np
import dlib

# 加载Dlib的面部检测器和关键点预测模型
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 读取图像
img = cv2.imread('face.png')
# img = cv2.imread('face2.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 计算旋转后的图像大小，以防止裁剪
def rotate_image(image, angle):
    # 获取图像中心
    center = (image.shape[1] // 2, image.shape[0] // 2)
    
    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 计算旋转后的图像尺寸
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    
    # 计算旋转后的图像尺寸
    new_w = int(image.shape[0] * abs_sin + image.shape[1] * abs_cos)
    new_h = int(image.shape[0] * abs_cos + image.shape[1] * abs_sin)
    
    # 更新旋转矩阵中的平移部分，使图像完全显示
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # 旋转图像并调整大小
    rotated_img = cv2.warpAffine(image, M, (new_w, new_h))
    return rotated_img

# 检测多个角度的人脸
def detect_face_in_rotated_images(image, detector, predictor):
    for angle in range(0, 360, 45):  # 从0到360，按5度旋转
        rotated_img = rotate_image(image, angle)  # 使用自定义旋转函数
        gray_rotated = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
        
        faces = detector(gray_rotated)
        
        if len(faces) > 0:
            for face in faces:
                shape = predictor(gray_rotated, face)
                
                # 获取左眼和右眼的中心点
                left_eye = (shape.part(36).x, shape.part(36).y)
                right_eye = (shape.part(45).x, shape.part(45).y)
                
                # 获取嘴巴的位置（这里只取嘴巴的底部中央点）
                mouth_center = (shape.part(54).x, shape.part(54).y)
               
                # 计算眼睛的角度
                dx = right_eye[0] - left_eye[0]
                dy = right_eye[1] - left_eye[1]
                eye_angle = np.degrees(np.arctan2(dy, dx))  # 计算旋转角度
                
                # 获取旋转矩阵并旋转图像，校正眼睛角度
                center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
                M = cv2.getRotationMatrix2D(center, eye_angle, 1)
                rotated_img_final = cv2.warpAffine(rotated_img, M, (rotated_img.shape[1], rotated_img.shape[0]))
                
                # 检查嘴巴是否在眼睛下方，如果不在，则旋转180度
                if mouth_center[1] < (left_eye[1] + right_eye[1]) // 2:
                    print("检测到上下颠倒，进行修正")
                    rotated_img_final = cv2.flip(rotated_img_final, 0)  # 颠倒图像

                # 显示旋转后的图像
                cv2.imshow(f"Rotated Image at {angle} degrees", rotated_img_final)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                return rotated_img_final  # 返回已旋转并对齐的图像
            
    print("没有检测到人脸")
    return None

# 调用函数检测并旋转图像
detect_face_in_rotated_images(img, detector, predictor)
