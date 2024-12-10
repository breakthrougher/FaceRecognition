import streamlit as st
import os
import time
from PIL import Image
import json

initial_path = "D:/实验报告/大三上/计算机视觉/实验三/code/database/initial"
predictions_path = "D:/实验报告/大三上/计算机视觉/实验三/code/database/predictions"
json_file_path = "D:/results/results.json"
# 设置页面标题
st.title("测试集结果展示")

# 读取 JSON 文件
with open(json_file_path, 'r') as json_file:
    results_dict = json.load(json_file)

# 选择文件夹
folders = [str(i) for i in range(10)]  # 文件夹 0 到 9
selected_folder = st.selectbox("选择文件夹", folders)

# 设置路径
initial_folder = os.path.join(initial_path, selected_folder)
predictions_folder = os.path.join(predictions_path, selected_folder)

# 获取两个子目录中的图片文件名
true_images = sorted(os.listdir(initial_folder))
predictions_images = sorted(os.listdir(predictions_folder))

# 确保两个目录中的文件名相同
common_images = sorted(set(true_images) & set(predictions_images))

# 创建一个状态变量来控制当前显示的图像索引和暂停状态
if 'index' not in st.session_state:
    st.session_state.index = 0
if 'paused' not in st.session_state:
    st.session_state.paused = False

# 创建一个容器用于更新图片
image_container = st.empty()

# 添加暂停/继续按钮
if st.button("继续" if st.session_state.paused else "暂停"):
    st.session_state.paused = not st.session_state.paused

# 自动更新图片
while True:
    if common_images:
        # 只在未暂停状态下更新索引
        if not st.session_state.paused:
            current_image = common_images[st.session_state.index]
            # 更新索引
            if st.session_state.index < len(common_images) - 1:
                st.session_state.index += 1
            else:
                st.session_state.index = 0  # 循环回到开头
        else:
            # 如果处于暂停状态，保持当前图像
            current_image = common_images[st.session_state.index]

        # 加载和显示真实图像
        true_image_path = os.path.join(initial_folder, current_image)
        true_image = Image.open(true_image_path)

        # 加载和显示预测图像
        predictions_image_path = os.path.join(predictions_folder, current_image)
        predictions_image = Image.open(predictions_image_path)

        # 获取当前图像的标签和预测
        index_offset = sum(results_dict["class nums"][:int(selected_folder)])
        idx = common_images.index(current_image)
        true_label = results_dict['True Labels'][idx + index_offset]
        prediction_label = results_dict['Predictions'][idx + index_offset]

        # 更新容器中的内容
        with image_container:
            col1, col2 = st.columns(2)  # 创建左右两列
            with col1:
                st.image(true_image, caption=f"原始图像: {current_image}\n真实标签: {true_label}", use_container_width=True)
            with col2:
                st.image(predictions_image, caption=f"检测结果: {current_image}\n预测标签: {prediction_label}", use_container_width=True)


    time.sleep(1)  # 每1秒更新一次
