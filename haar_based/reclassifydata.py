import os
import shutil

# 定义源文件夹和目标文件夹
source_folder = "samples"
train_folder = "train_samples"
test_folder = "test_samples"

# 创建目标文件夹
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# 遍历0~9的文件夹
for i in range(10):
    os.makedirs(os.path.join(train_folder, str(i)), exist_ok=True)
    os.makedirs(os.path.join(test_folder, str(i)), exist_ok=True)
    
    # 获取当前文件夹中的所有文件
    files = os.listdir(os.path.join(source_folder, str(i)))
    
    # 按序号分配文件到训练集和测试集
    for j, file in enumerate(files):
        # 将文件名转换为整数
        file_index = int(file.split('.')[0])  # 假设文件名格式为 "000X.ext"
        if file_index % 2 == 0:  # 偶数文件放入测试集
            shutil.copy(os.path.join(source_folder, str(i), file), os.path.join(test_folder, str(i), file))
        else:  # 奇数文件放入训练集
            shutil.copy(os.path.join(source_folder, str(i), file), os.path.join(train_folder, str(i), file))
