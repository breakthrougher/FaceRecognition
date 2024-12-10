import cv2
import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms, models
from PIL import Image

'''
CUDA_VISIBLE_DEVICES=1 python haar_based/05. DCN.py
'''
# 数据增强操作
train_transform = transforms.Compose([
    transforms.Lambda(lambda x: transforms.ToPILImage()(x)),  # 将Tensor转换为PIL.Image
    transforms.RandomHorizontalFlip(),           # 随机水平翻转
    transforms.RandomRotation(10),               # 随机旋转角度
    transforms.RandomResizedCrop(50, scale=(0.8, 1.0)),  # 随机裁剪并缩放图像
    transforms.Resize(224),  # 适应DenseNet输入尺寸
    transforms.ToTensor(),                       # 转换为Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化，适应三通道
])

test_transform = transforms.Compose([
    transforms.Lambda(lambda x: transforms.ToPILImage()(x)),  # 将Tensor转换为PIL.Image
    transforms.Resize(224),  # 适应DenseNet输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 测试集可以不做数据增强，主要做归一化
])

# 图像数据，先用CNN卷积，再用XGBoost分类，不用数据预处理
X_train = np.load("Vidface_one_dimension/X_train.npy") # (2178, 50, 50)
y_train = np.load("Vidface_one_dimension/y_train.npy") # (2178,)

X_test = np.load("Vidface_one_dimension/X_test.npy") # (3898, 50, 50)
y_test = np.load("Vidface_one_dimension/y_test.npy") # (3898,)

# 检查数据形状
print("Train samples shape:", X_train.shape)

# 使用 LabelEncoder 对标签进行编码
label_encoder = LabelEncoder()

# 将标签从字符串类型转换为整数类型
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# 转换为 PyTorch Tensor 并调整形状 (N, 3, H, W)
X_train_tensor = torch.tensor(np.repeat(X_train[:, np.newaxis, :, :], 3, axis=1), dtype=torch.float32)  # 转换为三通道
X_test_tensor = torch.tensor(np.repeat(X_test[:, np.newaxis, :, :], 3, axis=1), dtype=torch.float32)  # 转换为三通道
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 使用数据增强并创建TensorDataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# 创建 DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 使用预训练的 DenseNet121
class DenseNet121Model(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNet121Model, self).__init__()
        
        # 使用DenseNet121作为特征提取器
        self.densenet = models.densenet121(pretrained=True)
        
        # 修改最后的全连接层以适应我们的分类任务
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)

    def forward(self, x):
        return self.densenet(x)

# 模型实例化
model = DenseNet121Model(num_classes=10)  # 假设有10个类别

# 训练，预测，然后输出ACC
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 适用于分类问题
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # 计算准确率
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # 输出每个 epoch 的训练损失和准确率
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

# 测试模型
model.eval()
correct = 0
total = 0
all_preds = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())

# 计算准确率
test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# 可选：如果需要精确度、召回率等详细指标
from sklearn.metrics import classification_report
print(classification_report(y_test, all_preds))
