import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # 导入tqdm库

# 用RES101作为基础模型，进行预测（三折交叉验证）

# 用gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task = "googlenet"

if task == "resnet":
    model = models.resnet101(pretrained=True)
elif task == "vgg":
    model = models.vgg19_bn(pretrained=True)  # 修改为vgg16模型
    # 修改最后一层以适应120个类别
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 120)  
elif task == "densenet":
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 120)  # 修改最后一层以适应120个类别
elif task == "googlenet":
    model = models.googlenet(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 120)  # 修改最后一层以适应120个类别

# X.shape: torch.Size([3120, 100, 80, 3])
# y.shape: torch.Size([3120])

X = torch.load("ARface/X.pth") # torch.Size([3120, 100, 80, 3])
X = F.interpolate(X, size=(224, 224)) # torch.Size([3120, 224, 224, 3])
y = torch.load("ARface/y.pth") # torch.Size([3120]) # 1~120的标签

print(f"X.shape: {X.shape}")
print(f"y.shape: {y.shape}")

# 数据集准备
dataset = TensorDataset(X, y)

# 将数据移到cuda上
y = y.to(device)

acc_av = []
# 三折交叉验证
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(skf.split(X.cpu(), y.cpu())):  # 将数据移到cpu上进行分割
    train_dataset = torch.utils.data.Subset(dataset, train_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 将模型移到cuda上
    model = model.to(device)

    # 训练模型
    model.train()
    for epoch in range(10):  # 训练10个epoch
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):  # 添加进度条
            inputs, labels = inputs.to(device), labels.to(device)  # 将输入和标签移到cuda上
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # 测试模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):  # 添加进度条
            inputs, labels = inputs.to(device), labels.to(device)  # 将输入和标签移到cuda上
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Fold {fold + 1}, Accuracy: {100 * correct / total:.2f}%')
    acc_av.append(100 * correct / total)

# 输出平均
print(acc_av)
print(f'Average Accuracy: {sum(acc_av) / len(acc_av):.2f}%')

