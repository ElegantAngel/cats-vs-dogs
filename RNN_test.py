import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
device = torch.device('cuda:0')


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((150, 150)),  # 调整图像大小
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # 转化为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 像素值标准化
])

# 读取数据集
# 数据准备
data_dir = 'F:\\pycode\\NLP_task1\\data1'  # 数据集路径
batch_size = 32
accum_steps=4


train_dataset = datasets.ImageFolder(
    os.path.join(data_dir, 'train'),
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

val_dataset = datasets.ImageFolder(
    os.path.join(data_dir, 'val'),
    transform=transform
)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

dataset = ImageFolder(data_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# 创建RNN模型
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


# 设置模型参数
input_size = 150 * 150 * 3  # 输入特征数，将图像展平为一维
hidden_size = 128  # 隐层大小
num_classes = 2  # 猫狗分类

model = RNNModel(input_size, hidden_size, num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 创建学习率调度器
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    #在每个epoch中更新学习率
    lr_scheduler.step()

    i = 0
    for images, labels in train_loader:
        images = images.view(-1, 1, 150 * 150 * 3).to(device)  # 将图像展平为一维
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        i = i + 1
        #print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
        print(f'Epoch [{epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}')

# 模型评估
torch.save(model,'RNN_test.model')
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images = images.view(-1, 1, 150 * 150 * 3).to(device)  # 将图像展平为一维
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")




