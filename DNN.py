#导入需要的库
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
device = torch.device('cuda:0')

# 数据准备
data_dir = 'F:\\pycode\\NLP_task1\\data'  # 数据集路径
batch_size = 32
accum_steps = 4

data_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = datasets.ImageFolder(
    os.path.join(data_dir, 'train'),
    transform=data_transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

val_dataset = datasets.ImageFolder(
    os.path.join(data_dir, 'val'),
    transform=data_transform
)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

#定义模型
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(128 * 128 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)  # 添加一个新的隐藏层
        self.fc4 = nn.Linear(64, 32)   # 添加另一个新的隐藏层
        self.fc5 = nn.Linear(32, 2)    # 输出层

    def forward(self, x):
        x = x.view(-1, 128 * 128 * 3)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))  # 应用ReLU激活函数
        x = torch.relu(self.fc4(x))  # 添加额外的隐藏层并应用ReLU
        x = self.fc5(x)  # 输出层
        return x

#gpu
model = DNN()
model = model.to(device)

#损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 创建学习率调度器
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#训练
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    # 在每个epoch中更新学习率
    lr_scheduler.step()

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        if (i+1) % accum_steps == 0 or (i+1) ==len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
        optimizer.step()
        running_loss += loss.item()
        # 打印每批次的损失
        print(f'Epoch [{epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}')


#模型评估和预测
torch.save(model,'DNN_test.model')
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in val_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on validation set: {100 * correct / total:.2f}%')

