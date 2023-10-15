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
accum_steps=4

# 数据增强和加载
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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


# 构建CNN模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()  # 调用父类构造函数
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512 * 14 * 14, 1024)
        #self.fc1 = nn.Linear(256 * 28 * 28, 512)
        #self.fc2 = nn.Linear(512, 2)
        self.fc2 = nn.Linear(1024, 2)  # 2 classes: cat and dog

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.pool(nn.functional.relu(self.conv4(x)))
        #x = x.view(-1, 256 * 28 * 28)
        x = x.view(-1, 512 * 14 * 14)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 初始化模型
model = CNNModel()
model = model.to(device)  # 将模型移动到GPU，如果可用的话

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# 创建学习率调度器
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    # 在每个epoch中更新学习率
    lr_scheduler.step()

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        #optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        if (i+1) % accum_steps == 0 or (i+1) ==len(train_loader):
            optimizer.step()
            optimizer.zero_grad()


        running_loss += loss.item()
         # 打印每批次的损失
        print(f'Epoch [{epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}')


# 评估模型
torch.save(model,'CNN_test1.model')
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
