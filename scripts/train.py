import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from libcode.model.openllama import *
from libcode.model.lmoe import *



# 准备数据集
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 创建模型实例
model = LungsCancerMoE(args)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
criterion1 = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    for images, inputs in train_loader:
        # 前向传播
        gen_acc, loss, cls,label_ids, output_ids= model(images)
        loss0 = criterion(cls, inputs['cls'])
        loss1 = criterion1(label_ids, output_ids)
        loss_all = loss + loss0 + loss1
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print('Training complete')
