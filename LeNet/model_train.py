import copy
import time

import pandas as pd
from torchvision.datasets import FashionMNIST
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
from model import LeNet

def train_val_data_process():
    train_data = FashionMNIST(root='D:\\布匹项目\\LeNet\\data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=False)

    train_data, val_data = Data.random_split(train_data,[round(0.8*len(train_data)), round(0.2*len(train_data))])
    train_dataloader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=8)
    val_dataloader = Data.DataLoader(dataset=val_data, batch_size=128, shuffle=True, num_workers=8)
    return train_dataloader, val_dataloader

train_dataloader, val_dataloader = train_val_data_process()

def train_model_process(model,train_dataloader,val_dataloader,num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用Adam优化器，学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 将模型移动到指定设备
    model = model.to(device)
    # 深拷贝模型参数
    best_model_wts = copy.deepcopy(model.state_dict())
    # 初始化最佳模型参数
    # 初始化最佳准确率
    best_acc = 0.0
    # 初始化训练损失
    train_loss_all =[]
    # 初始化训练准确率
    train_acc_all =[]
    # 初始化验证损失
    val_loss_all =[]
    # 初始化验证准确率
    val_acc_all =[]
    # 记录训练时间
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        #初始化参数
        train_loss=0.0
        train_acc=0.0

        val_loss=0.0
        val_acc=0.0

        train_num = 0
        val_num = 0

        # 训练阶段
        for step,(b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 设置模型为训练模式
            model.train()
            # 前向传播，计算输出
            output = model(b_x)
            # 计算预测标签
            pre_label = torch.argmax(output, dim=1)
            # 计算损失
            loss= criterion(output, b_y)
            # 将梯度清零
            optimizer.zero_grad()
            # 反向传播，计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()
            # 累加损失
            train_loss += loss.item() * b_x.size(0)
            # 累加准确率
            train_acc += torch.sum(pre_label == b_y)
            # 累加样本数
            train_num += b_x.size(0)

        for step,(b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 设置模型为验证模式
            model.eval()
            # 前向传播，计算输出
            output = model(b_x)
            # 计算预测标签
            pre_label = torch.argmax(output, dim=1)
            # 计算损失
            loss= criterion(output, b_y)
            # 累加损失
            val_loss += loss.item() * b_x.size(0)
            # 累加准确率
            val_acc += torch.sum(pre_label == b_y)
            # 累加样本数
            val_num += b_x.size(0)

        # 记录训练损失
        train_loss_all.append(train_loss/train_num)
        # 记录训练准确率
        train_acc_all.append(train_acc.double()/train_num)
        # 记录验证损失
        val_loss_all.append(val_loss/val_num)
        # 记录验证准确率
        val_acc_all.append(val_acc.double()/val_num)
        print('Train Loss: {:.4f}, Train Acc: {:.4f},\nVal Loss: {:.4f}, Val Acc: {:.4f}'.format(
            train_loss_all[-1], train_acc_all[-1], val_loss_all[-1], val_acc_all[-1]))

        # 如果验证准确率提高，则更新最佳模型参数
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())  # 更新最佳模型参数
        # 打印训练时间
        time_use = time.time() - since
        print('Time elapsed: {:.0f}m {:.0f}s'.format(time_use // 60, time_use % 60))

    # 选择最佳模型参数
    #加载最高准确率的模型参数
    torch.save(best_model_wts, 'D:\\布匹项目\\LeNet\\best_model.pth')

    train_prcess = pd.DataFrame(data={"epoch": range(num_epochs),
                                      "train_loss": train_loss_all,
                                      "train_acc": train_acc_all,
                                      "val_loss": val_loss_all,
                                      "val_acc": val_acc_all})
    return train_prcess

def matplt_acc_loss(train_prcess):  # 绘制训练过程
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_prcess["train_loss"],'ro-', label="train_loss")
    plt.plot(train_prcess["val_loss"],'bs-', label="val_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1,2,2)
    plt.plot(train_prcess["train_acc"],'ro-', label="train_acc")
    plt.plot(train_prcess["val_acc"],'bs-', label="val_acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")

    plt.show()

if __name__ == '__main__':
    model = LeNet()
    train_dataloader, val_dataloader = train_val_data_process()
    train_prcess = train_model_process(model,train_dataloader,val_dataloader,num_epochs=20)
    matplt_acc_loss(train_prcess)

