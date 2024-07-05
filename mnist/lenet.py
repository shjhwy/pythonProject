import os
import random

import numpy as np
# 加载飞桨相关库
import paddle
from paddle.nn import Conv2D, MaxPool2D, Linear
import paddle.nn.functional as F

# 数据载入
class MNISTDataset():
  def __init__(self, mode='train'):
    self.mnist_data = paddle.vision.datasets.MNIST(mode=mode)

  def __getitem__(self, idx):
    data, label = self.mnist_data[idx]
    data = np.reshape(data, [1, 28, 28]).astype('float32') / 255
    label = np.reshape(label, [1]).astype('int64')
    return (data, label)

  def __len__(self):
    return len(self.mnist_data)

train_loader = paddle.io.DataLoader(MNISTDataset(mode='train'),
                                    batch_size=16,
                                    shuffle=True)

test_loader = paddle.io.DataLoader(MNISTDataset(mode='test'),
                                    batch_size=16,
                                    shuffle=False)

# 定义 mnist 数据识别网络模型结构
class MNIST(paddle.nn.Layer):
     def __init__(self):
         super().__init__()

         # 定义卷积层，输出特征通道 out_channels 设置为 20，卷积核的大小 kernel_size 为 5，卷积步长 stride=1，padding=2
         self.conv1 = Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
         # 定义池化层，池化核的大小 kernel_size 为 2，池化步长为 2
         self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
         # 定义卷积层，输出特征通道 out_channels 设置为 20，卷积核的大小 kernel_size 为 5，卷积步长 stride=1，padding=2
         self.conv2 = Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
         # 定义池化层，池化核的大小 kernel_size 为 2，池化步长为 2
         self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
         # 定义一层全连接层，输出维度是 10
         self.fc = Linear(in_features=980, out_features=10)

   # 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
   # 卷积层激活函数使用 Relu，全连接层激活函数使用 softmax
     def forward(self, inputs):
         x = self.conv1(inputs)
         x = F.relu(x)
         x = self.max_pool1(x)
         x = self.conv2(x)
         x = F.relu(x)
         x = self.max_pool2(x)
         x = paddle.reshape(x, [x.shape[0], -1])
         x = self.fc(x)
         return x

#创建模型
model = MNIST()


#设置优化器
opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
EPOCH_NUM = 10
for epoch_id in range(EPOCH_NUM):
    model.train()
    for batch_id, data in enumerate(train_loader()):
        #准备数据
        images, labels = data

        #前向计算的过程
        predicts = model(images)

        #计算损失，取一个批次样本损失的平均值
        loss = F.cross_entropy(predicts, labels)
        avg_loss = paddle.mean(loss)

        #每训练了 100 批次的数据，打印下当前 Loss 的情况
        if batch_id % 200 == 0:
            print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))

        #后向传播，更新参数的过程
        avg_loss.backward()
        # 最小化 loss,更新参数
        opt.step()
        # 清除梯度
        opt.clear_grad()

    # evaluate model after one epoch
    model.eval()
    accuracies = []
    losses = []
    for batch_id, data in enumerate(test_loader):
        #准备数据
        images, labels = data
        #前向计算的过程
        predicts = model(images)
        #计算损失
        loss = F.cross_entropy(predicts, labels)
        #计算准确率
        acc = paddle.metric.accuracy(predicts, labels)
        accuracies.append(acc.numpy())
        losses.append(loss.numpy())

    avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
    print("[validation]After epoch {}: accuracy/loss: {}/{}".format(epoch_id, avg_acc, avg_loss))

#保存模型参数
paddle.save(model.state_dict(), 'mnist.pdparams')
