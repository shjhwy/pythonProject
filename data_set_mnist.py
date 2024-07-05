import paddle
import numpy as np


# 数据载入
class MNISTDataset(paddle.io.Dataset):
  def __init__(self, mode='train'):
    super().__init__()
    self.mnist_data = paddle.vision.datasets.MNIST(mode=mode)

  def __getitem__(self, idx):
    data, label = self.mnist_data[idx]
    data = np.reshape(data, [1, 28, 28]).astype('float32') / 255
    label = np.reshape(label, [1]).astype('int64')
    return (data, label)

  def __len__(self):
    return len(self.mnist_data)


def dataLoader():
    train_loader = paddle.io.DataLoader(MNISTDataset(mode='train'),
                                        batch_size=16,
                                        shuffle=True)

    test_loader = paddle.io.DataLoader(MNISTDataset(mode='test'),
                                       batch_size=16,
                                       shuffle=False)
    return train_loader,test_loader
