import torch.optim
import torch
from torch import nn
import scipy.io as scio
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from model import *

num_epoch = 3
net = nn.Sequential(
    nn.Conv1d(60, 10, kernel_size=3, stride=4, padding=1),
    # nn.BatchNorm1d(626),
    nn.ReLU(),
    nn.AvgPool1d(kernel_size=3, stride=2),
    nn.Flatten(start_dim=0),
    nn.Dropout(),
    nn.Linear(3120, 2)
)


net2 = nn.Sequential(
    nn.Linear(60, 30),
    nn.Linear(30, 2),
    nn.Linear(30, 2)
)



def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net2.apply(init_weights)



X = torch.randn(1, 60, 2500)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t',X.shape)

lr, num_epochs = 0.001, 10
# loss = nn.MSELoss()
loss = nn.CrossEntropyLoss()


trainer = torch.optim.Adam(net.parameters(), lr=0.03)
#
#
# data fetch
data_path = 'D:\学习\python_test\Project_Data\Train\sample01.mat'
# data = h5py.File(data_path)
data = scio.loadmat(data_path)

epo = data['epo']
mnt = data['mnt']
trainX = epo['x'][0][0]
trainX = trainX.transpose((2, 1, 0))
trainY = epo['y'][0][0].transpose()
a = len(trainY)

#因为nn.CrossEntropyLoss的target只接受1维，对trainY进行降维
trainY_2d = np.zeros((100,1))
for i in range(len(trainY)):
    if trainY[i,0] ==0:
        trainY_2d[i] = torch.tensor([1])

# def load_array(data_arrays, batch_size, is_train=True):  #@save
#     """构造一个PyTorch数据迭代器"""
#     dataset = data.TensorDataset(*data_arrays)
#     return data.DataLoader(dataset, batch_size, shuffle=is_train)

# batch_size = 10
# data_iter = load_array((features, labels), batch_size)

# 添加tensorboard
writer = SummaryWriter("logs_train")
for epoch in range(num_epochs):
    print("------第 {} 轮训练开始------".format(epoch + 1))
    for i in range(100):
        X = torch.from_numpy(trainX[i]).to(torch.float32).unsqueeze(0)
        y = torch.from_numpy(trainY_2d[i]).to(torch.float32).long()
        # a = net(X).unsqueeze(0)
        l = loss(net(X).unsqueeze(0), y)
        print("loss:",l)
        trainer.zero_grad()
        l.backward()
        trainer.step()


    print("训练次数: {}, Loss: {}".format(100*(i+1), l.item()))
    writer.add_scalar("train_loss", l.item())




