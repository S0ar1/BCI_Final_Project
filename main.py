import torch
from torch import nn
import scipy.io as scio
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from train_epoch import *
from read_data import *
# from model import *
from linearNet_525 import *


if __name__ == '__main__':

    batch_size = 2      #bach_size只为1时候可以运行，否则在进入net()时，size mismatch

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    # net = linearNet(60,30,30,2)
    net = nn.Sequential(
        nn.Conv1d(60, 10, kernel_size=3, stride=4, padding=1),
        # nn.BatchNorm1d(626),
        nn.ReLU(),
        nn.AvgPool1d(kernel_size=3, stride=2),
        nn.Flatten(start_dim=0),
        nn.Dropout(),
        nn.Linear(3120, 2)
    )

    net.apply(init_weights)

    # net = CNN()

    lr, num_epochs = 0.001, 10
    # loss = nn.MSELoss()
    loss = nn.CrossEntropyLoss()

    trainer = torch.optim.Adam(net.parameters(), lr=0.03)
    train_iter, valid_iter = read_data(batch_size)

    # 添加tensorboard
    writer = SummaryWriter("logs_train")
    for epoch in range(num_epochs):
        print("------第 {} 轮训练开始------".format(epoch + 1))
        a,b = train_epoch(net, train_iter, loss, trainer)
        print("第{}次训练损失为 {}".format(epoch+1,a))
        print("第{}次训练精度为 {}".format(epoch+1,b))
        # print("训练次数: {}, Loss: {}".format(100*(i+1), l.item()))
        # writer.add_scalar("train_loss", l.item())