import torch
from torch import nn
import scipy.io as scio
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import plot
import net_template
from train_epoch import *
from read_data import *
# from model import *
# from linearNet_525 import *


def init_param():
    """
    init
    learning rate
    epoch
    loss function
    optimizer
    :return:
    """
    lr, num_epochs = 0.001, 5
    loss = nn.CrossEntropyLoss()

    trainer = torch.optim.Adam(net.parameters(), lr=0.03)
    return lr, num_epochs, loss, trainer

if __name__ == '__main__':

    batch_size = 20      #bach_size只为1时候可以运行，否则在进入net()时，size mismatch

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
    net = net_template.linearNet()

    net.apply(init_weights)
    lr, num_epochs, loss, trainer = init_param()
    train_iter, valid_iter = read_data(batch_size)

    writer = SummaryWriter("logs_train")
    loss_list = []
    acc_list = []
    valid_acc_list = []
    for index in range(5):
        print("------第 {} 人训练开始------".format(index + 1))
        loss_single_list = []
        acc_single_list = []
        valid_single_list = []
        for epoch in range(num_epochs):
            print("------第 {} 轮训练开始------".format(epoch + 1))
            a, b = train_epoch(net, train_iter[index], loss, trainer)
            print("第{}次训练损失为 {}".format(epoch+1,a))
            print("第{}次训练精度为 {}".format(epoch+1,b))
            loss_single_list.append(a)
            acc_single_list.append(b)
            valid_acc = evaluate_accuracy(net, valid_iter[index])
            valid_single_list.append(valid_acc)
        loss_list.append(loss_single_list)
        acc_list.append(acc_single_list)
        valid_acc_list.append(valid_single_list)
    plot.plt_loss(loss_list)
    plot.plt_train_acc(acc_list)
    plot.plt_valid_acc(valid_acc_list)
        # print("训练次数: {}, Loss: {}".format(100*(i+1), l.item()))
        # writer.add_scalar("train_loss", l.item())

