import torch
from torch import nn
import scipy.io as scio
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import plot
import net_template
from train_epoch_EEGNet import *
# from read_data import *
from read_data_merge import *        #此处用sample15个人做训练数据，只用一个sample做测试数据
# from read_data_test import  *
from model_EEGNet import *
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
    lr, num_epochs = 0.0005, 200
    loss = nn.CrossEntropyLoss().to(DEVICE)
    net = EEGNet(classes_num=2).to(DEVICE)
    # trainer = torch.optim.Adam(net.parameters(), lr=0.03)
    trainer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.01)      #【0604】加入正则化
    return net, lr, num_epochs, loss, trainer

if __name__ == '__main__':

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 20
    #bach_size只为1时候可以运行，否则在进入net()时，size mismatch

    # def init_weights(m):
    #     if type(m) == nn.Linear:
    #         nn.init.normal_(m.weight, std=0.01)
    # net = net_template.linearNet()
    # net.apply(init_weights)
    #net = EEGNet(classes_num=2).to(DEVICE)

    net, lr, num_epochs, loss, trainer = init_param()
    train_iter, valid_iter = read_data(batch_size)

    writer = SummaryWriter("logs_train")
    loss_list = []
    acc_list = []

    # valid_acc_list = []
    # loss_single_list = []
    # acc_single_list = []
    # valid_single_list = []
    # for epoch in range(num_epochs):
    #     print("------第 {} 轮训练开始------".format(epoch + 1))
    #     a, b = train_epoch_EEGNet(net, train_iter, loss, trainer)
    #     print("第{}次训练损失为 {}".format(epoch + 1, a))
    #     print("第{}次训练精度为 {}".format(epoch + 1, b))
    #     loss_single_list.append(a)
    #     acc_single_list.append(b)
    #     valid_acc = evaluate_accuracy(net, valid_iter)
    #     valid_single_list.append(valid_acc)
    # loss_list.append(loss_single_list)
    # acc_list.append(acc_single_list)
    # valid_acc_list.append(valid_single_list)


    valid_acc_list = []
    for index in range(1):
        print("------第 {} 人训练开始------".format(index + 1))
        loss_single_list = []
        acc_single_list = []
        valid_single_list = []
        for epoch in range(num_epochs):
            print("------第 {} 轮训练开始------".format(epoch + 1))
            a, b = train_epoch_EEGNet(net, train_iter[index], loss, trainer, batch_size)
            print("第{}次训练损失为 {}".format(epoch+1,a))
            print("第{}次训练精度为 {}".format(epoch+1,b))
            loss_single_list.append(a)
            acc_single_list.append(b)
            valid_acc = evaluate_accuracy(net, valid_iter[index])
            print("第{}次valid_acc精度为 {}".format(epoch + 1, valid_acc))
            valid_single_list.append(valid_acc)
        loss_list.append(loss_single_list)
        acc_list.append(acc_single_list)
        valid_acc_list.append(valid_single_list)

    PATH = "EEGNet_kernel1_200epoch_lr0.0005_BS20_0605_15samples_5samples.pt"   #
    # Save 保存整个网络
    torch.save(net, PATH)


    # Load
    # model = torch.load(PATH)
    # model.eval()

    plot.plt_loss(loss_list)
    plot.plt_train_acc(acc_list)
    plot.plt_valid_acc(valid_acc_list)
        # print("训练次数: {}, Loss: {}".format(100*(i+1), l.item()))
        # writer.add_scalar("train_loss", l.item())

