import torch
from torch import nn
import scipy.io as scio
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import plot
import net_template
from train_epoch_EEGNet import *
# from read_data import *
# from read_data_merge import *        #此处用sample15个人做训练数据，5个人sample做测试数据
from read_data_seprate import *        #此处用sample15个人做训练数据，15个人sample分别做测试数据
# from read_data_test import  *
from model_EEGNet_fusion import *
# from model_Net_test import *          #测试简化版的EEGNet
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

    lr, num_epochs = 0.0005, 50
    loss = nn.CrossEntropyLoss().to(DEVICE)
    PATH = "EEGNet_kernel1_200epoch_lr0.0005_BS20_0609_15samples_15samples_seprate_combine_train&valid.pt"
    net = torch.load(PATH)
    trainer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.01)      #【0604】加入正则化
    return net, lr, num_epochs, loss, trainer

if __name__ == '__main__':

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 20


    # train_iter, test_iter = read_data(batch_size)

    # writer = SummaryWriter("logs_train")
    loss_list = []
    acc_list = []


    valid_acc_list = []
    # 分别对每个subject进行训练
    for index in range(15):
        print("------第 {} 人训练开始------".format(index + 1))
        net, lr, num_epochs, loss, trainer = init_param()

        train_iter, test_iter = read_each_data(batch_size, index)
        loss_single_list = []
        acc_single_list = []
        valid_single_list = []
        for epoch in range(num_epochs):
            print("------第 {} 轮训练开始------".format(epoch + 1))
            a, b = train_epoch_EEGNet(net, train_iter, loss, trainer, batch_size)
            print("第{}次训练损失为 {}".format(epoch+1, a))
            print("第{}次训练精度为 {}".format(epoch+1, b))
            loss_single_list.append(a)
            acc_single_list.append(b)
            valid_acc = evaluate_accuracy(net, test_iter)
            print("第{}次valid_acc精度为 {}".format(epoch + 1, valid_acc))

        # Save 保存整个网络
        OUT_PATH = "transfer_subject_"+str(index + 1)+".pt"
        torch.save(net, OUT_PATH)


