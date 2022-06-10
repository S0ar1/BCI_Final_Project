# for test model only
import torch

import plot
import net_template
from train_epoch_EEGNet import *
# from read_data import *
# from read_data_merge import *        #此处用sample15个人做训练数据，5个人sample做测试数据
from read_data_seprate import *        #此处用sample15个人做训练数据，15个人sample分别做测试数据
# from read_data_test import  *
from model_EEGNet_fusion import *

if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    # net = EEGNet(classes_num=2).to(DEVICE)
    PATH = "./EEGNet_kernel1_200epoch_lr0.0005_BS20_0609_15samples_15samples_seprate_combine_train&valid.pt"
    net = torch.load(PATH)
    net.eval()

    _, test_iter = read_data(batch_size)
    acc_list = []
    for num in range(15):
        valid_acc = evaluate_accuracy(net, test_iter[num])
        print("第{}人valid_acc精度为 {}".format(num + 1, valid_acc))
        acc_list.append(valid_acc)
    average_acc = 0
    for i in range(len(acc_list)):
        average_acc += acc_list[i]
    average_acc = average_acc/len(acc_list)
    print("average acc:", average_acc)