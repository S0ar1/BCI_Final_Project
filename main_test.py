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
    _, test_iter = read_data(batch_size)
    acc_list = [0 for x in range(0, 15)]
    for num in range(15):
        PATH = "./transfer_subject_"+str(num+1)+".pt"
        net = torch.load(PATH)
        net.eval()
        for i in range(10):
            valid_acc = evaluate_accuracy(net, test_iter[num])
            print("第{}人valid_acc精度为 {}".format(num + 1, valid_acc))
            if valid_acc > acc_list[num]:
                acc_list[num] = valid_acc
    average_acc = 0
    for i in range(len(acc_list)):
        average_acc += acc_list[i]
    average_acc = average_acc/len(acc_list)
    print("average acc:", average_acc)

