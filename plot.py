import matplotlib.pyplot as plt
import numpy as np


def plt_loss(loss_list):
    x = np.linspace(1, len(loss_list[0]), len(loss_list[0]))
    plt.figure()
    for i in range(len(loss_list)):
        plt.plot(x, loss_list[i])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('the loss of training')
    plt.show()


def plt_train_acc(acc_list):
    x = np.linspace(1, len(acc_list[0]), len(acc_list[0]))
    plt.figure()
    for i in range(len(acc_list)):
        plt.plot(x, acc_list[i])
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.ylim(0, 1)
    plt.title('the acc of training')
    plt.show()


def plt_valid_acc(acc_valid_list):
    x = np.linspace(1, len(acc_valid_list[0]), len(acc_valid_list[0]))
    plt.figure()
    for i in range(len(acc_valid_list)):
        plt.plot(x, acc_valid_list[i])
    plt.ylim(0, 1)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('the acc of valid')
    plt.show()
