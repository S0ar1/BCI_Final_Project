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
    plt.title('the accuracy of training')
    plt.show()


def plt_valid_acc(acc_valid_list, i):
    x = np.linspace(1, len(acc_valid_list[0]), len(acc_valid_list[0]))
    plt.figure()
    for i in range(len(acc_valid_list)):
        plt.plot(x, acc_valid_list[i])
    plt.ylim(0, 1)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('the acc of {} valid'.format(i+1))        #增加显示第几个人的valid
    plt.show()

if __name__ == '__main__':
    acc_list = [0.78, 0.76, 0.62, 0.96, 0.68, 0.58, 0.48, 0.58, 0.54, 0.72, 0.56, 0.88, 0.76, 0.7, 0.84]
    x_bar = [f"{i}" for i in range(1, 16)]
    for i in range(15):
        plt.bar(x_bar[i], acc_list[i])
        plt.text(x_bar[i], acc_list[i], s=acc_list[i], ha="center", fontsize=7)

    plt.bar("aver", 0.696)
    plt.text("aver", 0.696, s=0.696, ha="center", fontsize=7)

    plt.ylim(0, 1)
    plt.xlabel('subject')
    plt.ylabel('accuracy')
    plt.title('The accuracy of test data')
    plt.show()