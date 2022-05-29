import matplotlib.pyplot as plt
import numpy as np


def plt_loss(loss_list):
    x = np.linspace(1, len(loss_list), len(loss_list))
    plt.figure()
    plt.plot(x, loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('the loss of training')
    plt.show()


def plt_train_acc(acc_list):
    x = np.linspace(1, len(acc_list), len(acc_list))
    plt.figure()
    plt.plot(x, acc_list)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('the acc of training')
    plt.show()

def plt_valid_acc(acc_valid_list):
    x = np.linspace(1, len(acc_valid_list), len(acc_valid_list))
    plt.figure()
    plt.plot(x, acc_valid_list)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('the acc of valid')
    plt.show()
