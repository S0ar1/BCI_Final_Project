import csv
import torch
from torch.utils import data as Data
import scipy.io as scio
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
# train_loader, valid_loader, weight = read_data(batch_size, 0.8, 0.1)  # 获取训练及验证数据

# batch_size = 128
# train_split =0.8
# valid_split = 0.1


def read_each_data(batch_size, index):
    if index < 9:
        data_path_train = './Project_Data/Train/sample0' + str(index + 1) + '.mat'

    else:
        data_path_train = './Project_Data/Train/sample' + str(index + 1) + '.mat'
    data_train = scio.loadmat(data_path_train)
    # 单次train的data
    epo_train = data_train['epo']
    mnt = data_train['mnt']
    trainX = epo_train['x'][0][0]
    trainX = trainX[1500:2500, :, :]
    trainX = trainX.transpose((2, 1, 0))
    trainY = y_to_1D(epo_train['y'][0][0].transpose())

    if index < 9:
        # data_path_train = './Project_Data/Train/sample0'+str(index+1)+'.mat'
        data_path_test = './Project_Data/Test/sample0'+str(index+1)+'.mat'
    else:
        # data_path_train = './Project_Data/Train/sample' + str(index + 1) + '.mat'
        data_path_test = './Project_Data/Test/sample' + str(index + 1) + '.mat'
    data_test = scio.loadmat(data_path_test)
    # 单次valid的data
    epo_test = data_test['epo']
    mnt = data_test['mnt']
    testX = epo_test['x'][0][0]
    testX = testX[1500:2500, :, :]
    testX = testX.transpose((2, 1, 0))
    testY = y_to_1D(epo_test['y'][0][0].transpose())
    dataset_train = DataAdapter(trainX, trainY)
    dataset_test = DataAdapter(testX, testY)
    train_loader = Data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)  # 加载DataLoader
    test_loader = Data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader, test_loader


def read_single_data(batch_size, index):
    """
    load single person
    :param batch_size:
    :param index: 1-15
    :return:train_loader, test_loader
    """
    # if index < 9:
    #     data_path_train = './Project_Data/Train/sample0'+str(index+1)+'.mat'
    #     data_path_valid = './Project_Data/Valid/sample0'+str(index+1)+'.mat'
    # else:
    #     data_path_train = './Project_Data/Train/sample' + str(index + 1) + '.mat'
    #     data_path_valid = './Project_Data/Valid/sample' + str(index + 1) + '.mat'

    # batch_size = 20
    for i in range(15):     #train的批量读取和合并
        if i < 9:
            data_path_train = './Project_Data/Train/sample0' + str(i + 1) + '.mat'
            # data_path_valid = './Project_Data/Valid/sample0' + str(index + 1) + '.mat'
        else:
            data_path_train = './Project_Data/Train/sample' + str(i + 1) + '.mat'
            # data_path_valid = './Project_Data/Valid/sample' + str(index + 1) + '.mat'
        data_train = scio.loadmat(data_path_train)
        # 单次train的data
        epo_train = data_train['epo']
        mnt = data_train['mnt']
        trainX = epo_train['x'][0][0]
        trainX = trainX[1500:2500, :, :]
        trainX = trainX.transpose((2, 1, 0))
        trainY = y_to_1D(epo_train['y'][0][0].transpose())

        if i == 0:
            trainX_merge = trainX
            trainY_merge = trainY
        else:
            trainX_merge = np.vstack((trainX_merge, trainX))
            trainY_merge = np.concatenate((trainY_merge, trainY), axis=0)

        if index < 9:
            # data_path_train = './Project_Data/Train/sample0'+str(index+1)+'.mat'
            data_path_valid = './Project_Data/Valid/sample0' + str(index + 1) + '.mat'
        else:
            # data_path_train = './Project_Data/Train/sample' + str(index + 1) + '.mat'
            data_path_valid = './Project_Data/Valid/sample' + str(index + 1) + '.mat'
        data_valid = scio.loadmat(data_path_valid)
        # 单次valid的data
        epo_valid = data_valid['epo']
        mnt = data_valid['mnt']
        validX = epo_valid['x'][0][0]
        validX = validX[1500:2500, :, :]
        validX = validX.transpose((2, 1, 0))
        validY = y_to_1D(epo_valid['y'][0][0].transpose())
        trainX_merge = np.vstack((trainX_merge, validX))
        trainY_merge = np.concatenate((trainY_merge, validY), axis=0)

    if index < 9:
        # data_path_train = './Project_Data/Train/sample0'+str(index+1)+'.mat'
        data_path_test = './Project_Data/Test/sample0'+str(index+1)+'.mat'
    else:
        # data_path_train = './Project_Data/Train/sample' + str(index + 1) + '.mat'
        data_path_test = './Project_Data/Test/sample' + str(index + 1) + '.mat'
    data_test = scio.loadmat(data_path_test)
    # 单次valid的data
    epo_test = data_test['epo']
    mnt = data_test['mnt']
    testX = epo_test['x'][0][0]
    testX = testX[1500:2500, :, :]
    testX = testX.transpose((2, 1, 0))
    testY = y_to_1D(epo_test['y'][0][0].transpose())

    # 单次train的data
    # epo_train = data_train['epo']
    # mnt = data_train['mnt']
    # trainX = epo_train['x'][0][0]
    # trainX = trainX[1500:2500, :, :]
    # trainX = trainX.transpose((2, 1, 0))
    # trainY = y_to_1D(epo_train['y'][0][0].transpose())



    #
    # data_path_valid = './Project_Data/Valid/sample15.mat'
    # data_valid = scio.loadmat(data_path_valid)
    # # 单次valid的data
    # epo_valid = data_valid['epo']
    # mnt = data_valid['mnt']
    # validX = epo_valid['x'][0][0]
    # validX = validX[1500:2500, :, :]
    # validX = validX.transpose((2, 1, 0))
    # validY = y_to_1D(epo_valid['y'][0][0].transpose())

    #处理valid的数据，目前只用了一个sample，故不参与for循环
    dataset_train = DataAdapter(trainX_merge, trainY_merge)  # 构造数据集
    dataset_test = DataAdapter(testX, testY)  # 构造数据集
    train_loader = Data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)  # 加载DataLoader
    test_loader = Data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=2)

    print('Data Loading Finished')

    return train_loader, test_loader


def read_data(batch_size):
    """
    :param batch_size:
    :return:
    """
    train_list = []
    test_list = []

    for i in range(15):         #15个人用来训练，单独15个人用来测试
        train_loader, test_loader = read_single_data(batch_size, i)
        train_list.append(train_loader)
        test_list.append(test_loader)
    return train_list, test_list

# 定义该函数用于重新打乱训练集和验证集
def shuffle_data(train_loader,valid_loader,valid_split,batch_size):
    train_dataset = train_loader.dataset.dataset # 获取训练集的数据集
    valid_dataset = valid_loader.dataset.dataset
    X = torch.cat((train_dataset.X,valid_dataset.X),0) # 拼接数据集
    Y = torch.cat((train_dataset.Y,valid_dataset.Y),0)
    dataset = DataAdapter(X,Y) # 重新生成数据集
    train_dataset,valid_dataset = Data.random_split(dataset,[len(dataset) - int(len(dataset)*valid_split),int(len(dataset)*valid_split)]) # 重新划分训练集和验证集
    train_loader = Data.DataLoader(train_dataset,batch_size = batch_size,shuffle = True,num_workers = 0)
    valid_loader = Data.DataLoader(valid_dataset,batch_size = batch_size,shuffle = True,num_workers = 0)
    return train_loader,valid_loader

# 定义数据适配器，用于加载数据至pytorch框架
class DataAdapter(Data.Dataset):

    def __init__(self,X,Y):
        super(DataAdapter,self).__init__()
        self.X = torch.FloatTensor(X)
        self.Y = torch.LongTensor(Y)

    def __getitem__(self,index):
        return self.X[index,:],self.Y[index]

    def __len__(self):
        return len(self.X)

def y_to_1D(datay):
    # 因为nn.CrossEntropyLoss的target只接受1维，对trainY进行降维
    # y_to_1D_data = np.zeros((datay.__len__(), 1))
    y_to_1D_data = np.zeros(datay.__len__())
    for i in range(len(datay)):
        if datay[i, 0] == 0:
            y_to_1D_data[i] = torch.tensor([1])
    return y_to_1D_data