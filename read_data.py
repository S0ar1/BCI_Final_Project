import csv
import torch
from torch.utils import data as Data
import scipy.io as scio
import numpy as np
# train_loader, valid_loader, weight = read_data(batch_size, 0.8, 0.1)  # 获取训练及验证数据

# batch_size = 128
# train_split =0.8
# valid_split = 0.1

def read_data(batch_size):

    data_path_train = './Project_Data/Train/sample01.mat'
    data_path_valid = './Project_Data/Valid/sample01.mat'

    # data = h5py.File(data_path)
    data_train = scio.loadmat(data_path_train)
    data_valid = scio.loadmat(data_path_valid)

    #单次train的data
    epo_train = data_train['epo']
    mnt = data_train['mnt']
    trainX = epo_train['x'][0][0]
    trainX = trainX.transpose((2, 1, 0))
    trainY = y_to_1D(epo_train['y'][0][0].transpose())

    #单次valid的data
    epo_valid = data_valid['epo']
    mnt = data_valid['mnt']
    validX = epo_valid['x'][0][0]
    validX = validX.transpose((2, 1, 0))
    validY = y_to_1D(epo_valid['y'][0][0].transpose())



    dataset_train = DataAdapter(trainX,trainY) # 构造数据集
    dataset_valid = DataAdapter(validX, validY)  # 构造数据集
    train_loader = Data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)  # 加载DataLoader
    valid_loader = Data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=2)


    print('Data Loading Finished')

    return train_loader,valid_loader

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