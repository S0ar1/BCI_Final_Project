import h5py
import pandas as pd
import scipy.io as scio



# data_train_label=data_train.get('label')#取出字典里的label
#
# data_train_data=data_train.get('data')#取出字典里的data

if __name__ == '__main__':
    data_path = '/Users/dgx/Desktop/BCI_final_project/Project_Data/Train/sample01.mat'
    # data = h5py.File(data_path)
    data = scio.loadmat(data_path)

    epo = data['epo']
    mnt = data['mnt']
    print(epo)
    # data = pd.read_csv('../Dataset/train.csv')