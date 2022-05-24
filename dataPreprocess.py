import h5py
import pandas as pd
import scipy.io as scio

if __name__ == '__main__':
    data_path = '/Users/dgx/Desktop/BCI_final_project/Project_Data/Train/sample01.mat'
    # data = h5py.File(data_path)
    data = scio.loadmat(data_path)

    epo = data['epo']
    mnt = data['mnt']
    x = epo['x'][0][0]

    print(x)












