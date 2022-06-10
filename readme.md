# 使用说明
## 数据放置
出于上传代码大小的考虑，并未包含训练数据，如需运行代码，请保存Project_Data在如下路径
 e.g. data_path_valid = './Project_Data/Valid/sample01.mat'
## 运行代码
我们已经将训练好的模型进行了保存。
EEGNet_kernel1_200epoch_lr0.0005_BS20_0609_15samples_15samples_seprate_combine_train&valid.pt
是corss_subject的模型
transfer_subject_1.pt - transfer_subject_15.pt
是transfer_learning 后每个subject单独训练的模型

如需运行模型训练过程，请运行 main_EEGNet_fusion.py
如需观察训练后效果，请运行 main_test.py


