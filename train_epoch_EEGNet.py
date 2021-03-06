import torch


def train_epoch_EEGNet(net, train_iter, loss, updater, batch_size):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:

        #EEGNet 需要满足（batchsize，1，C,T）的格式
        X = X.unsqueeze(0).permute((1, 0, 2, 3)).to(DEVICE)
        # 计算梯度并更新参数
        # X = X.unsqueeze(0)
        # y = y[0].long()
        y = y.long().to(DEVICE)
        # y_hat = net(X).unsqueeze(0)
        y_hat, _ = net(X)           #注意：：此处加了一个下划线_ 来，承接多余的元组
        # print("y_hat", y_hat)
        # print("y", y)
        l = loss(y_hat, y.squeeze())        #注意：：此处加了一个.squeeze() 来减少维度
        # l = loss(y_hat, y)
        # print("loss:", l)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())

        # print(net.state_dict()['block_1.0.weight'])  # 查看网络参数
        # print(net.state_dict()['block_1.1.bias'])  # 查看网络参数

    # 返回训练损失和训练精度
    # return metric[0] / metric[2] * batch_size , metric[1] / metric[2]
    return metric[0] / metric[2] , metric[1] / metric[2]

def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    # print(net.state_dict()['block_1.0.weight'])  # 查看网络参数
    # print(net.state_dict()['block_1.1.bias'])  # 查看网络参数

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            X = X.unsqueeze(0).permute((1, 0, 2, 3)).to(DEVICE)
            y = y.long().to(DEVICE)
            y_hat, _ = net(X)  # 注意：：此处加了一个下划线_ 来，承接多余的元组
            # print("y_hat", y_hat)
            # print("y", y)
            # l = loss(y_hat, y.squeeze())  # 注意：：此处加了一个.squeeze() 来减少维度
            metric.add(accuracy(y_hat, y), y.numel())
    return metric[0] / metric[1]

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]