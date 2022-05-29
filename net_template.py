from torch import nn
import torch


class linearNet(nn.Module):
    """
    构造网络
    """
    def __init__(self):
        super(linearNet, self).__init__()
        self._layer1 = nn.Sequential(
            nn.Conv1d(60, 10, kernel_size=3, stride=4, padding=1),
            # nn.BatchNorm1d(626),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=2),
            nn.Flatten(start_dim=1),
            nn.Dropout(),
            nn.Linear(3120, 2)
        )
        self.printModel()

    def forward(self, x):
        """
        模型训练时，不需要使用forward，只要在实例化一个对象中传入对应的参数就可以自动调用 forward 函数
        Net(x)
        """
        x = self._layer1(x)
        return x

    def printModel(self):
        """
        打印模型
        这里还是默认了输入维度
        #TODO: 改为使用统一参数
        """
        X = torch.randn(1, 60, 2500)
        X = self._layer1(X)
        print(self._layer1.__class__.__name__, 'output shape:\t', X.shape)

