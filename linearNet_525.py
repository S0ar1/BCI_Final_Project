from torch import nn
import torch


class linearNet(nn.Module):
    # 这样把维度都写出来感觉会更清楚不容易出错
    def __init__(self, in_dim, out_dim1, out_dim2, out_dim3):
        super(linearNet, self).__init__()
        self._layer1 = nn.Sequential(
            nn.linear(in_dim, out_dim1),
            nn.BatchNorm1d(out_dim1),
            nn.ReLU(True)
        )
        self._layer2 = nn.Sequential(
            nn.linear(out_dim1, out_dim2),
            nn.BatchNorm1d(out_dim2),
            nn.ReLU(True)
        )
        self._layer3 = nn.Sequential(
            nn.linear(out_dim2, out_dim3),
            nn.BatchNorm1d(out_dim3),
            nn.ReLU(True)
        )

    def forward(self, x):
        """
        需要看看何时被调用
        """
        x = self._layer1(x)
        x = self._layer2(x)
        x = self._layer3(x)
        return x

    def printModel(self):
        """
        打印模型
        """
        X = torch.randn(1, 60, 2500).view(1, 60*2500)
        X = self._layer1(X)
        print(self._layer1.__class__.__name__, 'output shape:\t', X.shape)
        X = self._layer2(X)
        print(self._layer2.__class__.__name__, 'output shape:\t', X.shape)
        X = self._layer3(X)
        print(self._layer3.__class__.__name__, 'output shape:\t', X.shape)
