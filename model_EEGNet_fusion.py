import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, classes_num):
        super(EEGNet, self).__init__()
        self.drop_out = 0.1

        # self.block_1_1 = nn.Sequential(
        #     # Pads the input tensor boundaries with zero
        #     # left, right, up, bottom
        #     # nn.ZeroPad2d((31, 32, 0, 0)),
        #     nn.Conv2d(
        #         in_channels=1,  # input shape (1, C, T)
        #         out_channels=4,  # num_filters
        #         kernel_size=(1, 40),  # filter size
        #         bias=False
        #     ),  # output shape (8, C, T)
        #     nn.BatchNorm2d(4)  # output shape (8, C, T)
        # )
        #
        # # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        # self.block_2_1 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=4,  # input shape (8, C, T)
        #         out_channels=8,  # num_filters
        #         kernel_size=(40, 1),  # filter size
        #         groups=4,
        #         bias=False
        #     ),  # output shape (16, 1, T)
        #     nn.BatchNorm2d(8),  # output shape (16, 1, T)
        #     nn.ELU(),
        #     nn.AvgPool2d((1, 2)),  # output shape (16, 1, T//4)
        #     nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        # )
        #
        # self.block_3_1 = nn.Sequential(
        #     nn.ZeroPad2d((7, 8, 0, 0)),
        #     nn.Conv2d(
        #         in_channels=8,  # input shape (16, 1, T//4)
        #         out_channels=8,  # num_filters
        #         kernel_size=(1, 8),  # filter size
        #         groups=8,
        #         bias=False
        #     ),  # output shape (16, 1, T//4)
        #     nn.Conv2d(
        #         in_channels=8,  # input shape (16, 1, T//4)
        #         out_channels=16,  # num_filters
        #         kernel_size=(1, 1),  # filter size
        #         bias=False
        #     ),  # output shape (16, 1, T//4)
        #     nn.BatchNorm2d(16),  # output shape (16, 1, T//4)
        #     nn.ELU(),
        #     nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)
        #     # nn.Linear(61, 448),
        #     nn.Dropout(self.drop_out)
        # )

        self.block_1_2 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            # nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(
                in_channels=1,  # input shape (1, C, T)
                out_channels=16,  # num_filters
                kernel_size=(1, 80),  # filter size
                bias=False
            ),  # output shape (8, C, T)
            nn.BatchNorm2d(16)  # output shape (8, C, T)
        )

        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        self.block_2_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,  # input shape (8, C, T)
                out_channels=32,  # num_filters
                kernel_size=(60, 1),  # filter size
                groups=8,
                bias=False
            ),  # output shape (16, 1, T)
            nn.BatchNorm2d(32),  # output shape (16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )

        self.block_3_2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(
                in_channels=32,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 16),  # filter size
                groups=16,
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.BatchNorm2d(16),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)
            nn.Linear(28, 29),
            nn.Dropout(self.drop_out)
        )
        self.block_1_3 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            # nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(
                in_channels=1,  # input shape (1, C, T)
                out_channels=8,  # num_filters
                kernel_size=(1, 60),  # filter size
                bias=False
            ),  # output shape (8, C, T)
            nn.BatchNorm2d(8)  # output shape (8, C, T)
        )

        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        self.block_2_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,  # input shape (8, C, T)
                out_channels=16,  # num_filters
                kernel_size=(60, 1),  # filter size
                groups=8,
                bias=False
            ),  # output shape (16, 1, T)
            nn.BatchNorm2d(16),  # output shape (16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )

        self.block_3_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 16),  # filter size
                groups=16,
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.BatchNorm2d(16),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)
            # nn.Linear(29, 29),
            nn.Dropout(self.drop_out)
        )

        # self.out = nn.Linear((16 * 8), classes_num)
        self.out = nn.Linear((464), classes_num)

    def forward(self, x):
        x1, x2, x3 = x, x, x

        # x = self.block_1(x)
        # x = self.block_2(x)
        # x = self.block_3(x)
        # model 1
        # x1 = self.block_1_1(x1)
        # x1 = self.block_2_1(x1)
        # x1 = self.block_3_1(x1)
        # model 2
        x2 = self.block_1_2(x2)
        x2 = self.block_2_2(x2)
        x2 = self.block_3_2(x2)
        # model 3
        x3 = self.block_1_3(x3)
        x3 = self.block_2_3(x3)
        x3 = self.block_3_3(x3)

        x = (x2+x3)/2

        x = x.view(x.size(0), -1)

        x = self.out(x)
        # return F.softmax(x, dim=1), x   # return x for visualization
        return x, x