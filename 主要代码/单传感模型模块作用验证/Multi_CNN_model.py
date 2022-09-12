import torch
from torch import nn, optim
import torch.nn.functional as F

dropout_value = 0.4


def ConvBNReLU(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.ReLU6(inplace=True),
    )


class Multi_features(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, out_channels3, out_channels4):
        super(Multi_features, self).__init__()
        # 第一层卷积
        self.features1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels1, kernel_size=16, stride=16,
                                    padding=0)
        self.features2 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels2, kernel_size=32, stride=16,
                                    padding=8)
        self.features3 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels3, kernel_size=64, stride=16,
                                    padding=24)
        self.features4 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=128, stride=16,
                                    padding=56)

    def forward(self, x):
        out1 = self.features1(x)
        out2 = self.features2(x)
        out3 = self.features3(x)
        out4 = self.features4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)  # 一行堆叠形成特征矩阵
        return out


# 主要模型
class MSCNN_model(nn.Module):
    def __init__(self, num_classes):
        """
        :param num_classes: 分类数量
        """
        super(MSCNN_model, self).__init__()
        self.Multi_block1 = nn.Sequential(
            Multi_features(in_channels=1, out_channels1=8, out_channels2=8, out_channels3=8, out_channels4=8),
            nn.MaxPool1d(kernel_size=2, stride=2)  # 2X2
        )
        # 每一层后包含神经元的过拟合操作
        # F.dropout(input, p=2)
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(128, 256, 3),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(256, 512, 3),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        self.FC1 = nn.Sequential(
            nn.Linear(9216, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        # 第一层多尺度卷积
        x = self.Multi_block1(x)
        x = F.dropout(x, p=dropout_value)
        # 卷积池化层、正则化层参数
        x = self.layer2(x)
        x = F.dropout(x, p=dropout_value)

        x = self.layer3(x)
        x = F.dropout(x, p=dropout_value)

        x = self.layer4(x)
        x = F.dropout(x, p=dropout_value)

        x1 = self.layer5(x)
        x = F.dropout(x1, p=dropout_value)
        # 展开层
        x2 = x1.view(-1, x1.shape[1] * x1.shape[2])
        out = self.FC1(x2)

        return out, x2
