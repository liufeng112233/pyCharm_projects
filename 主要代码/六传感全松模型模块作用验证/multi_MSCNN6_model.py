import torch
from torch import nn, optim
import torch.nn.functional as F

dropout_value = 0.4


def ConvBNReLU(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


class Multi_features(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Multi_features, self).__init__()
        # 第一层卷积
        self.features1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=(6, 16),
                                    stride=(1, 16), padding=(0, 0))
        self.features2 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=(6, 32),
                                    stride=(1, 16), padding=(0, 8))
        self.features3 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=(6, 64),
                                    stride=(1, 16), padding=(0, 24))
        self.features4 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=(6, 128),
                                    stride=(1, 16), padding=(0, 56))
        self.layer_fusion1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1),
                                       stride=(1, 1))
        self.fc = nn.ReLU()

    def forward(self, x):
        out1 = self.features1(x)
        out2 = self.features2(x)
        out3 = self.features3(x)
        out4 = self.features4(x)
        out = torch.cat([out1, out2, out3, out4], dim=2)  # 一行堆叠形成特征矩阵
        out = self.fc(self.layer_fusion1(out))
        return out  # 8*8*4*640


# 主要模型
class MSCNN_model(nn.Module):
    def __init__(self, num_classes):
        """
        :param num_classes: 分类数量
        """
        super(MSCNN_model, self).__init__()
        self.layer1_multi = Multi_features(in_channels=1, out_channels=8)  # 8*8*4*640
        # 每一层后包含神经元的过拟合操作
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(4, 3), stride=(1, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )  # 8*32*1*159
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, (1, 3), (1, 2)),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), (1, 2))
        )  # 8*64*1*39
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, (1, 3)),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), (1, 2))
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, (1, 3)),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), (1, 2))
        )  # 8*256*1*8
        self.FC1 = nn.Sequential(
            nn.Linear(256*8, 32*8),
            nn.ReLU(),
            nn.Linear(32*8, num_classes)
        )

    def forward(self, x):
        # 第一层多尺度卷积
        x1 = self.layer1_multi(x)
        x2 = self.layer2(x1)
        x2 = F.dropout(x2, p=dropout_value)
        # 卷积池化层、正则化层参数
        x3 = self.layer3(x2)
        x3 = F.dropout(x3, p=dropout_value)

        x4 = self.layer4(x3)
        x4 = F.dropout(x4, p=dropout_value)

        x5 = self.layer5(x4)
        x5 = F.dropout(x5, p=dropout_value)
        # 展开层
        x6 = x5.view(-1, x5.shape[1] * x5.shape[2] * x5.shape[3])
        out = self.FC1(x6)

        return out, x6
