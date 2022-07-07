"""
    分别对六个传感采用多尺度卷积提取特征然后融合，
    属于初级特征层特征融合
"""

import torch
from torch import nn, optim
import torch.nn.functional as F

dropout_value = 0.4


def ConvBNReLU1X1d(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.ReLU6(inplace=True),
    )


class Multi_features1X1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Multi_features1X1d, self).__init__()
        # 第一层卷积
        self.features1 = ConvBNReLU1X1d(in_channels=in_channels, out_channels=out_channels, kernel_size=16,
                                        stride=16, padding=0)
        self.features2 = ConvBNReLU1X1d(in_channels=in_channels, out_channels=out_channels, kernel_size=32,
                                        stride=16, padding=8)
        self.features3 = ConvBNReLU1X1d(in_channels=in_channels, out_channels=out_channels, kernel_size=64,
                                        stride=16, padding=24)
        self.features4 = ConvBNReLU1X1d(in_channels=in_channels, out_channels=out_channels, kernel_size=128,
                                        stride=16, padding=56)
        self.layer_fusion1 = nn.Conv1d(in_channels=out_channels * 4, out_channels=out_channels * 4, kernel_size=1,
                                       stride=1)
        self.fc = nn.ReLU()

    def forward(self, x):
        out1 = self.features1(x)
        out2 = self.features2(x)
        out3 = self.features3(x)
        out4 = self.features4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)  # 一行堆叠形成特征矩阵
        out = self.fc(self.layer_fusion1(out))
        return out  # 8*(8*4)*640


# 主要模型
class MSCNN_sensor_fusion_model(nn.Module):
    def __init__(self, num_classes):
        """
        对单个传感提取多尺度特征
        :param num_classes: 分类数量
        """
        super(MSCNN_sensor_fusion_model, self).__init__()
        self.layer1_multi1 = Multi_features1X1d(in_channels=1, out_channels=8)  # 8*32*1*640
        self.layer1_multi2 = Multi_features1X1d(in_channels=1, out_channels=8)  # 8*32*1*640
        self.layer1_multi3 = Multi_features1X1d(in_channels=1, out_channels=8)  # 8*32*1*640
        self.layer1_multi4 = Multi_features1X1d(in_channels=1, out_channels=8)  # 8*32*1*640
        self.layer1_multi5 = Multi_features1X1d(in_channels=1, out_channels=8)  # 8*32*1*640
        self.layer1_multi6 = Multi_features1X1d(in_channels=1, out_channels=8)  # 8*32*1*640
        self.layer2 = nn.Conv1d(in_channels=192, out_channels=64, kernel_size=1)
        self.layer2_fc1 = nn.ReLU()  # 8*64*1*106
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 2)),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 3), stride=(1, 2)),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(1, 2)),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.ReLU()
        )   # 8*512*1*10
        self.layer6_line = nn.Linear(512*10, 128*6)
        self.layer6_FC = nn.ReLU()
        self.layer7_line = nn.Linear(128*6, num_classes)
    def forward(self, x):
        # 第一层多尺度卷积
        x0 = x.view(8, 6, 10240).permute(1, 0, 2)
        x11 = self.layer1_multi1(x0[0].view(8, 1, 10240))
        x12 = self.layer1_multi2(x0[1].view(8, 1, 10240))
        x13 = self.layer1_multi3(x0[2].view(8, 1, 10240))
        x14 = self.layer1_multi4(x0[3].view(8, 1, 10240))
        x15 = self.layer1_multi5(x0[4].view(8, 1, 10240))
        x16 = self.layer1_multi6(x0[5].view(8, 1, 10240))
        x1 = torch.cat([x11, x12, x13, x14, x15, x16], dim=1)
        x2 = F.dropout(x1, p=dropout_value)
        x2 = self.layer2(x2)
        x2 = self.layer2_fc1(x2)
        x3 = x2.view(8, 64, 1, 640)

        x4 = self.layer3(x3)
        x4 = F.dropout(x4, p=dropout_value)

        x5 = self.layer4(x4)
        x5 = F.dropout(x5, p=dropout_value)

        x6 = self.layer5(x5)
        x6 = F.dropout(x6, p=dropout_value)

        x6 = x6.view(8, x6.size()[1]*x6.size()[2]*x6.size()[3])
        x7 = self.layer6_FC(self.layer6_line(x6))
        out = self.layer7_line(x7)
        return out, x7, x3
