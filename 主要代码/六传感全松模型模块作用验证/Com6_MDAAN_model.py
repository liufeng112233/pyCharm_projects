"""
    《A Multisource Dense Adaptation Adversarial Network for Fault Diagnosis of Machinery》
"""
import torch
from torch import nn, optim
import torch.nn.functional as F

dropout_value = 0.6


# input :[8,1,1,1024]
class DT_block(nn.Module):
    def __init__(self, inc, outc):
        super(DT_block, self).__init__()
        self.layer_conv = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=(1, 3), stride=(1, 1))
        self.layer_pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.layer_dropout = nn.Dropout(0.4)

    def forward(self, x):
        x1 = self.layer_conv(x)
        x2 = self.layer_pool(x1)
        x3 = self.layer_dropout(x2)
        return x3


# input [8,1,6,10240]
class MDAAN(nn.Module):
    def __init__(self, num_classes, sensor):
        super(MDAAN, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(sensor, 2), stride=(1, 10)),
            nn.ReLU()
        )
        self.layer_fusion_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 32), stride=(1, 1)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 16), stride=(1, 1)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 16), stride=(1, 1), dilation=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )  # [b,64,1,474]

        self.DT_block1 = DT_block(64, 57)  # [b,57,1,236]
        self.DT_block2 = DT_block(57, 53)  # [b,53,1,117]
        self.DT_block3 = DT_block(53, 51)  # [b,53,1,57]

        self.Dense_block4 = nn.Sequential(
            nn.Conv2d(in_channels=51, out_channels=25, kernel_size=(1, 3), stride=(1, 1)),
            nn.Dropout(0.4),
            nn.BatchNorm2d(25),
            nn.ReLU()
        )  # [b,25,1,55]

        self.Global_avepool = nn.Sequential(
            nn.AvgPool2d((1, 13)),
            nn.ReLU()
        )
        self.layer_FC = nn.Linear(100, num_classes)

    def forward(self, x):
        x1 = self.layer0(x)
        x2 = self.layer_fusion_conv1(x1)
        x3 = self.DT_block1(x2)
        x4 = self.DT_block2(x3)
        x5 = self.DT_block3(x4)
        x6 = self.Dense_block4(x5)
        x7 = self.Global_avepool(x6)
        x71 = x7.view(x7.size()[0], x7.size()[1] * x7.size()[2] * x7.size()[3])
        out = self.layer_FC(x71)
        return out, x71
