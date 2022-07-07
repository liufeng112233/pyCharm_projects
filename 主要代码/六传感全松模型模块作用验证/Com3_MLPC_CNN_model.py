"""
    论文《MLPC-CNN: A multi-sensor vibration signal fault diagnosis
                            method under less computing resources》
"""
# input [8,1,6,10240]
import torch
from torch import nn, optim


# input :[8,6,10240]
class MPLC_CNN(nn.Module):
    def __init__(self, num_classes, sensor):
        super(MPLC_CNN, self).__init__()
        self.layer0 = nn.Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 5))
        self.SSTSC_cov = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=num_classes, kernel_size=(sensor, 49), stride=(1, 1)),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(),
            nn.MaxPool2d(1)
        )
        self.layer_Main_path = nn.Sequential(
            nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=(1, 49)),
            nn.BatchNorm2d(num_classes),
            nn.ReLU()
        )
        self.layer_Bypass_path = nn.Sequential(
            nn.AvgPool2d((1, 49), (1, 1))
        )
        self.layer_adamaxpool_2 = nn.AdaptiveMaxPool2d((1, 1952))
        self.maxpool_3 = nn.MaxPool2d(kernel_size=(1, 100), stride=(1, 10))
        self.maxpool_4 = nn.MaxPool2d(kernel_size=(1, 100), stride=(1, 10))
        self.Global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x1 = self.layer0(x)
        x2 = self.SSTSC_cov(x1)
        x31 = self.layer_Main_path(x2)
        x32 = self.layer_Bypass_path(x2)
        x3 = x31 + x32
        x4 = self.layer_adamaxpool_2(x3)
        x5 = self.maxpool_3(x4)
        x6 = self.maxpool_4(x5)
        out = self.Global_pool(x6)
        out = out.reshape(out.size()[0], out.size()[1]*out.size()[2]*out.size()[3])
        hidden_feature = x5.reshape(x5.size()[0], x5.size()[1]*x5.size()[2]*x5.size()[3])
        return out, hidden_feature
