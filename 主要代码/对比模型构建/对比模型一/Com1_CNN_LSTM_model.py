"""
    对比模型一：《Intelligent monitoring and diagnostics using a novel integrated model
                based on deep learning and multi-sensor feature fusion》
"""
from torch import nn, optim
import torch


# input 8*1*3*2048
class Channel_Fusion(nn.Module):
    def __init__(self, sensors):
        super(Channel_Fusion, self).__init__()
        self.layer_fusion1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(sensors, 5), stride=(1, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )
        self.layer_fusion2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(sensors, 5), stride=(1, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )
        self.layer_fusion3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(sensors, 5), stride=(1, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )

    def forward(self, x):
        x1 = self.layer_fusion1(x)
        x2 = self.layer_fusion1(x)
        x3 = self.layer_fusion1(x)
        out = torch.cat([x1, x2, x3], dim=2)
        return out
    # output:8*16*3*511


class CNN_block(nn.Module):
    def __init__(self):
        super(CNN_block, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=48, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )  # 8*48*3*127
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2))
        )  # 8*64*3*63
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), (1, 2))
        )  # 8*128*3*63

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        out = x3
        return out
    # output:8*128*3*63


# input:8*128*3*63
class DRN_block(nn.Module):
    def __init__(self):
        super(DRN_block, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

    def forward(self, x):
        residual = x
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        out = residual + x3
        return out

    # input:8*128*3*63
    # #8*3*(63*128)


class LSTM_block(nn.Module):
    def __init__(self):
        super(LSTM_block, self).__init__()

        self.layer1 = nn.LSTM(input_size=128 * 63, hidden_size=256, bidirectional=True, batch_first=True)

    def forward(self, x):
        x_temp = x.permute(0, 2, 1, 3).reshape(8, 3, 128 * 63)
        x1, _ = self.layer1(x_temp)
        out = x1
        return out


# 汇总模型 [8,1,3,2048]
class CNN_DRN_LSTM_model(nn.Module):
    def __init__(self, num_classes, sensor):
        '''

        :param sensor:  传感数量
        :param num_classes:  分类数量
        '''
        super(CNN_DRN_LSTM_model, self).__init__()
        self.layer_fusion = Channel_Fusion(sensors=sensor)
        self.layer_CNN = CNN_block()
        self.layer_DRN = DRN_block()
        self.layer_LSTM = LSTM_block()
        self.layer_line = nn.Sequential(
            nn.Linear(512 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x1 = self.layer_fusion(x)
        x2 = self.layer_CNN(x1)
        x3 = self.layer_DRN(x2)
        x4 = self.layer_LSTM(x3)
        x5 = x4.reshape(8, 512 * 3)
        out = self.layer_line(x5)
        return out, x5
