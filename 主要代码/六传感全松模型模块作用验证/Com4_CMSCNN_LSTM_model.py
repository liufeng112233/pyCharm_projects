"""
    论文《Multiscale convolutional neural network and decision fusion for rolling bearing fault diagnosis》
"""
# input [8,1,6,10240]
import torch
from torch import nn, optim
import torch.nn.functional as F

dropout_value = 0.6


# input :[8,6,10240]
class DCNN_block(nn.Module):
    def __init__(self):
        super(DCNN_block, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2, 128), stride=(1, 5)),
            nn.MaxPool2d(kernel_size=(1, 64), stride=(1, 3)),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1, 2), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return x3


class CMSCNN_LSTM(nn.Module):
    def __init__(self, num_classes, sensor):
        super(CMSCNN_LSTM, self).__init__()
        self.layer1 = DCNN_block()
        self.layer2 = DCNN_block()
        self.layer3 = DCNN_block()

        self.layer_BiLSTM = nn.Sequential(
            nn.LSTM(input_size=240, hidden_size=30, batch_first=True, bidirectional=True),
        )
        self.FC = nn.ReLU()
        self.layer_linear = nn.Linear(8*60, num_classes)


    def forward(self, x):
        [B, C, S, L] = x.size()
        x0 = x.permute(2, 1, 0, 3)
        x01, x02, x03, x04, x05, x06 = x0[0].permute(1, 0, 2), x0[1].permute(1, 0, 2), x0[2].permute(1, 0, 2), \
                                       x0[3].permute(1, 0, 2), x0[4].permute(1, 0, 2), x0[5].permute(1, 0, 2)
        x11 = torch.cat([x01, x02], dim=1).view(B, 1, 2, L)
        x12 = torch.cat([x03, x04], dim=1).view(B, 1, 2, L)
        x13 = torch.cat([x05, x06], dim=1).view(B, 1, 2, L)

        x1 = self.layer1(x11)
        x2 = self.layer1(x12)
        x3 = self.layer1(x13)

        x4 = torch.cat([x1, x2, x3], dim=3)
        x4 = x4.view(x4.size()[0], x4.size()[1], x4.size()[2] * x4.size()[3])
        x5, _ = self.layer_BiLSTM(x4)
        x5 = self.FC(x5)
        x6 = x5.reshape(x5.size()[0], x5.size()[1]*x5.size()[2])
        out = self.layer_linear(x6)
        return out, x6
