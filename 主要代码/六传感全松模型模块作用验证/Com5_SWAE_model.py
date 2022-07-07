
# input [8,1,6,10240]
import torch
from torch import nn, optim
import torch.nn.functional as F

dropout_value = 0.6


# input :[8,1,1,10240]
class AE_block(nn.Module):
    def __init__(self):
        super(AE_block, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2, 128), stride=(1, 5)),
            nn.MaxPool2d(kernel_size=(1, 64), stride=(1, 3)),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 2), stride=(1, 4)),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        return x2


class SWAE(nn.Module):
    def __init__(self, num_classes, sensor):
        super(SWAE, self).__init__()
        self.layer1 = AE_block()
        self.layer2 = AE_block()
        self.layer3 = AE_block()

        self.layer_connect1 = nn.Sequential(
            nn.Linear(in_features=32 * 81, out_features=32),
            nn.ReLU()
        )
        self.layer_connect2 = nn.Sequential(
            nn.Linear(in_features=32 * 81, out_features=32),
            nn.ReLU()
        )
        self.layer_connect3 = nn.Sequential(
            nn.Linear(in_features=32 * 81, out_features=32),
            nn.ReLU()
        )

        self.layer_fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 3)),
            nn.ReLU()
        )
        self.layer_fusion_linear = nn.Sequential(
            nn.Linear(in_features=32 * 81, out_features=32),
            nn.ReLU()
        )
        self.layer_out = nn.Linear(in_features=128,out_features=num_classes)

    def forward(self, x):
        [B, C, S, L] = x.size()
        x0 = x.permute(2, 1, 0, 3)
        x01, x02, x03, x04, x05, x06 = x0[0].permute(1, 0, 2), x0[1].permute(1, 0, 2), x0[2].permute(1, 0, 2), \
                                       x0[3].permute(1, 0, 2), x0[4].permute(1, 0, 2), x0[5].permute(1, 0, 2)
        x11 = torch.cat([x01, x02], dim=1).view(B, 1, 2, L)
        x12 = torch.cat([x03, x04], dim=1).view(B, 1, 2, L)
        x13 = torch.cat([x05, x06], dim=1).view(B, 1, 2, L)

        x21 = self.layer1(x11)
        x22 = self.layer1(x12)
        x23 = self.layer1(x13)  # [b,32,1,81]

        # 数据维度修改
        x31 = x21.reshape(x21.size()[0], x21.size()[1] * x21.size()[2] * x21.size()[3])
        x32 = x22.reshape(x22.size()[0], x22.size()[1] * x22.size()[2] * x22.size()[3])
        x33 = x23.reshape(x23.size()[0], x23.size()[1] * x23.size()[2] * x23.size()[3])

        x51 = self.layer_connect1(x31)
        x52 = self.layer_connect1(x32)
        x53 = self.layer_connect1(x33)

        x34 = torch.cat([x21, x22, x23], dim=3)
        x44 = self.layer_fusion_conv(x34)
        x44 = x44.view(x44.size()[0], x44.size()[1] * x44.size()[2] * x44.size()[3])
        x54 = self.layer_fusion_linear(x44)  # [8,32]

        x6 = torch.cat([x51, x52, x53, x54], dim=1)
        out = self.layer_out(x6)
        return out,x6
