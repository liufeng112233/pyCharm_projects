import torch
from torch import nn

"""
    输入：8*1*6*10240
"""


# 多尺度特征提取模块
class MS_feature_ADD(nn.Module):
    def __init__(self, inc, outc):
        super(MS_feature_ADD, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7)),
            nn.BatchNorm2d(outc),
            nn.Mish()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=(1, 25), stride=(1, 1), padding=(0, 12)),
            nn.BatchNorm2d(outc),
            nn.Mish()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=(1, 75), stride=(1, 1), padding=(0, 37)),
            nn.BatchNorm2d(outc),
            nn.Mish()
        )

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(x)
        out3 = self.layer3(x)
        out = out1 + out2 + out3
        return out


# 特征融合模块 8*10*6*2048de cat
class IDMFFN(nn.Module):
    def __init__(self, num_classes, sensor):
        super(IDMFFN, self).__init__()
        self.layer0 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 5), stride=(1, 5), padding=(0, 2))
        self.layer1 = MS_feature_ADD(1, 10)
        self.layer2 = MS_feature_ADD(10, 10)
        self.layer3 = MS_feature_ADD(10, 10)

        self.layer1_conv = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=(sensor, 16), stride=(1, 3)),
            nn.Mish()
        )
        self.layer2_conv = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=(sensor, 16), stride=(1, 3)),
            nn.Mish()
        )
        self.layer3_conv = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=(sensor, 16), stride=(1, 3)),
            nn.Mish()
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 2)),
            nn.Mish()
        )

        self.layer5_GRU1 = nn.GRU(input_size=338 * 3, hidden_size=256, batch_first=True)
        self.layer5_GRU2 = nn.GRU(input_size=338 * 3, hidden_size=256, batch_first=True)
        self.layer5_GRU3 = nn.GRU(input_size=338 * 3, hidden_size=256, batch_first=True)
        self.layer5_GRU4 = nn.GRU(input_size=338 * 3, hidden_size=256, batch_first=True)
        self.layer5_FC = nn.Mish()

        self.layer7 = nn.Sequential(
            nn.Linear(64 * 1024, 1024),
            nn.Mish()
        )
        self.layer8 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Mish(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x1 = self.layer0(x)
        out1 = self.layer1(x1)  # 8*10*sensor*2048
        out2 = self.layer2(out1)  # 8*10*sensor*2048
        out3 = self.layer3(out2)  # 8*10*sensor*2048
        out11 = self.layer1_conv(out1)  # 8*32*1*678
        out22 = self.layer2_conv(out2)
        out33 = self.layer3_conv(out3)

        out4 = torch.cat([out11, out22, out33], dim=2)  # 8*32*3*2048
        out5 = self.layer4(out4)  # 8*128*3*338
        out51 = out5.reshape([out5.size()[0], out5.size()[1], out5.size()[2] * out5.size()[3]])  # 8*128*(3*338)
        out61, _ = self.layer5_GRU1(out51)  # 8*64*256
        out62, _ = self.layer5_GRU2(out51)
        out63, _ = self.layer5_GRU3(out51)
        out64, _ = self.layer5_GRU4(out51)
        out6 = self.layer5_FC(torch.cat([out61, out62, out63, out64], dim=2))  # 8*64*1024

        out6 = out6.reshape([out6.size()[0], out6.size()[1] * out6.size()[2]])
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)

        return out8, out7
