import torch
import numpy as np
from torch import nn

dropout_value = 0.4


def ConvBNReLU(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


# 提取灰度图的多维特征b*1*8*10240
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
        return out


'''
    输入：多传感构成的灰度图batch*1*8*10240 MSDGI
    编码：对数据级的融合数据通过多尺度卷积
    解码：反卷积，最后采用多层线性层解码8个(batch*1*10240)，最后对解码融合batch*1*8*10240
'''


class MyAE(nn.Module):
    def __init__(self):
        super(MyAE, self).__init__()
        # 1*8*10240

        self.en_M_layer1 = Multi_features(in_channels=1, out_channels=8)  # 8*8*4*640
        self.en_layer1 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(4, 3), stride=(3, 3))  # 213
        self.en_fc1 = nn.ReLU()
        self.en_layer2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))  # 32*1*106
        self.en_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1)  # 16*1*104
        self.en_encode = nn.ReLU()
        # 8*64*1*106
        # 解码器
        self.de_layer1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1, 3),
                                            stride=(1, 3))  # 8*62*318
        self.de_fc1 = nn.ReLU()
        self.de_layer2 = nn.ConvTranspose1d(in_channels=32, out_channels=8, kernel_size=(3, 3),
                                            stride=(1, 3))  # 8*64*4*954
        self.de_fc2 = nn.ReLU()
        self.de_layer3 = nn.ConvTranspose1d(in_channels=8, out_channels=1, kernel_size=(2, 3),
                                            stride=(2, 3))  # 8*64*2808
        self.de_fc3 = nn.ReLU()
        # 采用多个权重矩阵，单一权重矩阵难以很好的权衡各个通道数据
        self.de_layer31 = nn.Linear(2862, 10240)  # b*1*8*10240
        # self.de_layer32 = nn.Linear(2862, 10240)
        # self.de_layer33 = nn.Linear(2862, 10240)
        # self.de_layer34 = nn.Linear(2862, 10240)
        # self.de_layer35 = nn.Linear(2862, 10240)
        # self.de_layer36 = nn.Linear(2862, 10240)
        self.de_decode = nn.Sigmoid()

    def data_cut(self, features1):
        '''
        data：输入tensor（8,1，6,2862）[batch,kernel_inchannel,sensors, data_length]
        :return: 输出tensor[8,2048]
        '''
        [b, K, S1, L1] = features1.size()
        x1 = torch.cat([features1[0][0][0, :], features1[1][0][0, :], features1[2][0][0, :], features1[3][0][0, :],
                        features1[4][0][0, :], features1[5][0][0, :], features1[6][0][0, :], features1[7][0][0, :]],
                       dim=0)
        x1 = x1.view(b, L1)

        x2 = torch.cat([features1[0][0][1, :], features1[1][0][1, :], features1[2][0][1, :], features1[3][0][1, :],
                        features1[4][0][1, :], features1[5][0][1, :], features1[6][0][1, :], features1[7][0][1, :]],
                       dim=0)
        x2 = x2.view(b, L1)

        x3 = torch.cat([features1[0][0][2, :], features1[1][0][2, :], features1[2][0][2, :], features1[3][0][2, :],
                        features1[4][0][2, :], features1[5][0][2, :], features1[6][0][2, :], features1[7][0][2, :]],
                       dim=0)
        x3 = x3.view(b, L1)

        x4 = torch.cat([features1[0][0][3, :], features1[1][0][3, :], features1[2][0][3, :], features1[3][0][3, :],
                        features1[4][0][3, :], features1[5][0][3, :], features1[6][0][3, :], features1[7][0][3, :]],
                       dim=0)
        x4 = x4.view(b, L1)

        x5 = torch.cat([features1[0][0][4, :], features1[1][0][4, :], features1[2][0][4, :], features1[3][0][4, :],
                        features1[4][0][4, :], features1[5][0][4, :], features1[6][0][4, :], features1[7][0][4, :]],
                       dim=0)
        x5 = x5.view(b, L1)

        x6 = torch.cat([features1[0][0][5, :], features1[1][0][5, :], features1[2][0][5, :], features1[3][0][5, :],
                        features1[4][0][5, :], features1[5][0][5, :], features1[6][0][5, :], features1[7][0][5, :]],
                       dim=0)
        x6 = x6.view(b, L1)
        return x1, x2, x3, x4, x5, x6

    def forward(self, x):
        # 编码
        en_x1 = self.en_M_layer1(x)
        en_x1 = self.en_layer1(en_x1)  # 特征聚合，至关重要，使特征一体化
        en_x2 = self.en_fc1(en_x1)
        en_x3 = self.en_layer2(en_x2)
        en_x4 = self.en_layer3(en_x3)
        encode = self.en_encode(en_x4)
        # 解码
        de_x1 = self.de_layer1(encode)
        de_x2 = self.de_fc1(de_x1)
        de_x3 = self.de_layer2(de_x2)
        de_x4 = self.de_fc2(de_x3)  # 8*64*213
        de_x5 = self.de_layer3(de_x4)
        # 特征分解重构
        de_x51, de_x52, de_x53, de_x54, de_x55, de_x56 = self.data_cut(de_x5)
        de_x61, de_x62, de_x63, de_x64, de_x65, de_x66 = self.de_layer31(de_x51), self.de_layer31(de_x52), \
                                                         self.de_layer31(de_x53), self.de_layer31(de_x54), \
                                                         self.de_layer31(de_x55), self.de_layer31(de_x56)
        de_x6 = torch.cat([de_x61, de_x62, de_x63, de_x64, de_x65, de_x66], dim=1)
        de_x7 = de_x6.view(8, 6, 10240)
        decode = self.de_decode(de_x7)
        return encode, decode
        # 输出中encode：[batch_size,c,width,hight]
        # 输出中decode：[batch_size,c,width,hight]
        # 其中C：channels，图片的通道数，灰度图为1
