import torch
import numpy as np
from torch import nn
dropout_value = 0.4



def ConvBNReLU(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.ReLU6(inplace=True),
    )


# 提取灰度图的多维特征b*1*8*10240
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
        self.layer_fusion1 = nn.Conv1d(in_channels=out_channels1*4, out_channels=out_channels1*4, kernel_size=1, stride=1)
    def forward(self, x):
        out1 = self.features1(x)
        out2 = self.features2(x)
        out3 = self.features3(x)
        out4 = self.features4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)  # 一行堆叠形成特征矩阵
        out = self.layer_fusion1(out)
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

        self.en_M_layer1 = Multi_features(in_channels=1, out_channels1=8, out_channels2=8, out_channels3=8,
                                          out_channels4=8)  # 8*32*640
        self.en_layer0 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=3)  # 重要特征聚拢
        self.en_fc1 = nn.ReLU()
        self.en_layer2 = nn.MaxPool1d(kernel_size=2, stride=2)  # 64*2*320
        self.en_layer3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)  # 16*1*104
        self.en_encode = nn.ReLU()
        # 8*128*104
        # 解码器
        self.de_layer1 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, stride=3)  # 8*62*312
        self.de_fc1 = nn.ReLU()
        self.de_layer2 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=3)  # 8*64*936
        self.de_fc2 = nn.ReLU()
        self.de_layer3 = nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=3, stride=3)  # 8*64*2808
        self.de_fc3 = nn.ReLU()
        # 采用多个权重矩阵，单一权重矩阵难以很好的权衡各个通道数据
        self.de_layer31 = nn.Linear(2808, 10240)  # b*1*8*10240
        self.de_decode = nn.Sigmoid()

    def forward(self, x):
        # 编码
        en_x1 = self.en_M_layer1(x)
        en_x1 = self.en_layer0(en_x1)  # 特征聚合，至关重要，使特征一体化
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

        de_x7 = de_x5.view(8, 1 * 2808)
        de_x8 = self.de_layer31(de_x7)
        de_x8 = de_x8.view(8, 1, 10240)
        decode = self.de_decode(de_x8)
        return encode, decode
        # 输出中encode：[batch_size,c,width,hight]
        # 输出中decode：[batch_size,c,width,hight]
        # 其中C：channels，图片的通道数，灰度图为1




