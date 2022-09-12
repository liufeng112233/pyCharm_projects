"""
    自适应权重融合模块
    注意力去严权重矩阵融合，
    通过自适应方法是的两个矩阵更加有效的融合，
    提高注意力权重矩阵的特征加强，消除特征冗余
"""
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np


def conv2d1X1(channel):
    return nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(1, 1), stride=(1, 1))


# 多尺度特征权重处理 8*4*1*106
class Fusion_ADD_CAT_block(nn.Module):
    def __init__(self):
        super(Fusion_ADD_CAT_block, self).__init__()
        self.add_Fc = nn.ReLU()
        self.cat_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 1))
        self.cat_Fc = nn.ReLU()

    def forward(self, x, y):
        out_add = self.add_Fc(x + y)
        xy1 = torch.cat([x, y], dim=2)  # 8*64*2*106
        out_cat = self.cat_Fc(self.cat_conv(xy1))
        return out_add, out_cat


class Double_Adaptive_fusion_block(nn.Module):
    def __init__(self, channel):
        """
        轴注意力机制水平和轴向权重矩阵融合
        :param channel:  AE输出特征注意力权重通道数
        """
        super(Double_Adaptive_fusion_block, self).__init__()
        # 第一层权重融合
        self.layer11 = conv2d1X1(channel)
        self.layer12 = conv2d1X1(channel)
        self.layer13_softmax = nn.Softmax(dim=1)
        self.layer13 = conv2d1X1(channel)
        # 第二层权重融合
        self.layer21 = conv2d1X1(channel)
        self.layer22_softmax = nn.Softmax(dim=1)
        self.layer23 = conv2d1X1(channel)

    def forward(self, x, y):
        """
        :param x:   权重矩阵1(AE高级特征)
        :param y:   权重矩阵2（CNN低级特征）
        :return:  权重矩阵1的基础上融合
        """
        x1 = self.layer11(x)
        y1 = self.layer12(y)
        xy11 = torch.add(x1, y1)
        xy12 = self.layer13_softmax(xy11)
        xy13 = torch.mul(x, xy12)
        xy14 = self.layer13(xy13)
        Fusion_weight_x = torch.add(xy14, x)

        x2 = self.layer21(Fusion_weight_x)
        xy21 = torch.add(y1, x2)
        xy22 = self.layer22_softmax(xy21)
        xy23 = torch.mul(xy22, y1)
        xy24 = self.layer23(xy23)
        Fusion_weight_y = torch.add(xy24, y)

        return Fusion_weight_x, Fusion_weight_y  # 8*64*1*106


class Single_Adaptive_fusion_block(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(Single_Adaptive_fusion_block, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        xa = x + y
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * y * (1 - wei)
        return xo


class MS_CAM(nn.Module):
    '''
    单特征 进行通道加权,作用类似SE模块
    '''

    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei
