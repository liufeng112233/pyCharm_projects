import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 输入尺寸batch*C*H*W输出尺寸变


#  1X1输出H,W无变化，主要是改变通道数量
def Conv1x1(in_planes, out_planes):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=(1, 1), bias=False)


#  3X3输出H,W无变化，主要是改变通道数量
def Conv3x3(in_planes, out_planes):
    """带填充的3X3卷积"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)


# 上支路，包括自校准设计和残差结构b*C*H*W
class SCConv_UP_block(nn.Module):
    def __init__(self, channel):
        """
        SC上支路模块
        :param channel:  输入特征的通道数
        :param channel: 输入特征的通道数
        :param Hight:  特征的尺寸
        :param Width: 特征的尺寸
        :param pooling_r: 滤波器的尺寸，即池化操作的尺寸,主要是针对width
        """
        super(SCConv_UP_block, self).__init__()

        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=1),
            Conv1x1(channel, channel),
            nn.ReLU(),
            Conv1x1(channel, channel),
        )
        self.conv1x1_k2 = nn.Conv2d(channel, channel, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()

        self.k3 = nn.Sequential(
            Conv3x3(channel, channel)
        )
        self.k4 = nn.Sequential(
            Conv3x3(channel, channel)
        )

    def forward(self, x):
        identity = x

        x_up_1 = self.k2(x)
        # x_up_1 = F.dropout(x_up_1, p=0.4)
        # 插值上采样，得到b*C*H*W
        x_up_1 = F.interpolate(x_up_1, identity.size()[2:])
        x_up_1 = self.sigmoid1(x_up_1)
        x_up_2 = torch.mul(identity, x_up_1)
        x_up_3 = self.sigmoid2(self.conv1x1_k2(x_up_2))

        x_up_4 = self.k3(x)
        x_up_5 = torch.mul(x_up_4, x_up_3)

        x_up_6 = self.k4(x_up_5)
        out = x_up_6

        return out


# 自校准下支路
class SCConv_down_block(nn.Module):
    expansion = 4
    pooling_r = 4  # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, channel):
        """
        此处上半部分k1的计算过程batch*C*H
        加过残差块后的下支路
        """
        super(SCConv_down_block, self).__init__()
        # b*C/2*H*W
        self.k1 = nn.Sequential(
            Conv3x3(channel, channel),
            nn.ReLU(inplace=True)
        )
        # batch*C/2*H*W

    def forward(self, x):
        x_k1 = self.k1(x)
        return x_k1


# 残差块。提高模型的可训练性能，同时加深模型的结构，便于提取更加全面的特征
# 改变通道数量，不改变特征尺寸H*W
# 对输入特征b*C*H*W或分成B*C/2*H*W尺寸
class ResNet_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        """
        主要是加深网路结构
        :param in_channel:   输入的特征的通道数
        :param out_channel:  等于输入通道数
        """
        super(ResNet_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(3, 3), stride=(1, 1),
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=(3, 3), stride=(1, 1),
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        return out


# 自校准模块
class SC_Net(nn.Module):
    def __init__(self, in_channel):
        """

        :param in_channel: 输入特征通道数
        :param :通道减半进行校正(分上下两个分支)
        :param :out_channel = in_channel/2
        :param pooling_r:   平均池化层压缩率2
        """
        super(SC_Net, self).__init__()
        self.conv1x1_block1 = nn.Conv2d(in_channel, int(in_channel/2), kernel_size=(1, 1), stride=(1, 1))
        # self.ReNet = ResNet_block(out_channel, out_channel)
        self.sconv_up_block = SCConv_UP_block(int(in_channel/2))
        self.sconv_down_block = SCConv_down_block(int(in_channel/2))
        self.conv1x1_block2 = nn.Conv2d(in_channel, in_channel, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        identity = x
        x1 = self.conv1x1_block1(x)
        # x1 = self.ReNet(x1)
        x_scconv_up_6 = self.sconv_up_block(x1)
        x_scconv_up_7 = self.sconv_down_block(x1)
        x_concat_8 = torch.cat((x_scconv_up_6, x_scconv_up_7), dim=1)
        x_9 = self.conv1x1_block2(x_concat_8)
        out = torch.add(identity, x_9)
        return out
