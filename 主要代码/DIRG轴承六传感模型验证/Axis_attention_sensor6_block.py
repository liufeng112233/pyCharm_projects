"""
    定义输入特征尺度为[b,c,H,W]
    此处的输入主要是编码器的隐藏层特征和多尺度输出特征
    两个模型获得的特征都需要进过AXIS attention，构建中间的注意力群众进行进行特征融合8*64*1*106
"""
import torch
import torch.nn as nn
import numpy as n


# 横向的注意力机制，对通道的依赖性建模
# Qh[b,h,C*W],Kh[b,c*W,h],Vh[b,h,h]    线性矩阵操作过程
class Axis_Horizontal_Attention_block(nn.Module):
    def __init__(self, channel, activate_F, ration=1):
        """
            :param channel:  提取的特征图通道数量
            :param ration:   期望的通道压缩率，针对隐藏层的特征压缩处理
            """
        super(Axis_Horizontal_Attention_block, self).__init__()
        self.layer_Qh = nn.Conv2d(in_channels=channel, out_channels=channel // ration, kernel_size=(1, 1), bias=False)
        self.layer_Kh = nn.Conv2d(in_channels=channel, out_channels=channel // ration, kernel_size=(1, 1), bias=False)
        self.layer_Vh = nn.Conv2d(in_channels=channel, out_channels=channel // ration, kernel_size=(1, 1), bias=False)
        # self.BN = nn.BatchNorm2d(channel // ration)
        self.relu_fc_h = activate_F
        self.Softmax_fc_h = nn.Softmax(dim=1)
        self.gamma_h = nn.Parameter(torch.zeros(1))  # 定义权重参数

        self.layer_up = nn.Conv2d(channel // ration, channel, (1, 1))
        self.ration = ration

    def forward(self, x):
        batch, channel, Hight, Width = x.size()
        ration = self.ration
        # Qh
        x_qh1 = self.relu_fc_h(self.layer_Qh(x))  # bxC/rxHxW
        x_Qh = x_qh1.view(batch, Hight, -1)  # bxHx(C/r*W)
        # Kh
        x_Kh1 = self.relu_fc_h(self.layer_Kh(x))  # bxC/rxHxW
        x_Kh = x_Kh1.view(batch, Hight, -1).permute(0, 2, 1)  # bx(c/r*W)xH

        # Vh
        x_Vh1 = self.relu_fc_h(self.layer_Vh(x))
        x_Vh = x_Vh1.view(batch, Hight, -1).transpose(1, 2)  # bx(c/r*W)xH

        Attention_h_wight = self.Softmax_fc_h(torch.matmul(x_Qh, x_Kh))  # bxHxH
        gamma_h = self.gamma_h

        out1_h = torch.matmul(x_Vh, Attention_h_wight)  # Vh*weight——>bx(c/r*W)xH
        out_h = out1_h.view(batch, channel // ration, Hight, Width) * gamma_h  # 加入权重参数，防止数据过大
        out_h = self.layer_up(out_h)
        return out_h, x_Qh, x_Kh, x_Vh, gamma_h, Attention_h_wight
        # out_w: 表示注意力矩阵的输出注意力权重矩阵（最终输出矩阵）
        # Q、K、V：注意力机制内部的三个参数值
        # gamma_w：权重举证优化参数，防止矩阵数值过大
        # Attention_w_wight：K,Q点积矩阵


# 纵向的注意力机制，长程的全局的依赖性，降低运算量
# Qw[b,w,c*h],Kw[b,c*h,w],Vw[b,w,w]
class Axis_Portrait_Attention_block(nn.Module):
    def __init__(self, channel, activate_F, ration=1):
        """
        :param channel:  提取的特征图通道数量
        :param ration:   期望的通道压缩率
        """
        super(Axis_Portrait_Attention_block, self).__init__()
        self.layer_Qw = nn.Conv2d(in_channels=channel, out_channels=channel // ration, kernel_size=(1, 1))
        self.layer_Kw = nn.Conv2d(in_channels=channel, out_channels=channel // ration, kernel_size=(1, 1))
        self.layer_Vw = nn.Conv2d(in_channels=channel, out_channels=channel // ration, kernel_size=(1, 1))
        self.relu_FC_w = activate_F
        self.Softmax_FC_w = nn.Softmax(dim=1)
        self.gamma_w = nn.Parameter(torch.zeros(1))

        self.layer_up_w = nn.Conv2d(channel // ration, channel, (1, 1))
        self.ration = ration

    def forward(self, x):
        batch, channel, Hight, Width = x.size()
        ration = self.ration
        # Qh
        x_qw1 = self.relu_FC_w(self.layer_Qw(x))  # bxC/rxHxW
        x_Qw = x_qw1.view(batch, Width, -1)  # bxWx(H*C/r)
        # Kh
        x_Kw1 = self.relu_FC_w(self.layer_Kw(x))  # bxC/rxHxW
        x_Kw = x_Kw1.view(batch, Width, -1).transpose(1, 2)  # bx(H*C/r)xW

        # Vh
        x_Vw1 = self.relu_FC_w(self.layer_Vw(x))
        x_Vw = x_Vw1.view(batch, Width, -1).transpose(1, 2)  # bx(H*C/r)xW
        # 注意力矩阵
        Attention_w_wight = self.Softmax_FC_w(torch.matmul(x_Qw, x_Kw))  # bxWxW
        gamma_w = self.gamma_w

        out1_w = torch.matmul(x_Vw, Attention_w_wight)  # Vw*weight——>bx(c/r*H)xW
        out_w = out1_w.view(batch, channel // ration, Hight, Width) * gamma_w  # 加入权重参数，防止数据过大
        out_w = self.layer_up_w(out_w)
        return out_w, x_Qw, x_Kw, x_Vw, gamma_w, Attention_w_wight
        # out_w: 表示注意力矩阵的输出注意力权重矩阵（最终输出矩阵）
        # Q、K、V：注意力机制内部的三个参数值
        # gamma_w：权重举证优化参数，防止矩阵数值过大
        # Attention_w_wight：K,Q点积矩阵
