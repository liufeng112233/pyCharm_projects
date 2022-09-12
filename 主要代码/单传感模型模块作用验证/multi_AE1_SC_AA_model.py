import torch
import torch.nn as nn
import numpy as n
from Axis_attention_sensor1_block import Axis_Portrait_Attention_block, Axis_Horizontal_Attention_block
# 导入权重融合模块
from Adaptive_Fusion_block import Double_Adaptive_fusion_block, Single_Adaptive_fusion_block, MS_CAM
# 定义空白块，只改变通道数量
def conv2d1X1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), bias=False)


#  1X1输出H,W无变化，主要是改变通道数量
def Conv1d1x1(in_channel, out_channel):
    """1x1 convolution,聚合特征，只改变通道数"""
    return nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)


# 支路二主要多尺度特征
class Multi_features(nn.Module):
    def __init__(self, in_channels, out_channels, Activate_fun, sensors):
        """
        采用大尺度卷积进行特征提取，提高在特征的表达能力
        :param in_channels: 原始数据输入通道（1）
        :param out_channels1-4: : 多尺度输出通道数量（8）
        """
        super(Multi_features, self).__init__()
        # 第一层卷积
        self.features1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(sensors, 16),
                                   stride=(1, 16), padding=(0, 0))
        self.features2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(sensors, 32),
                                   stride=(1, 16), padding=(0, 8))
        self.features3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(sensors, 64),
                                   stride=(1, 16), padding=(0, 24))
        self.features4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(sensors, 128),
                                   stride=(1, 16), padding=(0, 56))
        self.layer0 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1),
                                stride=(1, 1))  # 特征聚合层，恒等映射
        self.FC1 = Activate_fun
        self.layer1_conv = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(4, 3), stride=(3, 3))  # 8*32*1*213
        self.layer1_pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))  # 32*1*106
        self.layer1_FC = Activate_fun
        self.layer2_conv = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1))  # 64*1*106
        self.layer2_FC = Activate_fun

    def forward(self, x):
        out1 = self.features1(x)
        out2 = self.features2(x)
        out3 = self.features3(x)
        out4 = self.features4(x)
        out = torch.cat([out1, out2, out3, out4], dim=2)  # 垂直堆叠形成特征矩阵
        out = self.FC1(self.layer0(out))
        out = self.layer1_FC(self.layer1_pool(self.layer1_conv(out)))
        out = self.layer2_FC(self.layer2_conv(out))
        return out


class Attention_Fusion_Net(nn.Module):
    def __init__(self, sc_ae_model, in_channel, out_channel, Activate_fun, sensor=1, ration=1):
        """

        :param SC_AE_mode:   （训练好）自校准多尺度卷积自编码器，无参数数输入
        :param in_channel:    样本通道数（8，1,8,10240）,默认为1
        :param out_channel:   特征输出通道数（多尺度初级特征输出通道数） [8,8,4,640]  默认为8
        :param ration:    特征压缩比例1
        """
        super(Attention_Fusion_Net, self).__init__()
        self.ration = ration
        # 获取初级特征
        self.SC_AE_layer = sc_ae_model  # 导入自校准编码器的中间层特征,编码器输出和解码器输出  [8,64,1,106]
        self.M_Conv_layer = Multi_features(in_channel, out_channel, Activate_fun, sensor)  # 获取多尺度的聚合特征
        # 特征矩阵聚拢
        self.layer_fusion0 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), bias=False)
        self.layer_relu0 = Activate_fun
        # 加入水平注意力机制
        self.Axis_Horizontal_layer = Axis_Horizontal_Attention_block(64, Activate_fun,ration)
        # 加入水平注意力机制
        self.Axis_Portrait_layer = Axis_Portrait_Attention_block(64, Activate_fun,ration)
        # 采用1X1卷积聚合特征，提高特征是的表现力[8,32,1,102]-->[8,16,1,102]  特征聚拢
        self.layer_fusion1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(1, 1),
                                       padding=(1, 1), bias=False)
        """
        方式一：特征权重聚合过程
        """
        self.layer_fusion21 = Conv1d1x1(2, 1)  # 水平注意力数据融合Qh
        self.layer_fusion22 = Conv1d1x1(2, 1)  # 水平注意力数据融合Kk
        self.layer_fusion23 = Conv1d1x1(2, 1)  # 水平注意力数据融合Vh

        self.layer_fusion31 = Conv1d1x1(128, 64)  # 垂直注意力中间数据融Qw
        self.layer_fusion32 = Conv1d1x1(128, 64)  # 垂直注意力中间数据融Kw
        self.layer_fusion33 = Conv1d1x1(128, 64)  # 垂直注意力中间数据融Vw
        # 归一化处理
        self.layer_Softmax1 = nn.Softmax(dim=1)
        self.layer_Softmax2 = nn.Softmax(dim=1)
        # 权重矩阵聚拢
        self.layer_fusion4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), bias=False)  # 注意力矩阵下采样
        self.layer_softmax3 = nn.Softmax(dim=1)  # 形成概率

        """
            方式二
        """
        self.layer2_conv1 = Conv1d1x1(2, 1)
        self.layer2_conv2 = Conv1d1x1(212, 106)
        self.layer2_fusion6 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), bias=False)
        self.layer2_softmax4 = nn.Softmax(dim=1)  # 形成概率
        """
            方式三
        """
        self.layer_softmax5 = nn.Softmax(dim=1)
        # [8,256,1,106]
        self.layer_fusion8 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), bias=False)
        # [8,64,1,106]
        # 权重
        self.gamma_h1 = nn.Parameter(torch.zeros(1))
        self.gamma_w1 = nn.Parameter(torch.zeros(1))
        self.gamma_h2 = nn.Parameter(torch.zeros(1))
        self.gamma_w2 = nn.Parameter(torch.zeros(1))

        """
            方式四:直接融合输出
        """
        self.layer4_fusion1 = nn.Conv2d(64, 64, (2, 1), (1, 1))

    def forward(self, x):
        ration = self.ration
        sc_AE_features, _ = self.SC_AE_layer(x)  # [8,64,1,106]
        multi_CNN_out = self.M_Conv_layer(x)  # [8,64,1,106]

        # 两条特征支路提取的特征进融合（拼接）
        identity1 = torch.cat([sc_AE_features, multi_CNN_out], dim=1)
        identity = self.layer_relu0(self.layer_fusion0(identity1))

        batch, channel, Hight, Width = sc_AE_features.size()
        """
        融合方式一：
        （1）对底层axis注意力机制进行融合构建新的注意力矩阵,
            即AE隐藏层特征、多尺度输出特征产生的q、k、v进行融合，构建新的注意矩阵
        （2）构建新的全局空间注意力
        （3）输出：out_method1  ,通过注意力机制处理的融合特征
        """
        # 自编码器的特征加入注意力分化
        out_h_SC, x_Qh_SC, x_Kh_SC, x_Vh_SC, gamma_h_SC, Attention_h_wight_SC = \
            self.Axis_Horizontal_layer(sc_AE_features)
        out_w_SC, x_Qw_SC, x_Kw_SC, x_Vw_SC, gamma_w_SC, Attention_w_wight_SC = \
            self.Axis_Portrait_layer(sc_AE_features)
        # 多尺度的特征加入注意力分化
        out_h_MS, x_Qh_MS, x_Kh_MS, x_Vh_MS, gamma_h_MS, Attention_h_wight_MS = \
            self.Axis_Horizontal_layer(multi_CNN_out)
        out_w_MS, x_Qw_MS, x_Kw_MS, x_Vw_MS, gamma_w_MS, Attention_w_wight_MS = \
            self.Axis_Portrait_layer(multi_CNN_out)
        # 进行相同方向的注意力分量进行聚合（拼接）,
        # 用两个通道描述支路的相互依赖性关系，然后的通过一维卷积综合两个通道信息进行特征融合
        x_Qh = torch.cat([x_Qh_SC, x_Qh_MS], dim=1)
        x_Kh = torch.cat([x_Kh_SC, x_Kh_MS], dim=2)  # 水平交互
        x_Vh = torch.cat([x_Vh_SC, x_Vh_MS], dim=2)

        x_Qw = torch.cat([x_Qw_SC, x_Qw_MS], dim=2)
        x_Kw = torch.cat([x_Kw_SC, x_Kw_MS], dim=1)  # 垂直交互
        x_Vw = torch.cat([x_Vw_SC, x_Vw_MS], dim=1)
        # 通道融合
        x_Qh, x_Kh, x_Vh = self.layer_fusion21(x_Qh), \
                           self.layer_fusion22(x_Kh.transpose(1, 2)).transpose(1, 2), \
                           self.layer_fusion23(x_Vh.transpose(1, 2)).transpose(1, 2)
        x_Qw, x_Kw, x_Vw = self.layer_fusion31(x_Qw.transpose(1, 2)).transpose(1, 2), \
                           self.layer_fusion32(x_Kw), \
                           self.layer_fusion33(x_Vw)

        # 构建注意力矩阵
        Attention_h_wight_fusion1 = self.layer_Softmax1(torch.matmul(x_Qh, x_Kh))  # bxWxW
        gamma_h_fusion1 = self.gamma_h1  # 权重参数取均值

        Attention_w_wight_fusion1 = self.layer_Softmax2(torch.matmul(x_Qw, x_Kw))  # bxWxW
        gamma_w_fusion2 = self.gamma_w1

        Attention_matrix_h = torch.matmul(x_Vh, Attention_h_wight_fusion1)  # Vw*weight——>bx(c/r*H)xW
        Attention_matrix_h = Attention_matrix_h.view(batch, channel // ration, Hight,
                                                     Width) * gamma_h_fusion1  # 加入权重参数，防止数据过大

        Attention_matrix_w = torch.matmul(x_Vw, Attention_w_wight_fusion1)  # Vw*weight——>bx(c/r*H)xW
        Attention_matrix_w = Attention_matrix_w.view(batch, channel // ration, Hight,
                                                     Width) * gamma_w_fusion2  # 加入权重参数，防止数据过大
        # 注意力矩阵拼接（拼接方式的选择dim=1?2）
        Attention_matrix1 = torch.cat([Attention_matrix_w, Attention_matrix_h], dim=1)  # [8,128,1,106]
        Attention_matrix = self.layer_softmax3(self.layer_fusion4(Attention_matrix1))  # [8,64,1,106]
        # 两条特征支路提取的特征进融合（拼接）
        out_method1 = torch.mul(identity, Attention_matrix)  # [8,64,1,106]

        """
        融合方式二：
            （1）对axis处理后的权重直接融合（拼接），即不进行底层q、k、v的交互和融合
            （2）attention直接融合归一化
            （3）输出：out_method2
            (4) 水平拼接，采用线性映射
        """
        # [8,2,1]-->[8,2]-->[8,1,1]
        Attention_h_wight = torch.cat([Attention_h_wight_SC, Attention_h_wight_MS], dim=1)
        Attention_h_wight = self.layer2_conv1(Attention_h_wight)  # 8*1*1
        # [8,102,102]-->[8,2*106,106]-->[8,106,106]
        Attention_w_wight = torch.cat([Attention_w_wight_SC, Attention_w_wight_MS], dim=1)
        Attention_w_wight = self.layer2_conv2(Attention_w_wight)
        # 权重矩阵融合
        gamma_h_fusion3 = self.gamma_h2
        gamma_w_fusion4 = self.gamma_w2
        Attention_matrix_h2 = torch.matmul(x_Vh, Attention_h_wight)  # Vw*weight——>bx(c/r*H)xW
        Attention_matrix_h2 = Attention_matrix_h2.view(batch, channel, Hight, Width) * gamma_h_fusion3  # 加入权重参数，防止数据过大

        Attention_matrix_w2 = torch.matmul(x_Vw, Attention_w_wight)  # Vw*weight——>bx(c/r*H)xW
        Attention_matrix_w2 = Attention_matrix_w2.view(batch, channel, Hight, Width) * gamma_w_fusion4  # 加入权重参数，防止数据过大
        # 注意力矩阵拼接
        Attention_matrix1_2 = torch.cat([Attention_matrix_w2, Attention_matrix_h2], dim=1)  # [8,128,1,106]
        Attention_matrix_2 = self.layer2_softmax4(self.layer2_fusion6(Attention_matrix1_2))  # [8,16,1,102]此处是否需要重建映射参数
        # 两条特征支路提取的特征进融合（拼接）
        out_method2 = torch.mul(identity, Attention_matrix_2)  # [8,16,1,102]
        """
        融合方式三：
            (1)注意力机制的输出权重矩阵聚合，
            (2)out_h，out_w直接融合拼接
            (3)采用压缩通道卷积降采样降维
        """
        Attention_matrix1_3 = torch.cat([out_h_SC, out_w_SC, out_h_MS, out_w_MS], dim=1)  # [8,256,1,106]
        Attention_matrix_3 = self.layer_softmax5(self.layer_fusion8(Attention_matrix1_3))  # [8,64,1,106]
        # 方式三采用方式的隐藏层输出特征映射矩阵
        out_method3 = torch.mul(identity, Attention_matrix_3)  # [8,64,1,106]
        # 此处是否需要重建映射参数

        """
        方式四融合
            （1）直接将多尺度特征和编码器特征融合
        """
        out_method4 = torch.cat([sc_AE_features, multi_CNN_out], dim=2)   # 8*64*2*106   通道叠加还是维度叠加
        out_method4 = self.layer4_fusion1(out_method4)

        weight = torch.cat([out_h_SC, out_w_SC, out_h_MS, out_w_MS], dim=2)
        return out_method1, out_method2, out_method3, out_method4, weight   # 输出尺寸[8,64,1,106]
