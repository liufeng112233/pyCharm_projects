"""
    定义输入特征尺度为[b,c,H,W]
    此处的输入主要是编码器的隐藏层特征和多尺度输出特征
    两个模型获得的特征都需要进过AXIS attention，构建中间的注意力群众进行进行特征融合
"""
import torch
import torch.nn as nn
import numpy as n


# 定义空白块，只改变通道数量
def conv2d3X3(in_channel, out_channel, stride=1):
    return nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)


#  1X1输出H,W无变化，主要是改变通道数量
def Conv2d1x1(in_planes, out_planes):
    """1x1 convolution,聚合特征，只改变通道数"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=(1, 1), bias=False)


# 初始特征提取，采用多尺度卷积提取主路特征结果
class Multi_features(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, out_channels3, out_channels4):
        """
        采用大尺度卷积进行特征提取，提高在特征的表达能力
        :param in_channels: 原始数据输入通道（1）
        :param out_channels1-4: : 多尺度输出通道数量（8）
        """
        super(Multi_features, self).__init__()
        # 第一层卷积
        self.features1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels1, kernel_size=(8, 16),
                                   stride=(1, 16), padding=(0, 0))
        self.features2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels2, kernel_size=(8, 32),
                                   stride=(1, 16), padding=(0, 8))
        self.features3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels3, kernel_size=(8, 64),
                                   stride=(1, 16), padding=(0, 24))
        self.features4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels4, kernel_size=(8, 128),
                                   stride=(1, 16), padding=(0, 56))
        self.layer0 = nn.Conv2d(in_channels=out_channels1, out_channels=out_channels1, kernel_size=(1, 1),
                                stride=(1, 1))  # 特征聚合层，恒等映射
        self.FC1 = nn.ReLU()
        self.layer_end = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8*2*320
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 16), stride=3, padding=0),  # 16*1*102
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1,1), padding=0),
            nn.ReLU()
        )

    def forward(self, x):
        out1 = self.features1(x)
        out2 = self.features2(x)
        out3 = self.features3(x)
        out4 = self.features4(x)
        out = torch.cat([out1, out2, out3, out4], dim=2)  # 垂直堆叠形成特征矩阵
        out = self.layer0(out)
        out_multi_features = self.FC1(out)  # 8*8*4*640
        out_features = self.layer_end(out_multi_features)  # 8*16*1*102
        return out_multi_features, out_features


# 横向的注意力机制，对通道的依赖性建模
# Qh[b,h,C*W],Kh[b,c*W,h],Vh[b,h,h]    线性矩阵操作过程
class Axis_Horizontal_Attention_block(nn.Module):
    def __init__(self, channel, ration):
        """
            :param channel:  提取的特征图通道数量
            :param ration:   期望的通道压缩率，针对隐藏层的特征压缩处理
            """
        super(Axis_Horizontal_Attention_block, self).__init__()
        self.layer_Qh = nn.Conv2d(in_channels=channel, out_channels=channel // ration, kernel_size=(1, 1), bias=False)
        self.layer_Kh = nn.Conv2d(in_channels=channel, out_channels=channel // ration, kernel_size=(1, 1), bias=False)
        self.layer_Vh = nn.Conv2d(in_channels=channel, out_channels=channel // ration, kernel_size=(1, 1), bias=False)
        # self.BN = nn.BatchNorm2d(channel // ration)
        self.relu_fc_h = nn.ReLU()
        self.Softmax_fc_h = nn.Softmax(dim=1)
        self.gamma_h = nn.Parameter(torch.zeros(1))  # 定义权重参数

    def forward(self, x):
        batch, channel, Hight, Width = x.size()
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
        out_h = out1_h.view(batch, channel, Hight, Width) * gamma_h  # 加入权重参数，防止数据过大
        return out_h, x_Qh, x_Kh, x_Vh, gamma_h, Attention_h_wight


# 纵向的注意力机制，长程的全局的依赖性，降低运算量
# Qw[b,w,c*h],Kw[b,c*h,w],Vw[b,w,w]
class Axis_Portrait_Attention_block(nn.Module):
    def __init__(self, channel, ration):
        """
        :param channel:  提取的特征图通道数量
        :param ration:   期望的通道压缩率
        """
        super(Axis_Portrait_Attention_block, self).__init__()
        self.layer_Qw = nn.Conv2d(in_channels=channel, out_channels=channel // ration, kernel_size=(1, 1))
        self.layer_Kw = nn.Conv2d(in_channels=channel, out_channels=channel // ration, kernel_size=(1, 1))
        self.layer_Vw = nn.Conv2d(in_channels=channel, out_channels=channel // ration, kernel_size=(1, 1))
        self.relu_FC_w = nn.ReLU()
        self.Softmax_FC_w = nn.Softmax(dim=1)
        self.gamma_w = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, channel, Hight, Width = x.size()
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
        out_w = out1_w.view(batch, channel, Hight, Width) * gamma_w  # 加入权重参数，防止数据过大
        return out_w, x_Qw, x_Kw, x_Vw, gamma_w, Attention_w_wight


class Attention_Fusion_Net(nn.Module):
    def __init__(self, sc_ae_model, in_channel, out_channel, ration):
        """

        :param SC_AE_mode:   （训练好）自校准多尺度卷积自编码器，无参数数输入
        :param in_channel:    样本通道数（8，1,8,10240）,默认为1
        :param out_channel:   特征输出通道数（多尺度初级特征输出通道数） [8,8,4,640]  默认为8
        :param ration:    特征压缩比例
        """
        super(Attention_Fusion_Net, self).__init__()
        # 获取初级特征
        self.SC_AE_layer = sc_ae_model  # 导入自校准编码器的中间层特征,编码器输出和解码器输出  [8,128,1,102]
        self.M_Conv_layer = Multi_features(in_channel, out_channel, out_channel, out_channel, out_channel)  # 获取多尺度的聚合特征
        # 加入水平注意力机制
        self.Axis_Horizontal_layer = Axis_Horizontal_Attention_block(128, ration)
        # 加入水平注意力机制
        self.Axis_Portrait_layer = Axis_Portrait_Attention_block(128, ration)
        # 采用1X1卷积聚合特征，提高特征是的表现力[8,32,1,102]-->[8,16,1,102]  特征聚拢
        self.layer_fusion1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(1, 1),
                                       padding=(1, 1), bias=False)
        """
        方式一：特征权重聚合过程
        """
        self.layer_fusion21 = conv2d3X3(2, 1)  # 水平注意力数据融合Qh
        self.layer_fusion22 = conv2d3X3(2, 1)  # 水平注意力数据融合Kk
        self.layer_fusion23 = conv2d3X3(2, 1)  # 水平注意力数据融合Vh

        self.layer_fusion31 = conv2d3X3(256, 128)  # 垂直注意力中间数据融Qw
        self.layer_fusion32 = conv2d3X3(256, 128)  # 垂直注意力中间数据融Kw
        self.layer_fusion33 = conv2d3X3(256, 128)  # 垂直注意力中间数据融Vw
        # 归一化处理
        self.layer_Softmax1 = nn.Softmax(dim=1)
        self.layer_Softmax2 = nn.Softmax(dim=1)
        # 权重矩阵聚拢
        self.layer_fusion4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=(1, 1),
                                       bias=False)  # 注意力矩阵下采样
        self.layer_softmax3 = nn.Softmax(dim=1)  # 形成概率
        # 特征矩阵聚拢
        self.layer_fusion5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.layer_relu1 = nn.ReLU()

        """
            方式二
        """
        self.layer_line1 = nn.Linear(2, 1)
        self.layer_line2 = nn.Linear(2 * 102 * 102, 102 * 102)
        self.layer_fusion6 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=(1, 1),
                                       bias=False)
        self.layer_softmax4 = nn.Softmax(dim=1)  # 形成概率
        self.layer_fusion7 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.layer_relu2 = nn.ReLU()
        """
            方式三
        """
        self.layer_softmax5 = nn.Softmax(dim=1)
        # [8,62,1,102]
        self.layer_fusion8 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=False)
        # [8,16,1,102]
        self.layer_relu3 = nn.ReLU()
        # 舍弃部分网络节点
        self.gamma_h1 = nn.Parameter(torch.zeros(1))
        self.gamma_w1 = nn.Parameter(torch.zeros(1))
        self.gamma_h2 = nn.Parameter(torch.zeros(1))
        self.gamma_w2 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        sc_AE_features, _ = self.SC_AE_layer(x)  # [8,16,1,102]
        out_multi_features, out_Hide_features = self.M_Conv_layer(x)  # [8,8,4,640],[8,16,1,102]
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
            self.Axis_Horizontal_layer(out_Hide_features)
        out_w_MS, x_Qw_MS, x_Kw_MS, x_Vw_MS, gamma_w_MS, Attention_w_wight_MS = \
            self.Axis_Portrait_layer(out_Hide_features)
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
        Attention_matrix_h = Attention_matrix_h.view(batch, channel, Hight, Width) * gamma_h_fusion1  # 加入权重参数，防止数据过大

        Attention_matrix_w = torch.matmul(x_Vw, Attention_w_wight_fusion1)  # Vw*weight——>bx(c/r*H)xW
        Attention_matrix_w = Attention_matrix_w.view(batch, channel, Hight, Width) * gamma_w_fusion2  # 加入权重参数，防止数据过大

        # 注意力矩阵拼接
        Attention_matrix1 = torch.cat([Attention_matrix_w, Attention_matrix_h], dim=1)  # [8,128,1,102]
        Attention_matrix = self.layer_softmax3(self.layer_fusion4(Attention_matrix1))  # [8,16,1,102]
        # 两条特征支路提取的特征进融合（拼接）
        identity1 = torch.cat([sc_AE_features, out_Hide_features], dim=1)
        identity = self.layer_relu1(self.layer_fusion5(identity1))

        out_method1 = torch.mul(identity, Attention_matrix)  # [8,16,1,102]

        """
        融合方式二：
            （1）对axis处理后的权重直接融合（拼接），即不进行底层q、k、v的交互和融合
            （2）attention直接融合归一化
            （3）输出：out_method2
            (4) 水平拼接，采用线性映射
        """
        # [8,2,1]-->[8,2]-->[8,1,1]
        Attention_h_wight = torch.cat([Attention_h_wight_SC, Attention_h_wight_MS], dim=1). \
            view(-1, 2 * (Attention_h_wight_SC.shape[1]) * Attention_h_wight_SC.shape[2])
        Attention_h_wight = self.layer_line1(Attention_h_wight)
        Attention_h_wight = Attention_h_wight.reshape(8, Attention_h_wight_SC.shape[1], Attention_h_wight_SC.shape[2])
        # [8,102,102]-->[8,2*102,102]-->[8,102,102]
        Attention_w_wight = torch.cat([Attention_w_wight_SC, Attention_w_wight_MS], dim=1). \
            view(-1, 2 * (Attention_w_wight_SC.shape[1]) * Attention_w_wight_SC.shape[2])
        Attention_w_wight = self.layer_line2(Attention_w_wight)
        Attention_w_wight = Attention_w_wight.reshape(8, Attention_w_wight_SC.shape[1], Attention_w_wight_SC.shape[2])
        # 权重矩阵融合
        gamma_h_fusion3 = self.gamma_h2
        gamma_w_fusion4 = self.gamma_w2
        Attention_matrix_h2 = torch.matmul(x_Vh, Attention_h_wight)  # Vw*weight——>bx(c/r*H)xW
        Attention_matrix_h2 = Attention_matrix_h2.view(batch, channel, Hight, Width) * gamma_h_fusion3  # 加入权重参数，防止数据过大

        Attention_matrix_w2 = torch.matmul(x_Vw, Attention_w_wight)  # Vw*weight——>bx(c/r*H)xW
        Attention_matrix_w2 = Attention_matrix_w2.view(batch, channel, Hight, Width) * gamma_w_fusion4  # 加入权重参数，防止数据过大
        # 注意力矩阵拼接
        Attention_matrix1_2 = torch.cat([Attention_matrix_w2, Attention_matrix_h2], dim=1)  # [8,32,1,102]
        Attention_matrix_2 = self.layer_softmax4(
            self.layer_fusion6(Attention_matrix1_2))  # [8,16,1,102]   # 此处是否需要重建樱色映射参数
        # 两条特征支路提取的特征进融合（拼接）
        identity1_2 = torch.cat([sc_AE_features, out_Hide_features], dim=1)
        identity2 = self.layer_relu2(self.layer_fusion7(identity1_2))

        out_method2 = torch.mul(identity2, Attention_matrix_2)  # [8,16,1,102]
        """
        融合方式三：
            (1)注意力机制的输出权重矩阵精聚合，
            (2)out_h，out_w直接融合拼接
            (3)采用压缩通道卷积降采样降维
        """
        Attention_matrix1_3 = torch.cat([out_h_SC, out_w_SC, out_w_SC, out_w_MS], dim=1)  # [8,512,1,102]
        Attention_matrix_3 = self.layer_softmax5(self.layer_fusion8(Attention_matrix1_3))  # [8,128,1,102]
        # 方式三采用方式的隐藏层输出特征映射矩阵
        out_method3 = torch.mul(identity2, Attention_matrix_3)  # [8,16,1,102]
        # 此处是否需要重建映射参数
        return out_method1, out_method2, out_method3  # 输出尺寸[8,16,1,102]
