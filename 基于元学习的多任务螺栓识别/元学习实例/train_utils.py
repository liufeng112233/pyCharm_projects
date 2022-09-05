import torch
import torch.nn as nn
import numpy as np
import random
from torch.backends import cudnn
import csv
from time import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Iterable
from math import log, exp
from itertools import islice  # 跳过前几行读取文件内容
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def random_seed(seed):    # 固定参数的函数
#     torch.manual_seed(1)  # 固定torch中的参数  相当于固定随机初始化的参数，这样每次训练的时候他都是从同一个地方开始初始化参数，有助于调试程序解决过拟合问题。1可以随便设置
#     np.random.seed(1)     # 固定np中的参数 同上


def random_seed(seed: int = 0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # print(f"set random seed: {seed}")
        # 下面两项比较重要，搭配使用。以精度换取速度，保证过程可复现。
        # https://pytorch.org/docs/stable/notes/randomness.html
        cudnn.benchmark = False  # False: 表示禁用
        cudnn.deterministic = True  # True: 每次返回的卷积算法将是确定的，即默认算法。
        # cudnn.enabled = True


def weights_init(m):      # 初始化权重的函数
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.0)  # BN的权重设为1
        nn.init.constant_(m.bias, 0.0)    # BN的偏差设为0


def file2matrix(filename, num, length):  # 读取文件的函数
    """

    :param length: sample length
    :param filename: file path
    :param num: output sample numbers
    :return: (n, length)
    """
    data_num = num*length         # 所要读取的数据量（扶梯每个文件有1280000个数据，我现在要取300个样本，每个样本包含1024个数据，相当于读取的数据量就是300*1024）
    start_line = 20                                               # 设定从第20行开始读取数据，其实文件里第17行开始就有数据了，只要设的值大于16就可以
    fr = open(filename)                                           # 打开文件
    array_lines = fr.readlines()[start_line:start_line+data_num]  # 从文件中读数据  从20行开始，20+300*1024行结束
    return_mat = np.zeros((data_num, 1))                          # 构造矩阵存放数据（data_num行1列的零矩阵）
    index = 0                                                     # 定义索引
    for line in array_lines:                                      # 对每一行进行处理
        line = line.strip()                                       # 去除行的尾部的换行符   strip()移除字符串头尾指定的序列
        list_from_line = line.split('\t')                         # 将一行数据按空进行分割  \t:横向制表符
        return_mat[index, :] = list_from_line[1:2]                # 下标为index的所有值
        index += 1  # 索引加1  (data_num, 1)

    p = 0  # 数据的标准化处理
    q = 0
    n = len(return_mat)
    while p < n:
        q += return_mat[p]
        p += 1
    s = q / n  # 均值
    p = 0
    q = 0
    while p < n:
        q += (return_mat[p] - s) * (return_mat[p] - s)
        p += 1
    ss = np.sqrt(q / (n - 1)+10**(-8))  # 标准差
    p = 0
    while p < n:
        return_mat[p] = (return_mat[p] - s) / ss
        p += 1
    # return return_mat
    return return_mat.reshape(num, length)  # (num=300, length=1024)(300,1024)


def new_ck(data):
    n = len(data)
    mean = 0
    var = 0
    square_root = 0
    root_mean_square = 0
    kurtosis = 0
    skewness = 0
    ck0 = np.zeros((1, 16))
    for i in range(n):
        mean += data[i]
    mean = mean/n                                   # 均值
    max = np.max(data)                         # 最大值
    peak = np.max(np.abs(data))                # 峰值
    min = np.min(data)                         # 最小值
    for j in range(n):
        var += (data[j]-mean)**2
    var = var/(n-1)                                 # 样本方差
    standard = np.sqrt(var)                         # 样本标准差
    for k in range(n):
        square_root += np.sqrt(np.abs(data[k]))
    square_root = (square_root/n)**2                # 方根幅值
    for m in range(n):
        root_mean_square += data[m]**2
    root_mean_square = np.sqrt(root_mean_square/n)  # 均方根幅值
    for p in range(n):
        kurtosis += (data[p]-mean)**4
    kurtosis = kurtosis/n                           # 峭度
    for o in range(n):
        skewness += (data[o]-mean)**3
    skewness = skewness/n                           # 偏斜度
    waveform_index = root_mean_square/mean          # 波形指标
    peak_index = peak/root_mean_square              # 峰值指标
    pulse_index = peak/mean                         # 脉冲指标
    margin_index = peak/square_root                 # 裕度指标
    skewness_index = skewness/(standard**3)         # 偏斜度指标
    kurtosis_index = kurtosis/(standard**4)         # 峭度指标
    ck = np.hstack((mean, peak, max, min, var, standard, square_root, root_mean_square, kurtosis, skewness, waveform_index, peak_index, pulse_index, margin_index, skewness_index, kurtosis_index))
    return ck


def new_episode(x, bs):                       # x代表要打乱的矩阵，bs代表取得batch_size
    select_item = torch.randperm(x.shape[0])  # x代表几行几列，shape[0]代表行，shape[1]代表列  randperm对行数进行打乱
    select_item2 = select_item[0:bs]          # 打乱以后取前bs个打乱的数据
    train_xx = x[select_item2]                # train_xx代表提取到的打乱后的数据
    return train_xx


def prototype(aa, dim, sample, m):
    aa = aa.detach().cpu().numpy()  # 将从网络中得到的支持集变为numpy形式
    ck = np.zeros((1, dim))
    for j in range(dim):
        for i in range(sample*m, sample*(m+1)):
            ck[0][j] += aa[i][j]
        ck[0][j] /= sample
    # for t in range(batch_size):
    #     ck[t][j] = ck[0][j]
    ck = torch.from_numpy(ck).to(device)
    return ck


def softplus(x):
    x = 1+(np.log(1+np.exp(-x))+x)
    return x


def sigmoid(y):
    y = 1+(1/(1+np.exp(-y)))
    return y


def ma_distance(hou, yang, z):
    # a = x-y
    # right = torch.mul(Sc.repeat(1, 6, 1), a.transpose((1, 2)))
    # d = torch.sqrt(torch.mm(a, right)+0.000000001)

    a = hou - yang  # (5,6)
    b = a.unsqueeze(1)  # (5,1,6)
    c = b.transpose(1, 2)  # (5,6,1)
    # right = torch.mul(z.transpose(0, 1), c)  # SQ gauss prototype D
    right = torch.mul(z.unsqueeze(1), c)  # (5,6,1) SQ gauss prototype
    d = torch.sqrt(torch.bmm(b, right) + 0.000000001)  # (5,1,1)
    f = d.squeeze(1).squeeze(1)
    # ddd = torch.sqrt(torch.sum((aaa - bbb) * (aaa - bbb), dim=-1))
    return f


def ma_Ddistance(hui, fang, zhang):
    ttt = hui.to(device) - fang.to(device)  # (5, 64)
    u = ttt.unsqueeze(1)  # (5, 1, 64)
    w = u.transpose(1, 2)  # (5, 64, 1)
    v = torch.mul(zhang.to(device).unsqueeze(1), u)  # (5, 1, 64)
    y = torch.sqrt(torch.bmm(v, w) + 0.000000001)  # (5, 1, 1)
    q = y.squeeze(1).squeeze(1)
    del ttt, u, w, v, y
    return q


def covariance(x):  # 定义如何求协方差矩阵   样本协方差矩阵
    x = x.unsqueeze(-1)
    a = torch.mean(x, dim=1)  # 竖着相加求均值 (7,1,1)
    b = a.reshape(7, 1).unsqueeze(-1).repeat(1, 5, 1)  # (7,1,1)->(7,5,1)
    c = x - b  # (7,5,1)
    d = c.transpose(1, 2)  # (7,1,5)
    e = ((torch.bmm(d, c))/4).reshape(7, -1)  # (7,1,1)->(7,1)
    g = 1 / e  # (7,1)  求逆后的原始协方差
    # S = softplus(g)
    return g


def gauss_prototype(ru, jie, hdim, hsample, mm):
# def gauss_prototype(support_covr, support_vector):
    # ee = support_covr.unsqueeze(-1).repeat(1, 1, 6)
    # aa = torch.mul(ee, support_vector)  # (7,5,6)
    # bb = torch.sum(aa, dim=1)  # (7,1,6)
    # ff = bb.unsqueeze(1)
    # cc = torch.sum(support_covr.unsqueeze(-1), dim=1)  # (7,1,1)
    # gg = cc.unsqueeze(-1)
    # dd = ff / gg  # (7,1,6)
    ru = ru.detach().cpu().numpy()  # 将从网络中得到的支持集变为numpy形式
    jie = jie.detach().cpu().numpy()
    ck = np.zeros((1, hdim))
    ruru = 0
    jiejie = 0
    for j in range(hdim):
        for i in range(hsample * mm, hsample * (mm + 1)):
            ruru += jie[i]*ru[i]
            jiejie += jie[i]
        ck = ruru/(jiejie+0.0000000001)
    ck = torch.from_numpy(ck).to(device)
    del ru, jie, i, j
    return ck.unsqueeze(0)


def Sc(tt, rsample, rmm):
    support_covr = tt.detach().cpu().numpy()
    Sraw = 0
    for i in range(rsample * rmm, rsample * (rmm+1)):
        Sraw += support_covr[i]
    Sc = sigmoid(Sraw)
    # Sc = softplus(Sraw)  # (7,1)
    Sc = torch.from_numpy(Sc)
    del support_covr, Sraw, i
    return Sc.unsqueeze(0).to(device)


def fang_cha(xick, gec_ck, madim):
    xi_ck = 0
    for r in range(madim-1):
        xi_ck += (xick[0][r]-gec_ck)**2
    fang_chaa = xi_ck/(madim-1)
    return torch.sqrt(fang_chaa)


def hhsoftmax(dist, n_class):  # 利用softmax进行归一化
    distt = torch.zeros((5, n_class))  # (5, 10)
    a = 0
    # b = torch.max(dist, 1)[0]  # 只返回每一行中最大的那个数
    # dist = dist-b
    for i in range(5):
        for j in range(n_class):
            a += torch.exp(dist[i][j])
        for j in range(n_class):
            distt[i][j] = torch.exp(dist[i][j])/(a+0.00000000001)
    return distt


def transform_1(label_vstack, n_class):  # 标签转换为0-1的形式
    label_t = np.zeros((len(label_vstack), n_class))  # (950, 10)
    for i in range(len(label_vstack)):
        label_t[i][int(label_vstack[i][0])] = 1
    return label_t


def oversampling(tr_outer1, nn, mm):  # 较多样本进行过采样保持与最大样本数的类别一致
    tr_ou1 = torch.randperm(tr_outer1.shape[0])
    select_ou1 = tr_ou1[0:nn]  # 挑选nn=5个扩大mm=3倍
    train_ou1 = tr_outer1[select_ou1, :]  # (5, 4096)
    train_ou1 = train_ou1.repeat(mm, axis=0)  # (15, 4096)
    train_out1 = np.vstack((tr_outer1, train_ou1))  # (35, 4096)
    np.random.shuffle(train_out1)
    return train_out1


def gauss_lishu_function(batch, ldim, query, c_ck, xi_ck):
    lk = torch.zeros((batch, ldim-1))  # (5,63)
    for mm in range(batch):
        for kk in range(ldim-1):
            zhou = (query[mm][kk]-c_ck)**2
            yu = (-zhou)/(2*(0.01**2)+0.0000000001)
            lk[mm][kk] = torch.exp(yu)
    lkk = torch.sum(lk, dim=1)
    return lkk/(ldim-1)  # (5,1)


def hsoftmax(x):  # 用softmax归一化的过程
    p = x-torch.max(x)
    p = torch.exp(p)/torch.sum(torch.exp(p))
    return p


def bnn(x):   # 标准化的过程
    a = torch.mean(x)
    b = torch.sqrt(((x-a)**2)/(len(x)-1))
    c = (x-a)/b
    return c


def eucliden(a, b):	     # 计算两个tensor的欧氏距离
    a = a.to(device)
    b = b.to(device)
    d = torch.sqrt(torch.sum((a-b)*(a-b), dim=-1))  # 按列进行计算
    return d


def antenna(filename):  # 定义读取天线数据集的函数
    data_file = open(filename)
    data_list = csv.reader(data_file)
    data = []
    for i in data_list:
        data.append(i)
    data_file.close()
    le = len(data)
    dataset = np.zeros(le)
    p = 0
    while p < le:
        data1 = ','.join(data[p])  # 给每个数据后面加上逗号
        all_values = data1.split(',')  # 根据数据的逗号分割开
        dataset[p] = float(all_values[0])
        p += 1
    # aaa = np.min(dataset)
    # bbb = np.max(dataset)
    # m = 0
    # while m < le:
    #     dataset[m] = (dataset[m]-aaa)/(bbb-aaa)
    # return dataset
    m = 0
    n = 0
    while m < le:
        n = n+dataset[m]
        m += 1
    k = n/le  # 求均值
    m = 0
    n = 0
    while m < le:
        n = n+(dataset[m]-k)*(dataset[m]-k)
        m += 1
    kk = np.sqrt((n/(le-1)+10**(-8)))  # 标准值
    m = 0
    while m < le:
        dataset[m] = (dataset[m]-k)/kk
        m += 1
        return dataset


def collect_antenna(dataset, ll, m, label):  # 定义划分天线数据集的函数
    # ==================================ll:样本长度 m:样本间距
    x = (len(dataset)-ll+m)//m  # x代表样本数,划分了x个样本
    a0 = label*np.ones((x, ll+1))  # (x,l+1)最后一列是标签
    i = 0
    while i < x:
        j = 0
        while j < ll:
            a0[i][j] = float(dataset[m*i+j])
            j += 1
        i += 1
    return a0


# class SAM(torch.optim.Optimizer):
#     def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
#         assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
#         defaults = dict(rho=rho, **kwargs)
#         super(SAM, self).__init__(params, defaults)
#         self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
#         self.param_groups = self.base_optimizer.param_groups
#
#     def first_step(self, zero_grad=False):
#         grad_norm = self._grad_norm()
#         for group in self.param_groups:
#             scale = group['rho']/(grad_norm+1e-12)
#
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 e_w = p.grad*scale.to(p)
#                 p = p+e_w
#                 # p.add_(e_w)
#                 self.state[p][e_w] = e_w
#
#         if zero_grad:
#             self.zero_grad()
#
#     def second_step(self, zero_grad=False):
#         # grad_norm = self._grad_norm()
#         for group in self.param_groups:
#             # scale = group['rho'] / (grad_norm + 1e-12)
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 # e_w = p.grad * scale.to(p)
#                 # p = p-e_w
#                 p.sub_(self.state[p][{"e_w"}])
#         self.base_optimizer.step()
#
#         if zero_grad:
#             self.zero_grad()
#
#     def _grad_norm(self):
#         shared_device = self.param_groups[0]['params'][0].device
#         norm = torch.norm(torch.stack([p.grad.norm(p=2).to(shared_device)
#                                        for group in self.param_groups for p in group['params']
#                                        if p.grad is not None]),
#                           p=2)
#         return norm


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm1d(input_channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()
        # self.conv2 = nn.Conv1d(output_channels, output_channels, 3, 1, padding=1)
        # self.bn3 = nn.BatchNorm1d(output_channels)
        # self.relu = nn.ReLU()
        # self.conv3 = nn.Conv1d(output_channels, output_channels, 3, 1, padding=1)
        # self.bn4 = nn.BatchNorm1d(output_channels)
        # self.relu = nn.ReLU()
        self.conv3 = nn.Conv1d(output_channels, output_channels, 3, 1, padding=1)  # 尝试一下看瓶颈结构行不行
        self.conv4 = nn.Conv1d(input_channels, output_channels, 3, 1, padding=1)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        # out = self.conv2(out)
        # out = self.bn3(out)
        # out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride != 1):
            residual = self.conv4(out1)
        out += residual
        return out


# class attention(nn.Module):
#     def __init__(self, input_channel, output_channel):
#         super(attention, self).__init__()
#         self.input_channel = input_channel
#         self.output_channel = output_channel
#         self.conv1 = nn.Sequential(nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=1, padding=0),
#                                    nn.BatchNorm1d(output_channel),
#                                    nn.Sigmoid())
#         self.conv2 = nn.Sequential(nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=1, padding=1),
#                                    nn.ReLU())
#
#     def forward(self, x):
#         residual = x
#         a = self.conv1(x)
#         b = self.conv2(x)
#         out = torch.mul(a, b)  # 矩阵对应点相乘
#         out = residual+out
#         return out


class position_attention(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(position_attention, self).__init__()
        self.input_channel = input_channel
        self.out_channel = output_channel
        self.conv1 = nn.Sequential(nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm1d(output_channel),
                                   nn.Softmax(dim=2))  # 5*16*4096
        self.conv2 = nn.Sequential(nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm1d(output_channel),
                                   nn.ReLU())

    def forward(self, x):
        residual = x
        a = self.conv1(x)
        b = self.conv2(x)
        # c = self.conv1(x)
        # aa = torch.transpose(a, 1, 2)  # 5*4096*16
        # bb = torch.bmm(aa, b)  # 矩阵乘法 5*4096*4096
        # bb = F.softmax(bb, dim=-1)  # 5*4096*4096
        c = torch.mul(a, b)  # 5*16*4096
        out = residual+c
        return out


def new_prototype(ck_i, ck0, ck1, ck2, ck3):
    d00 = eucliden(ck_i, ck0); d01 = eucliden(ck_i, ck1); d02 = eucliden(ck_i, ck2); d03 = eucliden(ck_i, ck3);
    x = torch.cat((d00, d01, d02, d03))
    d = torch.sort(x)
    if d[0][1] == d00:
        ck_i = ck0
    elif d[0][1] == d01:
        ck_i = ck1
    elif d[0][1] == d02:
        ck_i = ck2
    elif d[0][1] == d03:
        ck_i = ck3
    # elif d[0][1] == d04:
    #     ck_i = ck4
    # elif d[0][1] == d05:
    #     ck_i = ck5
    # else:
    #     ck_i = ck6
    return ck_i


# class channel_attention(nn.Module):
#     def __init__(self, input_channel, output_channel):
#         super(channel_attention, self).__init__()
#         self.input_channel = input_channel
#         self.output_channel = output_channel  # 5*16*4096
#         # self.aa = nn.Softmax(dim=1)
#         self.conv1 = nn.Sequential(nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=1, padding=0),
#                                    nn.BatchNorm1d(output_channel),
#                                    nn.Softmax(dim=1))  # 5*252*256
#         self.conv2 = nn.Sequential(nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=1, padding=1),
#                                    nn.BatchNorm1d(output_channel),
#                                    nn.ReLU())
#
#     def forward(self, x):
#         mm = x
#         # trans = torch.transpose(x, 1, 2)  # 5*4096*16
#         # uu = torch.bmm(x, trans)  # 5*16*16
#         # uu = F.softmax(uu, dim=-1)  # 5*16*16
#         uu = self.conv1(x)
#         vv = self.conv2(x)
#         # uu = self.aa(x)
#         ww = torch.mul(uu, vv)  # 5*16*4096
#         out2 = mm+ww
#         return out2


def plot_embedding(data, label, title):
    # plt.rcParams['figure.dpi'] = 500
    fig = plt.figure()
    ax = plt.subplot(111)
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    type4_x = []
    type4_y = []
    type5_x = []
    type5_y = []
    type6_x = []
    type6_y = []
    type7_x = []
    type7_y = []
    # type8_x = []
    # type8_y = []
    # type9_x = []
    # type9_y = []
    # type10_x = []
    # type10_y = []
    # type11_x = []
    # type11_y = []
    # type12_x = []
    # type12_y = []
    # type13_x = []
    # type13_y = []
    for i in range(data.shape[0]):
        if label[i] == 0:
            type1_x.append(data[i][0])
            type1_y.append(data[i][1])
        if label[i] == 1:
            type2_x.append(data[i][0])
            type2_y.append(data[i][1])
        if label[i] == 2:
            type3_x.append(data[i][0])
            type3_y.append(data[i][1])
        if label[i] == 3:
            type4_x.append(data[i][0])
            type4_y.append(data[i][1])
        if label[i] == 4:
            type5_x.append(data[i][0])
            type5_y.append(data[i][1])
        # if label[i] == 5:
        #     type6_x.append(data[i][0])
        #     type6_y.append(data[i][1])
        # if label[i] == 6:
        #     type7_x.append(data[i][0])
        #     type7_y.append(data[i][1])
        # if label[i] == 7:
        #     type8_x.append(data[i][0])
        #     type8_y.append(data[i][1])
        # if label[i] == 8:
        #     type9_x.append(data[i][0])
        #     type9_y.append(data[i][1])
        # if label[i] == 9:
        #     type10_x.append(data[i][0])
        #     type10_y.append(data[i][1])
        # if label[i] == 10:
        #     type11_x.append(data[i][0])
        #     type11_y.append(data[i][1])
        # if label[i] == 11:
        #     type12_x.append(data[i][0])
        #     type12_y.append(data[i][1])
        # if label[i] == 12:
        #     type13_x.append(data[i][0])
        #     type13_y.append(data[i][1])

        type1 = plt.scatter(type1_x, type1_y, s=50, c='r', marker='o', alpha=0.5)  # s指的是点的大小
        type2 = plt.scatter(type2_x, type2_y, s=50, c='g', marker='+', alpha=0.5)
        type3 = plt.scatter(type3_x, type3_y, s=50, c='b', marker='*', alpha=0.5)
        type4 = plt.scatter(type4_x, type4_y, s=50, c='k', marker='^', alpha=0.5)
        type5 = plt.scatter(type5_x, type5_y, s=50, c='c', marker='<', alpha=0.5)
        # type6 = plt.scatter(type6_x, type6_y, s=50, c='m', marker='>', alpha=0.5)  # 品红色
        # type7 = plt.scatter(type7_x, type7_y, s=50, c='y', marker='v', alpha=0.5)
        # type8 = plt.scatter(type8_x, type8_y, s=50, c='gold', marker='D', alpha=0.5)  # 金 菱形
        # type9 = plt.scatter(type9_x, type9_y, s=50, c='violet', marker='s', alpha=0.5)  # 粉 正方形
        # type10 = plt.scatter(type10_x, type10_y, s=50, c='coral', marker='p', alpha=0.5)  # 偏橙 五角星
        # type11 = plt.scatter(type11_x, type11_y, s=50, c='pink', marker='h', alpha=0.5)  # 六角星
        # type12 = plt.scatter(type12_x, type12_y, s=50, c='darkviolet', marker='x', alpha=0.5)  # 乘号
        # type13 = plt.scatter(type13_x, type13_y, s=50, c='lightskyblue', marker='8', alpha=0.5)  # 8角形

        # plt.legend((type1, type2), ('wear', 'normal'), loc='best')
        plt.legend((type1, type2, type3, type4, type5), ('NC', 'IF-1', 'IF-2', 'OF-1', 'OF-2'), loc='best')
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.title(title)
        # plt.rcParams['figure.dpi'] = 500
        plt.savefig("C:/Users/asus/Desktop/T-SNE-5.jpg", dpi=600)
    return plt.show(fig)


def plot_2D(data):
    print('Computing T-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)  # 使用TSNE对特征降到二维
    result = tsne.fit_transform(data)  # 降维后的数据
    # fig = plot_embedding(result, label, 'T-SNE embedding of the digits')
    # fig.subplots_adjust(right=0.8)  # 图例过大，保存figure时无法保存完全，故对此参数调整
    result = MinMaxScaler().fit_transform(result)
    return result


if __name__ == "__main__":  # 加这句话 从其他程序里调用这个py程序不会运行这句话以后的  意思是可以只调用前面的包  这句话后面可以用来对前面的包进行测试
    weights_init(1)