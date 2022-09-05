import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from my_utils.train_utils import random_seed, file2matrix, weights_init, gauss_prototype, ma_distance, Sc, ma_Ddistance, ResidualBlock, position_attention, plot_2D, plot_embedding, hhsoftmax, transform_1, oversampling
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle
from sklearn.metrics import confusion_matrix

random_seed(1)  # 固定参数的函数
num_tr = 80
num_va = 20
num_te = 100
data_len = 2048

# 内圈1
innera1 = file2matrix("F:/机器学习与深度学习视频/实验室数据/SQ轴承故障数据/SQ轴承故障数据/内圈1/REC" + str(3596) + "_ch2.txt", 100, data_len)
innerb1 = file2matrix("F:/机器学习与深度学习视频/实验室数据/SQ轴承故障数据/SQ轴承故障数据/内圈1/REC" + str(3595) + "_ch2.txt", 100, data_len)
inner1 = np.vstack((innera1, innerb1))  # (200,2048)
tr_inner1 = inner1[0:num_tr]  # (30,2048)
# tr_inner1 = inner1[0:5]
va_inner1 = inner1[num_tr:num_tr + num_va]  # (20,2048)
te_inner1 = inner1[100:200]  # (150,2048)
# 内圈2
innera2 = file2matrix("F:/机器学习与深度学习视频/实验室数据/SQ轴承故障数据/SQ轴承故障数据/内圈2/REC" + str(3618) + "_ch2.txt", 100, data_len)
innerb2 = file2matrix("F:/机器学习与深度学习视频/实验室数据/SQ轴承故障数据/SQ轴承故障数据/内圈2/REC" + str(3617) + "_ch2.txt", 100, data_len)
inner2 = np.vstack((innera2, innerb2))  # (200,2048)
# tr_inner2 = inner2[0:num_tr]  # (30,2048)
# tr_inner2 = inner2[0:15]
va_inner2 = inner2[num_tr:num_tr + num_va]  # (20,2048)
te_inner2 = inner2[100:200]  # (150,2048)
# 内圈3
# innera3 = file2matrix("F:/机器学习与深度学习视频/实验室数据/SQ轴承故障数据/SQ轴承故障数据/内圈3/REC" + str(3531) + "_ch2.txt", 100, data_len)
# innerb3 = file2matrix("F:/机器学习与深度学习视频/实验室数据/SQ轴承故障数据/SQ轴承故障数据/内圈3/REC" + str(3530) + "_ch2.txt", 100, data_len)
# inner3 = np.vstack((innera3, innerb3))  # (200,2048)
# # tr_inner3 = inner3[0:num_tr]  # (30,2048)
# tr_inner3 = inner3[0:25]
# va_inner3 = inner3[num_tr:num_tr + num_va]  # (20,2048)
# te_inner3 = inner3[100:200]  # (150,2048)

# 外圈1
outera1 = file2matrix("F:/机器学习与深度学习视频/实验室数据/SQ轴承故障数据/SQ轴承故障数据/外圈1/REC" + str(3511) + "_ch2.txt", 100, data_len)
outerb1 = file2matrix("F:/机器学习与深度学习视频/实验室数据/SQ轴承故障数据/SQ轴承故障数据/外圈1/REC" + str(3510) + "_ch2.txt", 100, data_len)
outer1 = np.vstack((outera1, outerb1))
tr_outer1 = outer1[0:num_tr]  # (30,2048)
# tr_outer1 = outer1[0:25]
va_outer1 = outer1[num_tr:num_tr + num_va]  # (20,2048)
te_outer1 = outer1[100:200]  # (150,2048)
# 外圈2
outera2 = file2matrix("F:/机器学习与深度学习视频/实验室数据/SQ轴承故障数据/SQ轴承故障数据/外圈2/REC" + str(3493) + "_ch2.txt", 100, data_len)
outerb2 = file2matrix("F:/机器学习与深度学习视频/实验室数据/SQ轴承故障数据/SQ轴承故障数据/外圈2/REC" + str(3492) + "_ch2.txt", 100, data_len)
outer2 = np.vstack((outera2, outerb2))
# tr_outer2 = outer2[0:num_tr]  # (30,2048)
# tr_outer2 = outer2[0:35]
va_outer2 = outer2[num_tr:num_tr + num_va]  # (20,2048)
te_outer2 = outer2[100:200]  # (150,2048)
# 外圈3
# outera3 = file2matrix("F:/机器学习与深度学习视频/实验室数据/SQ轴承故障数据/SQ轴承故障数据/外圈3/REC" + str(3475) + "_ch2.txt", 100, data_len)
# outerb3 = file2matrix("F:/机器学习与深度学习视频/实验室数据/SQ轴承故障数据/SQ轴承故障数据/外圈3/REC" + str(3474) + "_ch2.txt", 100, data_len)
# outer3 = np.vstack((outera3, outerb3))
# # tr_outer3 = outer3[0:num_tr]  # (30,2048)
# tr_outer3 = outer3[0:55]
# va_outer3 = outer3[num_tr:num_tr + num_va]  # (20,2048)
# te_outer3 = outer3[100:200]  # (150,2048)

# 正常
normala = file2matrix("F:/机器学习与深度学习视频/实验室数据/SQ轴承故障数据/SQ轴承故障数据/正常/REC" + str(3640) + "_ch2.txt", 100, data_len)
normalb = file2matrix("F:/机器学习与深度学习视频/实验室数据/SQ轴承故障数据/SQ轴承故障数据/正常/REC" + str(3639) + "_ch2.txt", 100, data_len)
normal = np.vstack((normala, normalb))
tr_normal = normal[0:num_tr]
# tr_normal = normal[0:45]
va_normal = normal[num_tr:num_tr + num_va]
te_normal = normal[100:200]



# tr_in1 = torch.randperm(tr_inner1.shape[0])  # (80, 4096)打乱的是80里
# select_in1 = tr_in1[0:5]
# train_in1 = tr_inner1[select_in1, :]  # (5, 4096)
#
# tr_in2 = torch.randperm(tr_inner2.shape[0])
# select_in2 = tr_in2[0:5]
# train_in2 = tr_inner2[select_in2, :]
#
# tr_outer1 = oversampling(tr_outer1, 5, 4)
# tr_outer2 = oversampling(tr_outer2, 5, 2)

n_class = 3
# ========================add labels and separate
# to train
tr_data = np.vstack((tr_inner1, tr_outer1, tr_normal))
# tr_data = np.vstack((tr_inner1, tr_inner2, tr_inner3, tr_outer1, tr_outer2, tr_outer3, tr_normal))  # (210,2048)
labels0 = np.zeros((80, 1))  # (30,1)
labels1 = np.ones((80, 1))
labels2 = 2 * np.ones((80, 1))
# labels3 = 3 * np.ones((65, 1))
# labels4 = 4 * np.ones((num_tr, 1))
# labels5 = 5 * np.ones((num_tr, 1))
# labels6 = 6 * np.ones((num_tr, 1))
labels = np.vstack((labels0, labels1, labels2))
# labels = np.vstack((labels0, labels1, labels2, labels3, labels4, labels5, labels6))  # (210,1)
tr_data0 = np.hstack((tr_data, labels))  # (210,2049)

# to validation
va_data = np.vstack((va_inner1, va_inner2, va_outer1, va_outer2, va_normal))
# va_data = np.vstack((va_inner1, va_inner2, va_inner3, va_outer1, va_outer2, va_outer3, va_normal))  # (140,2048)
v_labels0 = np.zeros((num_va, 1))  # (20,1)
v_labels1 = np.ones((num_va, 1))
v_labels2 = 2 * np.ones((num_va, 1))
v_labels3 = 3 * np.ones((num_va, 1))
v_labels4 = 4 * np.ones((num_va, 1))
# v_labels5 = 5 * np.ones((num_va, 1))
# v_labels6 = 6 * np.ones((num_va, 1))
v_labels = np.vstack((v_labels0, v_labels1, v_labels2, v_labels3, v_labels4))
# v_labels = np.vstack((v_labels0, v_labels1, v_labels2, v_labels3, v_labels4, v_labels5, v_labels6))  # (140,1)
va_data0 = np.hstack((va_data, v_labels))  # (140,2049)

# to test
# te_data = np.vstack((te_inner1, te_inner2, te_inner3))
te_data = np.vstack((te_inner1, te_inner2, te_outer1, te_outer2, te_normal))  # (1050,2048)
t_labels0 = np.zeros((num_te, 1))  # (150,1)
t_labels1 = np.ones((num_te, 1))
t_labels2 = 2 * np.ones((num_te, 1))
t_labels3 = 3 * np.ones((num_te, 1))
t_labels4 = 4 * np.ones((num_te, 1))
# t_labels5 = 5 * np.ones((num_te, 1))
# t_labels6 = 6 * np.ones((num_te, 1))
# t_labels = np.vstack((t_labels0, t_labels1, t_labels2))
t_labels = np.vstack((t_labels0, t_labels1, t_labels2, t_labels3, t_labels4))  # (1050,1)
te_data0 = np.hstack((te_data, t_labels))  # (1050,2049)

dim = 64
D = dim * 2


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()   # 继承自父类的属性进行初始化
        self.cnn = nn.Sequential(nn.Conv1d(1, 32, kernel_size=9, stride=1, padding=4),
                                 nn.BatchNorm1d(32),
                                 nn.LeakyReLU(),
                                 nn.MaxPool1d(kernel_size=4),  # 1024

                                 nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3),
                                 nn.BatchNorm1d(64),
                                 nn.LeakyReLU(),
                                 nn.MaxPool1d(kernel_size=2),  # 512

                                 nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2),
                                 nn.BatchNorm1d(64),
                                 nn.LeakyReLU(),
                                 nn.MaxPool1d(2),  # 256

                                 nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(),
                                 nn.MaxPool1d(2),

                                 nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(),
                                 nn.MaxPool1d(2))  # 128

        self.dense = nn.Sequential(nn.Linear(128, 1024),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(1024),
                                   nn.Linear(1024, D))

    def forward(self, x):
        out = self.cnn(x)
        nn = torch.nn.AdaptiveAvgPool1d(1)  # 15*128*1
        result = nn(out)
        out = result.view(result.size(0), -1)    # 相当于numpy中的reshape,-1代表任意值，此处相当于平铺层
        out = self.dense(out)
        return out


class hou_loss(nn.Module):
    def __init__(self):
        super(hou_loss, self).__init__()

    def forward(self, x, y):
        mm = torch.zeros([batch_size, dim])  # 把标签和原型进行转换 (5,6)
        Scc = torch.zeros([batch_size, dim])  # (5,1)
        for h in range(batch_size):
            if y[h] == torch.tensor([0]).to(device):
                mm[h] = ck0
                Scc[h] = Sc0
            elif y[h] == torch.tensor([1]).to(device):
                mm[h] = ck1
                Scc[h] = Sc1
            elif y[h] == torch.tensor([2]).to(device):
                mm[h] = ck2
                Scc[h] = Sc2
            # elif y[h] == torch.tensor([3]).to(device):
            #     mm[h] = ck3
            #     Scc[h] = Sc3
            # elif y[h] == torch.tensor([4]):
            #     mm[h] = ck4
            #     Scc[h] = Sc4
            # elif y[h] == torch.tensor([5]):
            #     mm[h] = ck5
            #     Scc[h] = Sc5
            # else:
            #     mm[h] = ck6
            #     Scc[h] = Sc6
        # m1 = eucliden(x, mm)
        # m2 = eucliden(x, ck0) + eucliden(x, ck1) + eucliden(x, ck2) + eucliden(x, ck3)
        m1 = ma_Ddistance(x, mm, Scc)
        m2 = ma_Ddistance(x, ck0, Sc0)+ma_Ddistance(x, ck1, Sc1)+ma_Ddistance(x, ck2, Sc2)
        # m2 = ma_Ddistance(x, ck0, Sc0)+ma_Ddistance(x, ck1, Sc1)+ma_Ddistance(x, ck2, Sc2)+ma_Ddistance(x, ck3, Sc3)+ma_Ddistance(x, ck4, Sc4)+ma_Ddistance(x, ck5, Sc5)+ma_Ddistance(x, ck6, Sc6)
        # re = torch.mean(m1)
        # re = torch.mean(m1-m2)
        re = torch.mean(m1/(m2+0.0000000001), dim=0)
        return re


# 求4个类别的原型
batch_size = 5
model = Model()
model.to(device)
model.apply(weights_init)  # 初始化权重
criterion = hou_loss()
# base_optimizer = torch.optim.Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
# optimizer = SAM(model.parameters(), lr=0.1, rho=0.05)
epoch = 15
episode = 3

for n_epoch in range(epoch):
    print("[==================={}/{}=====================]".format(n_epoch, epoch))
    tr_data0 = tr_data0.reshape(3, 80, 2049)
    va_data0 = va_data0.reshape(5, 20, 2049)
    train = torch.randperm(tr_data0.shape[1])  # 打乱的是80里面的
    validation = torch.randperm(va_data0.shape[1])
    select_support = train[0:batch_size]  # 选前5个作为支持集的序列
    select_query = train[batch_size:]  # 剩下的作为查询集的序列
    va_support = validation[0:batch_size]
    va_query = validation[batch_size:]
    support = tr_data0[:, select_support, :]  # (3,5,4097)
    query = tr_data0[:, select_query, :]  # (7,75,2049)
    support_v = va_data0[:, va_support, :]  # (7,5,2049)
    query_v = va_data0[:, va_query, :]  # (7,15,2049)
    # 训练集
    support_x = torch.FloatTensor(support.reshape(-1, 2049))[:, 0:data_len].unsqueeze(1)  # (15, 4096)-->(15,1,4096)
    support_x = support_x.to(device)
    support_pred = model(support_x).reshape(n_class, 5, D)  # (15,128)->(3,5,128)
    support_vector = support_pred[:, :, :dim].reshape(-1, dim)  # (3,5,64)->(15,64)
    support_covr = support_pred[:, :, dim:].reshape(-1, dim)  # (3,5,64)->(15,64)
    # 验证集
    support_v_x = torch.FloatTensor(support_v.reshape(-1, 2049))[:, 0:data_len].unsqueeze(1)
    support_v_x = support_v_x.to(device)
    support_v_pred = model(support_v_x).reshape(5, 5, D)
    support_v_vector = support_v_pred[:, :, :dim].reshape(-1, dim)
    support_v_covr = support_v_pred[:, :, dim:].reshape(-1, dim)

    ck0 = gauss_prototype(support_vector, support_covr, dim, 5, 0)
    ck1 = gauss_prototype(support_vector, support_covr, dim, 5, 1)
    ck2 = gauss_prototype(support_vector, support_covr, dim, 5, 2)
    # ck3 = gauss_prototype(support_vector, support_covr, dim, 5, 3)
    # ck4 = gauss_prototype(support_vector, support_covr, dim, 5, 4)
    # ck5 = gauss_prototype(support_vector, support_covr, dim, 5, 5)
    # ck6 = gauss_prototype(support_vector, support_covr, dim, 5, 6)
    Sc0 = Sc(support_covr, 5, 0)
    Sc1 = Sc(support_covr, 5, 1)
    Sc2 = Sc(support_covr, 5, 2)
    # Sc3 = Sc(support_covr, 5, 3)
    # Sc4 = Sc(support_covr, 5, 4)
    # Sc5 = Sc(support_covr, 5, 5)
    # Sc6 = Sc(support_covr, 5, 6)

    vck0 = gauss_prototype(support_v_vector, support_v_covr, dim, 5, 0)
    vck1 = gauss_prototype(support_v_vector, support_v_covr, dim, 5, 1)
    vck2 = gauss_prototype(support_v_vector, support_v_covr, dim, 5, 2)
    vck3 = gauss_prototype(support_v_vector, support_v_covr, dim, 5, 3)
    vck4 = gauss_prototype(support_v_vector, support_v_covr, dim, 5, 4)
    # vck5 = gauss_prototype(support_v_vector, support_v_covr, dim, 5, 5)
    # vck6 = gauss_prototype(support_v_vector, support_v_covr, dim, 5, 6)
    # vck7 = gauss_prototype(support_v_vector, support_v_covr, dim, 5, 7)
    # vck8 = gauss_prototype(support_v_vector, support_v_covr, dim, 5, 8)
    # vck9 = gauss_prototype(support_v_vector, support_v_covr, dim, 5, 9)
    vSc0 = Sc(support_v_covr, 5, 0)
    vSc1 = Sc(support_v_covr, 5, 1)
    vSc2 = Sc(support_v_covr, 5, 2)
    vSc3 = Sc(support_v_covr, 5, 3)
    vSc4 = Sc(support_v_covr, 5, 4)
    # vSc5 = Sc(support_v_covr, 5, 5)
    # vSc6 = Sc(support_v_covr, 5, 6)
    # vSc7 = Sc(support_v_covr, 5, 7)
    # vSc8 = Sc(support_v_covr, 5, 8)
    # vSc9 = Sc(support_v_covr, 5, 9)

    for n_episode in range(episode):
        query = query.reshape(-1, 2049)  # (175, 2049)
        query_v = query_v.reshape(-1, 2049)  # (105, 2049)
        np.random.shuffle(query)  # 对查询集进行打乱
        np.random.shuffle(query_v)
        query_x = torch.FloatTensor(query[:, 0:data_len]).unsqueeze(1)  # (175,2048)-->(175,1,2048)
        query_y = torch.LongTensor(query[:, data_len])  # (175,1)
        query_v_x = torch.FloatTensor(query_v[:, 0:data_len]).unsqueeze(1)  # (105,1,2048)
        query_v_y = torch.LongTensor(query_v[:, data_len])  # (105,1)
        query_set = TensorDataset(query_x, query_y)
        query_loader = DataLoader(query_set, batch_size=batch_size, shuffle=True, num_workers=0)
        va_set = TensorDataset(query_v_x, query_v_y)
        va_loader = DataLoader(va_set, batch_size=batch_size, shuffle=True, num_workers=0)
        query_acc = 0
        query_loss = 0.0
        model.train()
        for i, data in enumerate(query_loader):
            query_pred = model(data[0])  # (5,128)->(5,64)
            query_true = query_pred[:, :dim]
            optimizer.zero_grad()  # 参数的梯度设为0
            loss = criterion(query_true, data[1])  # 误差计算
            loss.backward(retain_graph=True)  # 误差的反向传播
            optimizer.step()  # 参数更新
            dis = torch.zeros([batch_size, n_class])  # (5,3)
            for t in range(batch_size):
                dis[t][0] = ma_Ddistance(query_true[t], ck0, Sc0)
                dis[t][1] = ma_Ddistance(query_true[t], ck1, Sc1)
                dis[t][2] = ma_Ddistance(query_true[t], ck2, Sc2)
                # dis[t][3] = ma_Ddistance(query_true[t], ck3, Sc3)
                # dis[t][4] = ma_Ddistance(query_true[t], ck4, Sc4)
                # dis[t][5] = ma_Ddistance(query_true[t], ck5, Sc5)
                # dis[t][6] = ma_Ddistance(query_true[t], ck6, Sc6)
            query_acc += np.sum(np.argmin(dis.detach().cpu().numpy(), axis=1) == data[1].cpu().numpy())  # 按行取最小值的索引
            query_loss = loss.item()  # item()从标量里获取数字
        query_acc = 100 * query_acc / query_set.__len__()
        # accuracy_dict.append(query_acc)
        # loss_dict.append(query_loss)
        print("\n[{}/{}]".format(n_episode, episode))
        print("query Loss is :{:.4f}\tquery Acc is:{:.4f}%".format(query_loss, query_acc))

        model.eval()
        va_acc = 0
        va_loss = 0.0
        for j, data2 in enumerate(va_loader):
            data2[0] = data2[0].to(device)
            data2[1] = data2[1].to(device)
            va_true = model(data2[0])
            va_pred = va_true[:, :dim]
            # v_loss = criterion(va_pred, data2[1])
            disv = torch.zeros([batch_size, 5])
            for w in range(batch_size):
                disv[w][0] = ma_Ddistance(va_pred[w], vck0, vSc0)
                disv[w][1] = ma_Ddistance(va_pred[w], vck1, vSc1)
                disv[w][2] = ma_Ddistance(va_pred[w], vck2, vSc2)
                disv[w][3] = ma_Ddistance(va_pred[w], vck3, vSc3)
                disv[w][4] = ma_Ddistance(va_pred[w], vck4, vSc4)
                # disv[w][5] = ma_Ddistance(va_pred[w], vck5, vSc5)
                # disv[w][6] = ma_Ddistance(va_pred[w], vck6, vSc6)
                # disv[w][7] = ma_Ddistance(va_pred[w], vck7, vSc7)
                # disv[w][8] = ma_Ddistance(va_pred[w], vck8, vSc8)
                # disv[w][9] = ma_Ddistance(va_pred[w], vck9, vSc9)

            va_acc += np.sum(np.argmin(disv.detach().cpu().numpy(), axis=1) == data2[1].cpu().numpy())
            # va_loss = v_loss.item()
        va_acc = 100 * va_acc / va_set.__len__()
        print("val Loss is :{:.4f}\tval Acc is :{:.4f}%".format(va_loss, va_acc))

model.eval()
te_data0 = te_data0.reshape(5, 100, 2049)
# test = torch.randperm(te_data0.shape[1])
# te_support = test[0:batch_size]
# te_query = test[batch_size:]
support_t = te_data0[:, 0:5, :]  # (7,10,4097) 取前10个相当于已知的样本 用来作为支持集得到原型
query_t = te_data0[:, 5:, :]  # (7,90,4097)  后面的90个样本用来进行测试
support_te = torch.FloatTensor((support_t.reshape(-1, 2049))[:, 0:data_len]).unsqueeze(1)  # (70, 4096)-->(70,1,4096)
support_te = support_te.to(device)
support_te_pred = model(support_te).reshape(5, 5, D)  # (70,128)
support_te_vector = support_te_pred[:, :, :dim].reshape(-1, dim)  # (70,64)
support_te_covr = support_te_pred[:, :, dim:].reshape(-1, dim)
tck0 = gauss_prototype(support_te_vector, support_te_covr, dim, 5, 0)  # (1,6)
tck1 = gauss_prototype(support_te_vector, support_te_covr, dim, 5, 1)
tck2 = gauss_prototype(support_te_vector, support_te_covr, dim, 5, 2)
tck3 = gauss_prototype(support_te_vector, support_te_covr, dim, 5, 3)
tck4 = gauss_prototype(support_te_vector, support_te_covr, dim, 5, 4)
# tck5 = gauss_prototype(support_te_vector, support_te_covr, dim, 5, 5)
# tck6 = gauss_prototype(support_te_vector, support_te_covr, dim, 5, 6)
# tck7 = gauss_prototype(support_te_vector, support_te_covr, dim, 5, 7)
# tck8 = gauss_prototype(support_te_vector, support_te_covr, dim, 5, 8)
# tck9 = gauss_prototype(support_te_vector, support_te_covr, dim, 5, 9)
tSc0 = Sc(support_te_covr, 5, 0)
tSc1 = Sc(support_te_covr, 5, 1)
tSc2 = Sc(support_te_covr, 5, 2)
tSc3 = Sc(support_te_covr, 5, 3)
tSc4 = Sc(support_te_covr, 5, 4)
# tSc5 = Sc(support_te_covr, 5, 5)
# tSc6 = Sc(support_te_covr, 5, 6)
# tSc7 = Sc(support_te_covr, 5, 7)
# tSc8 = Sc(support_te_covr, 5, 8)
# tSc9 = Sc(support_te_covr, 5, 9)

query_t = query_t.reshape(-1, 2049)  # (1015, 2049)
np.random.shuffle(query_t)
query_t_x = torch.FloatTensor(query_t[:, 0:data_len]).unsqueeze(1)  # (1015,1,2048)
query_t_y = torch.LongTensor(query_t[:, data_len])  # (1015,1)
te_set = TensorDataset(query_t_x, query_t_y)
te_loader = DataLoader(te_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
te_acc = 0
te_loss = 0.0
aaa = []
bbb = []
mmm = []
nnn = []
data_tlist = []
label_tlist = []
for t, data3 in enumerate(te_loader):
    data3[0] = data3[0].to(device)
    data3[1] = data3[1].to(device)
    te_true = model(data3[0])
    te_pred = te_true[:, :dim]
    # t_loss = criterion(te_pred, data3[1])
    dist = torch.zeros([batch_size, 5])
    for v in range(batch_size):
        dist[v][0] = ma_Ddistance(te_pred[v], tck0, tSc0)
        dist[v][1] = ma_Ddistance(te_pred[v], tck1, tSc1)
        dist[v][2] = ma_Ddistance(te_pred[v], tck2, tSc2)
        dist[v][3] = ma_Ddistance(te_pred[v], tck3, tSc3)
        dist[v][4] = ma_Ddistance(te_pred[v], tck4, tSc4)
        # dist[v][5] = ma_Ddistance(te_pred[v], tck5, tSc5)
        # dist[v][6] = ma_Ddistance(te_pred[v], tck6, tSc6)
        # dist[v][7] = ma_Ddistance(te_pred[v], tck7, tSc7)
        # dist[v][8] = ma_Ddistance(te_pred[v], tck8, tSc8)
        # dist[v][9] = ma_Ddistance(te_pred[v], tck9, tSc9)
    te_acc += np.sum(np.argmin(dist.detach().cpu().numpy(), axis=1) == data3[1].cpu().numpy())
    dist_1 = -dist
    dist_2 = hhsoftmax(dist_1, 5)
    mmm = np.argmin(dist.detach().cpu().numpy(), axis=1)  # 预测数据
    nnn = data3[1].cpu().numpy()  # 真实标签
    data_tlist.append(dist_2.detach().cpu().numpy())
    label_tlist.append(nnn.reshape(-1, 1))  # (5, 1)
    aaa = np.append(aaa, mmm)  # 预测
    bbb = np.append(bbb, nnn)  # 真实
    # mmm = np.argmin(dist.detach().cpu().numpy(), axis=1)  # 预测数据
    # nnn = data3[1].cpu().numpy()  # 真实标签
    # aaa = np.append(aaa, mmm)  # 预测
    # bbb = np.append(bbb, nnn)  # 真实
    # te_loss = t_loss.item()
te_acc = 100 * te_acc / te_set.__len__()
data_vstack = np.vstack(data_tlist)  # (950, 10)
label_vstack = np.vstack(label_tlist)  # (950, 1)
label_vstack = transform_1(label_vstack, 5)  # (950, 10)
print("test Loss is :{:.4f}\ttest Acc is :{:.4f}%".format(te_loss, te_acc))

metrics.roc_auc_score(label_vstack, data_vstack, average='macro')  # 调用函数计算AUC
fpr, tpr, thresholds = metrics.roc_curve(label_vstack.ravel(), data_vstack.ravel())
auc = metrics.auc(fpr, tpr)
print('手动计算auc', auc)
plt.plot(fpr, tpr, c='r', lw=2, alpha=0.7, label='AUC=%.5f' % auc)  # c:点的颜色,alpha:透明度,lw:线宽,ls:线的样式
plt.plot((0, 1), (0, 1), c="#808080", lw=1, ls='--', alpha=0.7)
plt.xlim((-0.01, 1.02))
plt.ylim((-0.01, 1.02))
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.grid(b=True, ls=':')  # ls-->linestyle
plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)  # fancybox 图例标记周围采用圆边,framealpha图例框架的透明度,loc图例的位置
plt.show()

# 保存数组
# np.savetxt('C:/Users/asus/Desktop/初稿所用图/SQ gaussian prototype-5.csv', data_vstack, delimiter=',')
# np.savetxt('C:/Users/asus/Desktop/初稿所用图/SQ gaussian prototype label-5.csv', label_vstack, delimiter=',')
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(7):
#     fpr[i], tpr[i], _ = roc_curve(label_vstack[:, i], data_vstack[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
# # Compute micro-average ROC curve and ROC area
# # fpr["micro"], tpr["micro"], _ = roc_curve(label_vstack.ravel(), data_vstack.ravel())
# # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# # Compute macro-average ROC curve and ROC area
# # First aggregate all false positive rates
# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(7)]))
# # Then interpolate all ROC curves at this points
# mean_tpr = np.zeros_like(all_fpr)
# for i in range(7):
#     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# # Finally average it and compute AUC
# mean_tpr /= 7
# fpr["macro"] = all_fpr
# tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#
# lw = 2
# plt.figure()
# # plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc["micro"]),
# #          color='navy', linestyle=':', linewidth=4)
# plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.5f})' ''.format(roc_auc["macro"]),
#          color='deeppink', linestyle=':', linewidth=4)
# # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
# # for i, color in zip(range(10), colors):
# #     plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))
#
# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc="lower right")
# plt.show()


# dataa = model(query_t_x)  # (175,128)
# labell = query_t_y  # (175, 1)
# aaaa = plot_2D(dataa)
# plot_embedding(aaaa, labell, 'T-SNE')


# classes = list(set(bbb))  # 类别
# classes.sort()  # 对比，准确对上分类结果
# confusion = confusion_matrix(aaa, bbb)  # 对比，得到混淆矩阵
# plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Oranges)
# indices = range(len(confusion))
# plt.xticks(indices, classes)
# plt.yticks(indices, classes)
# plt.colorbar()
# plt.title('Prototype Network')
# plt.xlabel('Predict Label')
# plt.ylabel('Real Label')
#
# for first_index in range(len(confusion)):
#     for second_index in range(len(confusion[first_index])):
#         plt.text(first_index, second_index, confusion[first_index][second_index])
#
# plt.savefig("C:/Users/asus/Desktop/prototype.png", dpi=500)
# plt.show()