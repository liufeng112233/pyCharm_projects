"""
    复合工况
    东南大学轴承齿轮数据集：
    1 电机振动;2，3，4行星齿轮箱振动：x，y和z;5电机扭矩;6，7，8平行齿轮箱在三个方向上的振动：x，y和z
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import scipy.io as SC
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
import csv
#
folder_path_bear20 = 'D:/pyCharm_projects/数据集汇总/东南轴承齿轮数据/齿轮轴承混杂工况/负载20'
# folder_path_bear30 = 'D:/pyCharm_projects/数据集汇总/东南轴承齿轮数据/齿轮轴承混杂工况/负载30'
dpath = folder_path_bear20


def data_processing(dpath, batch_size):
    dpath = dpath

    def bear_data_load(dpath):
        files = os.listdir(dpath)
        Bear_data = np.zeros([9, 1048560, 6])
        data_length = np.zeros((6, 1))
        for i in range(9):
            data_DE = SC.loadmat(os.path.join(dpath, files[i]))
            name_key = list(data_DE)
            # [Len, _] = data_DE[name_key[3]].shape
            Bear_data[i][:] = np.hstack((data_DE[name_key[4]], data_DE[name_key[5]], data_DE[name_key[6]],
                                         data_DE[name_key[8]], data_DE[name_key[9]], data_DE[name_key[10]]))
            # data_length[i] = Len
        return Bear_data

    Bear_data = bear_data_load(dpath)  # 10*1048560*2:ball\comb\health\inner\outer

    # 重叠采样数据data_len = 10240,steps = 1024
    def data_resample(data, data_length_min, data_len, data_step):
        '''
        :param data:   2个传感器的一种工况数据
        :param data_length_min:    样本数据长度最小值
        :param data_len： 数据划分长度
        :param data_step:    数据步长
        :return:
        '''
        N = int((data_length_min - data_len) / data_step)
        data_temp = np.zeros((N + 1, 6, data_len), dtype=np.float64)  # 一个传感数划分
        for i in range(N + 1):
            data_temp[i][:] = np.transpose(data[i * data_step:i * data_step + data_len])  # 10240*8转置为8*10240
        return data_temp

    # 工况4种，每种118个样本118*2*2048
    data_len = 10240
    data_step = 5120
    data_point_number_min = Bear_data[0].shape[0]
    data_work1 = data_resample(Bear_data[3][:], data_point_number_min, data_len, data_step)
    data_work2 = data_resample(Bear_data[4][:], data_point_number_min, data_len, data_step)
    data_work3 = data_resample(Bear_data[0][:], data_point_number_min, data_len, data_step)
    data_work4 = data_resample(Bear_data[6][:], data_point_number_min, data_len, data_step)
    data_work5 = data_resample(Bear_data[2][:], data_point_number_min, data_len, data_step)
    data_work6 = data_resample(Bear_data[1][:], data_point_number_min, data_len, data_step)
    data_work7 = data_resample(Bear_data[5][:], data_point_number_min, data_len, data_step)
    data_work8 = data_resample(Bear_data[7][:], data_point_number_min, data_len, data_step)
    data_work9 = data_resample(Bear_data[8][:], data_point_number_min, data_len, data_step)

    # data_work10 = data_resample(Bear_data[9][:], data_point_number_min, data_len, data_step)

    def processing(data, labels):
        # Code_data=124*10240
        train_data = data[0:160][:, :]  # 80
        test_data = data[160:200][:, :]  # 36

        train_labels = np.array([labels for i in range(0, 160)])
        test_labels = np.array([labels for i in range(0, 40)])
        return train_data, test_data, train_labels, test_labels

    # 制作数据集

    tr_x1, te_x1, tr_y1, te_y1 = processing(data_work1, 0)
    tr_x2, te_x2, tr_y2, te_y2 = processing(data_work2, 1)
    tr_x3, te_x3, tr_y3, te_y3 = processing(data_work3, 2)
    tr_x4, te_x4, tr_y4, te_y4 = processing(data_work4, 3)
    tr_x5, te_x5, tr_y5, te_y5 = processing(data_work5, 4)
    tr_x6, te_x6, tr_y6, te_y6 = processing(data_work6, 5)
    tr_x7, te_x7, tr_y7, te_y7 = processing(data_work7, 6)
    tr_x8, te_x8, tr_y8, te_y8 = processing(data_work8, 7)
    tr_x9, te_x9, tr_y9, te_y9 = processing(data_work9, 8)

    # tr_x10, te_x10, tr_y10, te_y10 = processing(data_work10, 9)

    # 数据堆叠
    def process_label(a1, b1, c1, d1, a2, b2, c2, d2, ):
        data_tr = np.vstack((a1, a2))  # 训练数据垂直堆叠，一行一个样本
        data_tes = np.vstack((b1, b2))
        lab_tr = np.hstack((c1, c2))
        lab_te = np.hstack((d1, d2))  # 标签水平堆叠
        return data_tr, data_tes, lab_tr, lab_te

    train_data1, test_data1, lab_train1, lab_test1 = process_label(tr_x1, te_x1, tr_y1, te_y1,
                                                                   tr_x2, te_x2, tr_y2, te_y2)
    train_data2, test_data2, lab_train2, lab_test2 = process_label(train_data1, test_data1, lab_train1, lab_test1,
                                                                   tr_x3, te_x3, tr_y3, te_y3)
    train_data3, test_data3, lab_train3, lab_test3 = process_label(train_data2, test_data2, lab_train2, lab_test2,
                                                                   tr_x4, te_x4, tr_y4, te_y4)
    train_data4, test_data4, lab_train4, lab_test4 = process_label(train_data3, test_data3, lab_train3, lab_test3,
                                                                   tr_x5, te_x5, tr_y5, te_y5)
    train_data5, test_data5, lab_train5, lab_test5 = process_label(train_data4, test_data4, lab_train4, lab_test4,
                                                                   tr_x6, te_x6, tr_y6, te_y6)
    train_data6, test_data6, lab_train6, lab_test6 = process_label(train_data5, test_data5, lab_train5, lab_test5,
                                                                   tr_x7, te_x7, tr_y7, te_y7)
    train_data7, test_data7, lab_train7, lab_test7 = process_label(train_data6, test_data6, lab_train6, lab_test6,
                                                                   tr_x8, te_x8, tr_y8, te_y8)
    train_data8, test_data8, lab_train8, lab_test8 = process_label(train_data7, test_data7, lab_train7, lab_test7,
                                                                   tr_x9, te_x9, tr_y9, te_y9)
    # train_data9, test_data9, lab_train9, lab_test9 = process_label(train_data8, test_data8, lab_train8, lab_test8,
    #                                                                tr_x10, te_x10, tr_y10, te_y10)
    train_data = train_data8  # 800*2*2048
    train_label = lab_train8
    test_data = test_data8  # 360*2*2048
    test_label = lab_test8

    # 灰度图数据归一化，减小差异性
    def max_min_Normalization(data):
        [N_1, S_1, L_1] = data.shape
        Max_Min_scaler = preprocessing.MinMaxScaler()
        for i in range(N_1):
            data[i][:] = np.transpose(Max_Min_scaler.fit_transform(np.transpose(data[i][:])))
        return data

    train_data = max_min_Normalization(train_data)
    test_data = max_min_Normalization(test_data)

    train_data = np.array(train_data, dtype=np.float32)
    test_data = np.array(test_data, dtype=np.float32)
    train_label = np.array(train_label, dtype=np.int64)
    test_label = np.array(test_label, dtype=np.int64)

    # 封装数据
    train_data = train_data.reshape(train_data.shape[0], train_data1.shape[1], train_data.shape[2])
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2])

    class Mydata(Dataset):
        def __init__(self, x, y):
            self.data = x
            self.label = y

        # 数据+标签
        def __getitem__(self, index):
            data = torch.from_numpy(self.data)
            label = torch.from_numpy(self.label)
            return data[index], label[index]

        def __len__(self):
            return len(self.label)

    data_train = Mydata(train_data, train_label)
    data_test = Mydata(test_data, test_label)
    # 打乱顺序的数据
    seed = 1927
    torch.manual_seed(seed)
    train_loader_shuffle = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    test_loader_shuffle = DataLoader(data_test, batch_size=batch_size, shuffle=True)
    # 未打乱顺序的数据
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False)

    return train_data, test_data, train_label, test_label, \
           train_loader_shuffle, test_loader_shuffle, train_loader, test_loader
