"""
    主要是法兰螺栓振动数据分析，数据级融合形成灰度图
    包括数据的标签和预处理过程
    6螺栓松动包含数据R1-R18；扭矩5/6/7/8/9/10/12/16/20共九种工况
    使用5/6/7/8/9/10/12/20七种工况分析  R1/R2
    设置标签：0,1,2,3,4,5,6
    重叠采样：样本长度10240，重叠量5120，工况每种数据样本
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader

folder_path = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_六螺栓松动'
# folder_path = 'D:/pyCharm_projects/主要代码/数据/振动_法兰_横向六螺栓松动九工况'
data_save_path = 'D:/pyCharm_projects/主要代码/预处理数据保存'
dpath = folder_path
class_number = 8  # 说明数据分为四类


def data_processing(path, batch_size, sensors_order):
    dpath = path

    def beam_load(dpath):
        """
        :param dpath:  数据所在是的路径
        :return: 返回所有数据
        """
        files = os.listdir(dpath)
        R = np.zeros([18, 327680, 16])
        for i in range(18):
            with open(os.path.join(dpath, files[i])) as f:
                read_data = f.read()
                a = read_data.split()
            del a[0:904]  # 删除字符部分，保留数据部分
            data_temp1 = np.array(a)
            R[i][:] = data_temp1.reshape(327680, 16)
        return R

    R1_18 = beam_load(dpath)

    # 4号传感器选择
    def data_select(data, sensor):
        new_data = np.zeros([data.shape[0], data.shape[1]])
        for i in range(18):
            new_data[i, :] = data[i][:, 2 * sensor - 1]  # python数组从0开始

        return new_data

    sensor = sensors_order  # 传感号码
    data = data_select(R1_18, sensor)  # 4号传感9种工况18条数据

    def data_resample(data1, data2, data_len, data_step):
        # Code_data[11488,8],60*2048
        data_temp = np.zeros((126, data_len), dtype=np.float64)
        N = int((len(data1) - data_len) / data_step)
        for i in range(N + 1):
            data_temp[2 * i, :] = data1[i * data_step:i * data_step + data_len]
            data_temp[2 * i + 1, :] = data2[i * data_step:i * data_step + data_len]
        return data_temp

    data_len = 10240
    data_step = 5120

    data_work1 = np.vstack(data_resample(data[0, :], data[1, :], data_len, data_step)).reshape(126, 1, 10240)  # 5NM
    data_work2 = np.vstack(data_resample(data[2, :], data[3, :], data_len, data_step)).reshape(126, 1, 10240)  # 6NM
    data_work3 = np.vstack(data_resample(data[4, :], data[5, :], data_len, data_step)).reshape(126, 1, 10240)  # 7NM
    data_work4 = np.vstack(data_resample(data[6, :], data[7, :], data_len, data_step)).reshape(126, 1, 10240)  # 8NM
    data_work5 = np.vstack(data_resample(data[8, :], data[9, :], data_len, data_step)).reshape(126, 1, 10240)  # 9NM
    data_work6 = np.vstack(data_resample(data[10, :], data[11, :], data_len, data_step)).reshape(126, 1, 10240)  # 10NM
    data_work7 = np.vstack(data_resample(data[12, :], data[13, :], data_len, data_step)).reshape(126, 1, 10240)  # 12NM
    data_work8 = np.vstack(data_resample(data[14, :], data[15, :], data_len, data_step)).reshape(126, 1, 10240)  # 16NM
    data_work9 = np.vstack(data_resample(data[16, :], data[17, :], data_len, data_step)).reshape(126, 1, 10240)  # 20NM

    def processing(data, labels):
        # Code_data=124*10240
        train_data = data[0:96][:, :]  # 96
        test_data = data[94:126][:, :]  # 32

        train_labels = np.array([labels for i in range(0, 96)])
        test_labels = np.array([labels for i in range(0, 32)])
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

    # 数据堆叠
    def process_label(a1, b1, c1, d1, a2, b2, c2, d2):
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
    train_data = train_data8  # (96*num)*1*10240
    train_label = lab_train8
    test_data = test_data8  # (32*num)*1*10240
    test_label = lab_test8

    # 数据归一化，减小差异性
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
