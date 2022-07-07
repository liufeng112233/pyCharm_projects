"""
    主要是法兰螺栓振动数据分析，数据级融合形成灰度图
    包括数据的标签和预处理过程
    6螺栓松动包含数据R1-R18；扭矩5/6/7/8/9/10/12/16/20共九种工况
    使用5/7/9/12/20五种工况分析  R1/R2
    设置标签：0,1,2,3,
    重叠采样：样本长度10240，重叠量5120，工况每种数据样本
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader

folder_path = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_六螺栓松动'
data_save_path = 'D:/pyCharm_projects/主要代码/预处理数据保存'
dpath = folder_path
class_number = 5  # 说明数据分为四类


def data_processing(path, batch_size, train_sample):
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

    # 获取四类数据8传感
    R1_18 = beam_load(dpath)
    data_CM = np.zeros([10, 327680, 16])
    data_CM[0][:] = R1_18[0][:]  # 扭矩5NM
    data_CM[1][:] = R1_18[1][:]
    data_CM[2][:] = R1_18[4][:]  # 7NM
    data_CM[3][:] = R1_18[5][:]
    data_CM[4][:] = R1_18[8][:]  # 9NM
    data_CM[5][:] = R1_18[9][:]
    data_CM[6][:] = R1_18[12][:]  # 12NM
    data_CM[7][:] = R1_18[13][:]
    data_CM[8][:] = R1_18[16][:]  # 20NM
    data_CM[9][:] = R1_18[17][:]

    # 去除时间序列数据一号传感数据干扰量太大，所以剔除1、6号传感
    data = np.delete(data_CM, [0, 1, 2, 4, 6, 8, 10, 11, 12, 14], 2)  # 8*327680*6

    # 归一化处理(效果不佳)
    # def Normal_data(x):
    #     '''
    #     :param x: 输入数据18*327680
    #     :return:  max_min归一化数据
    #     '''
    #     temp_data = np.zeros((x.shape[0],x.shape[1]))
    #     for i in range(x.shape[0]):
    #         max1 = max(x[i,:])
    #         min1 = min(x[i,:])
    #         for j in range(x.shape[1]):
    #             temp_data[i,j] = (x[i,j]-min1)/(max1-min1)
    #     return temp_data
    #
    # data = Normal_data(data)

    # 重叠采样数据data_len = 10240,steps = 5120
    def data_resample(data1, data2, data_len, data_step):
        '''

        :param data:   8个传感器的一种工况数据327680*8
        :param data_len:    样本数据长度
        :param data_step:    数据步长
        :return:
        '''
        data_temp = np.zeros((126, 6, data_len), dtype=np.float64)  # 一个传感数划分
        N = int((len(data1) - data_len) / data_step)
        for i in range(N + 1):
            data_temp[2 * i][:] = np.transpose(data1[i * data_step:i * data_step + data_len])  # 10240*8转置为8*10240
            data_temp[2 * i + 1][:] = np.transpose(data2[i * data_step:i * data_step + data_len])  # 10240*8转置为8*10240
        return data_temp

    # 工况4种，每种126个样本126*8*10240
    data_len = 10240
    data_step = 5120

    data_work1 = data_resample(data[0][:], data[1][:], data_len, data_step)
    data_work2 = data_resample(data[2][:], data[3][:], data_len, data_step)
    data_work3 = data_resample(data[4][:], data[5][:], data_len, data_step)
    data_work4 = data_resample(data[6][:], data[7][:], data_len, data_step)
    data_work5 = data_resample(data[8][:], data[9][:], data_len, data_step)

    def processing(data, labels, sample):
        # Code_data=124*10240
        train_data = data[0:sample][:, :]  # 96
        test_data = data[70:126][:, :]  # 56

        train_labels = np.array([labels for i in range(0, sample)])
        test_labels = np.array([labels for i in range(0, 56)])
        return train_data, test_data, train_labels, test_labels

    # 制作数据集

    tr_x1, te_x1, tr_y1, te_y1 = processing(data_work1, 0, train_sample)
    tr_x2, te_x2, tr_y2, te_y2 = processing(data_work2, 1, train_sample)
    tr_x3, te_x3, tr_y3, te_y3 = processing(data_work3, 2, train_sample)
    tr_x4, te_x4, tr_y4, te_y4 = processing(data_work4, 3, train_sample)
    tr_x5, te_x5, tr_y5, te_y5 = processing(data_work5, 4, train_sample)

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
    train_data = train_data4  # 384*8*10240
    train_label = lab_train4
    test_data = test_data4  # 112*8*10240
    test_label = lab_test4

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
