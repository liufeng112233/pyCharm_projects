import numpy as np
import torch
import matplotlib.pyplot as plt
from Com1_CNN_LSTM_model import CNN_DRN_LSTM_model
from Train_CNN_LSTM_Net import train_test_Net
# CWRU data
import Data_CWRU_load3_work9_processing as data_CWRU_work9
import Data_CWRU_load3_DEFE_sensor2_work19_processing as data_CWRU_work19
# SouthEast data
import SE_Bear_Gear_work5_processing as SE_work5
import SE_Bear_Gear_work9_processing as SE_work9
# DIRG data
import DIRG_work7_processing as DIRG_Work7
# Flange bolt data(FB data)
import Bolt6_work3_processing as Bolt6_work3
import Bolt6_work4_processing as Bolt6_work4
import Bolt6_work5_processing as Bolt6_work5
import Bolt6_work6_processing as Bolt6_work6
import Bolt6_work7_processing as Bolt6_work7
import Bolt6_work8_processing as Bolt6_work8
import Bolt6_work9_processing as Bolt6_work9
import Bolt4_work4_processing as Bolt4_work4


devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dropout_value = 0.6

if __name__ == '__main__':
    batch_size = 8
    sensors_num = 6
    Total_Epochs = 100
    # num_classes = 9
    learning_rate = 2e-5
    order_classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # CWRU数据九、十九工况
    path_load3 = 'D:/pyCharm_projects/数据集汇总/西储轴承数据12K/负载三不同故障'  # sensor3
    path_load3_DEFE = 'D:/pyCharm_projects/数据集汇总/西储轴承数据12K/负载三不同故障风扇端驱动端'   # sensor2
    # 东南轴承齿轮数据集
    path_bear20 = 'D:/pyCharm_projects/数据集汇总/东南轴承齿轮数据/轴承数据五工况/负载20'  # 单一工况  # sensor6
    path_bear30 = 'D:/pyCharm_projects/数据集汇总/东南轴承齿轮数据/轴承数据五工况/负载30'
    path_gear20 = 'D:/pyCharm_projects/数据集汇总/东南轴承齿轮数据/齿轮箱五工况/负载20'
    path_gear30 = 'D:/pyCharm_projects/数据集汇总/东南轴承齿轮数据/齿轮箱五工况/负载30'
    path_gear_bear20 = 'D:/pyCharm_projects/数据集汇总/东南轴承齿轮数据/齿轮轴承混杂工况/负载20'  # 混合工况
    path_gear_bear30 = 'D:/pyCharm_projects/数据集汇总/东南轴承齿轮数据/齿轮轴承混杂工况/负载30'
    # # DIRG数据集
    path_DIRG_S100_L000 = 'D:/pyCharm_projects/数据集汇总/DIRG轴承数据/同速100同负载000不同深度'   # sensor6
    path_DIRG_S200_L000 = 'D:/pyCharm_projects/数据集汇总/DIRG轴承数据/同速200同负载000不同深度'
    path_DIRG_S300_L000 = 'D:/pyCharm_projects/数据集汇总/DIRG轴承数据/同速300同负载000不同深度'
    path_DIRG_S400_L000 = 'D:/pyCharm_projects/数据集汇总/DIRG轴承数据/同速400同负载000不同深度'
    path_DIRG_S500_L000 = 'D:/pyCharm_projects/数据集汇总/DIRG轴承数据/同速500同负载000不同深度'
    path_DIRG_S100_L500 = 'D:/pyCharm_projects/数据集汇总/DIRG轴承数据/同速100同负载500不同深度'
    path_DIRG_S200_L500 = 'D:/pyCharm_projects/数据集汇总/DIRG轴承数据/同速200同负载500不同深度'
    path_DIRG_S300_L500 = 'D:/pyCharm_projects/数据集汇总/DIRG轴承数据/同速300同负载500不同深度'
    path_DIRG_S400_L500 = 'D:/pyCharm_projects/数据集汇总/DIRG轴承数据/同速400同负载500不同深度'
    path_DIRG_S500_L500 = 'D:/pyCharm_projects/数据集汇总/DIRG轴承数据/同速500同负载500不同深度'
    # Flange data
    path_bolt6 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_六螺栓松动'
    path_bolt4 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_四螺栓松动四工况'
    # 导入训练、测试数据(原始数据、乱序数据、有序数据)
    train_data0, test_data0, train_label0, test_label0, \
    train_loader_shuffle, test_loader_shuffle, train_loader, test_loader \
        = Bolt6_work7.data_processing(path_bolt6, batch_size)
    num_classes = 7
    MSCNN = CNN_DRN_LSTM_model(num_classes=num_classes, sensor=sensors_num).to(devices)
    train_epochs_avgloss, test_epochs_avgloss, test_accuracy, hidden_out_features, hidden_out_label = \
        train_test_Net(MSCNN,
                       learning_rate,
                       Total_Epochs,
                       num_classes,
                       sensors_num,
                       train_loader_shuffle,
                       test_loader_shuffle)

    ##可视化
    plt.subplot(1, 2, 1)
    plt.plot(train_epochs_avgloss, color='r', label="train_loss")
    plt.plot(test_epochs_avgloss, color='g', label="test_loss")
    plt.title("train_test_loss")
    plt.xlabel("EPOCH")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracy, color='b', label='test_acc')
    plt.title("Test_acc")
    plt.xlabel("EPOCH")
    plt.legend(loc=2)
    # plt.savefig('E:/桌面/训练数据记录/西储轴承数据处理记录/准确率损失.png', dpi=3600)
    plt.show()

    # import numpy as np
    # import pandas as pd

    # a_pd = pd.DataFrame(test_accuracy)
    # writer = pd.ExcelWriter('E:/桌面/训练数据记录/MSCNN_test.xlsx')
    # a_pd.to_excel(writer, 'sheet1', float_format='%.6f')
    # writer.save()
    # writer.close()
