import numpy as np
import torch
import matplotlib.pyplot as plt
from Com6_MDAAN_model import MDAAN
from Train_MDAAN_Net import train_test_Net
# CWRU data
import Bolt_work5_sensor6_MSDGI_graph_processing   # 轴向振动，六螺栓松动程度识别 1NM
import Bolt_work3_sensor6_MSDGI_graph_processing   # 轴向振动，六螺栓松动程度识别2NM

import Bolt4_work4_sensor6_MSDGI_graph_processing   # 轴向振动，四螺栓松动程度识别
import Bolt2_work4_sensor6_MSDGI_graph_processing   # 轴向振动，双螺栓松动程度识别
import Bolt1_work4_sensor6_MSDGI_graph_processing   # 轴向振动，单螺栓松动程度识别

import Bolt_heng_work6_sensor6_MSDGI_graph_processing   # 横向振动，单螺栓分别松动定位
import Bolt_work6_sensor6_MSDGI_graph_processing    # 轴向振动，单螺栓分别松动定位 2NM

devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dropout_value = 0.6

if __name__ == '__main__':
    Total_Epochs = 100
    # num_classes = 9
    learning_rate = 1e-4
    order_classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # CWRU数据九、十九工况
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path6 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_六螺栓松动'
    path4 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_四螺栓松动四工况'
    path2 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_二螺栓松动四工况'
    path1 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_单螺栓松动四工况'

    path6_heng = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_六螺栓松动_横向'

    sensor_num = 6  # 传感器的数量
    num_classes = 4
    batch_size = 8
    train_sample_number = 16
    # 导入训练、测试数据(原始数据、乱序数据、有序数据)
    train_data0, test_data0, train_label0, test_label0, \
    train_loader_shuffle, test_loader_shuffle, train_loader, test_loader \
        = Bolt1_work4_sensor6_MSDGI_graph_processing.data_processing(path1, batch_size, train_sample_number)

    SWAE_model = MDAAN(num_classes=num_classes, sensor=sensor_num).to(devices)
    train_epochs_avgloss, test_epochs_avgloss, test_accuracy,out_feature, out_label = \
        train_test_Net(SWAE_model,
                       learning_rate,
                       Total_Epochs,
                       num_classes,
                       train_sample_number,
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
