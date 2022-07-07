import numpy as np
import torch
import matplotlib.pyplot as plt
from Com2_IDMFFN_model import IDMFFN
from Train_IDMFFN_Net import train_test_Net

import Bolt_work3_sensor6_MSDGI_graph_processing   # 轴向振动，六螺栓松动程度识别2NM
import Bolt_work5_sensor6_MSDGI_graph_processing   # 轴向振动，六螺栓松动程度识别 1NM
import Bolt_work6_sensor6_MSDGI_graph_processing    # 轴向振动，单螺栓分别松动定位 2NM

import Bolt4_work4_sensor6_MSDGI_graph_processing   # 轴向振动，四螺栓松动程度识别
import Bolt2_work4_sensor6_MSDGI_graph_processing   # 轴向振动，双螺栓松动程度识别
import Bolt1_work4_sensor6_MSDGI_graph_processing   # 轴向振动，单螺栓松动程度识别
import Bolt_heng_work6_sensor6_MSDGI_graph_processing   # 横向振动，单螺栓分别松动定位


devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dropout_value = 0.6

if __name__ == '__main__':
    Total_Epochs = 100
    # num_classes = 9
    order_classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path6 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_六螺栓松动'
    path4 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_四螺栓松动四工况'
    path2 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_二螺栓松动四工况'
    path1 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_单螺栓松动四工况'

    path6_heng = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_六螺栓松动_横向'

    sensors_num = 6
    batch_size = 8
    num_classes = 4
    learning_rate = 3e-3
    train_sample_number = 16
    # 导入训练、测试数据(原始数据、乱序数据、有序数据)
    train_data0, test_data0, train_label0, test_label0, \
    train_loader_shuffle, test_loader_shuffle, train_loader, test_loader \
        = Bolt1_work4_sensor6_MSDGI_graph_processing.data_processing(path1, batch_size, train_sample_number)

    MSCNN = IDMFFN(num_classes=num_classes, sensor=sensors_num).to(devices)
    train_epochs_avgloss, test_epochs_avgloss, test_accuracy, hidden_out_features, hidden_out_label = \
        train_test_Net(MSCNN,
                       learning_rate,
                       Total_Epochs,
                       num_classes,
                       sensors_num,
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

    # import numpy as np
    # import pandas as pd

    # a_pd = pd.DataFrame(test_accuracy)
    # writer = pd.ExcelWriter('E:/桌面/训练数据记录/MSCNN_test.xlsx')
    # a_pd.to_excel(writer, 'sheet1', float_format='%.6f')
    # writer.save()
    # writer.close()
