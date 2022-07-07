import numpy as np
import torch
import matplotlib.pyplot as plt
from multi_MSCNN6_model import MSCNN_model
import My_TSNE_sensor6
from torch import nn
import torch.nn.functional as F
from Bolt_Train_MSCNN_Net import train_test_Net

import Bolt_work3_sensor6_MSDGI_graph_processing
import Bolt_work4_sensor6_MSDGI_graph_processing
import Bolt_work5_sensor6_MSDGI_graph_processing
import Bolt_work6_sensor6_MSDGI_graph_processing
import Bolt_work7_sensor6_MSDGI_graph_processing
import Bolt_work8_sensor6_MSDGI_graph_processing
import Bolt_work9_sensor6_MSDGI_graph_processing

import Bolt4_work4_sensor6_MSDGI_graph_processing

devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dropout_value = 0.4

if __name__ == '__main__':
    batch_size = 8
    Total_Epochs = 300
    num_classes = 9
    learning_rate = 1e-4
    # order_classes = ['1', '2', '3']
    # order_classes = ['1', '2', '3', '4']
    # order_classes = ['1', '2', '3', '4', '5']
    # order_classes = ['1', '2', '3', '4', '5', '6']
    # order_classes = ['1', '2', '3', '4', '5', '6', '7']
    # order_classes = ['1', '2', '3', '4', '5', '6', '7', '8']
    order_classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path6 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_六螺栓松动'
    path4 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_四螺栓松动四工况'

    # 导入训练、测试数据(原始数据、乱序数据、有序数据)
    train_data0, test_data0, train_label0, test_label0, \
    train_loader_shuffle, test_loader_shuffle, train_loader, test_loader \
        = Bolt_work9_sensor6_MSDGI_graph_processing.data_processing(path6, batch_size)

    MSCNN = MSCNN_model(num_classes=num_classes).to(devices)
    train_epochs_avgloss, test_epochs_avgloss, test_accuracy, hidden_out_features, hidden_out_label = \
        train_test_Net(MSCNN,
                       learning_rate,
                       Total_Epochs,
                       num_classes,
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
    # plt.savefig('E:/桌面/训练数据记录/四螺栓四工况/MSCNN_work{}_epoch{}_lr{}.jpg'.format(num_classes, Total_Epochs, learning_rate),
    #             dpi=1200)
    plt.show()

    # My_TSNE_sensor6.myTSNE(hidden_out_features, hidden_out_label, num_classes)

    # import numpy as np
    # import pandas as pd

    # a_pd = pd.DataFrame(test_accuracy)
    # writer = pd.ExcelWriter('E:/桌面/训练数据记录/MSCNN_test.xlsx')
    # a_pd.to_excel(writer, 'sheet1', float_format='%.6f')
    # writer.save()
    # writer.close()
