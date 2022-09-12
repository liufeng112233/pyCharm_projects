import numpy as np
import torch
import matplotlib.pyplot as plt

import Bolt_work3_sensor1_MSDGI_graph_processing
import Bolt_work4_sensor1_MSDGI_graph_processing
import Bolt_work5_sensor1_MSDGI_graph_processing
import Bolt_work6_sensor1_MSDGI_graph_processing
import Bolt_work7_sensor1_MSDGI_graph_processing
import Bolt_work8_sensor1_MSDGI_graph_processing
import Bolt_work9_sensor1_MSDGI_graph_processing


import Beam_work5_sensor1_MSDGI_graph_processing

from Multi_CNN_model import MSCNN_model
from Bolt_Train_MSCNN__Net import train_test_Net
import My_TSNE_sensor1

if __name__ == '__main__':
    sensors_order = 8
    batch_size = 8
    Total_Epochs = 300
    num_classes = 6
    learning_rate = 1e-3
    # order_classes = ['1', '2', '3']
    # order_classes = ['1', '2', '3', '4']
    # order_classes = ['1', '2', '3', '4', '5']
    # order_classes = ['1', '2', '3', '4', '5','6']
    # order_classes = ['1', '2', '3', '4', '5','6','7']
    # order_classes = ['1', '2', '3', '4', '5','6','7','8']
    order_classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # path = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_六螺栓松动'
    path_heng = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_六螺栓松动_横向'

    path_beam_work5 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_横梁_双螺栓松动'
    path_beam_work3 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_横梁_单螺栓松动'
    # 导入训练、测试数据(原始数据、乱序数据、有序数据)
    train_data0, test_data0, train_label0, test_label0, \
    train_loader_shuffle, test_loader_shuffle, train_loader, test_loader \
        = Bolt_work6_sensor1_MSDGI_graph_processing.data_processing(path_heng, batch_size, sensors_order)

    mscnn_model = MSCNN_model(num_classes=num_classes).to(devices)
    print(mscnn_model)
    train_epochs_avgloss, test_epochs_avgloss, test_accuracy, hidden_out_features, hidden_out_label = \
        train_test_Net(mscnn_model,
                       learning_rate,
                       Total_Epochs,
                       num_classes,
                       train_loader_shuffle,
                       test_loader_shuffle)

    # 可视化
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
    # plt.savefig('E:/桌面/训练数据记录/单传感模型训练/MSCNN_work{}_epoch{}_lr{}.jpg'.format(num_classes, Total_Epochs, learning_rate),
    #             dpi=1200)
    plt.show()

    # My_TSNE_sensor1.myTSNE(hidden_out_features, hidden_out_label, num_classes)
    #
    # import numpy as np
    # import pandas as pd
    #
    # a_pd = pd.DataFrame(test_accuracy)
    # writer = pd.ExcelWriter('E:/桌面/训练数据记录/单螺栓单传感对比数据.xlsx')
    # a_pd.to_excel(writer, 'sheet1', float_format='%.6f')
    # writer.save()
    # writer.close()
