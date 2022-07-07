import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from Com1_CNN_LSTM_model import CNN_DRN_LSTM_model
from Train_CNN_LSTM_Net import train_test_Net
from Confusion_Matrix import Count_confusion_matrix, plot_confusion_matrix
# CWRU data
import Bolt_work3_sensor6_MSDGI_graph_processing  # 轴向振动，六螺栓松动程度识别2NM
import Bolt_work5_sensor6_MSDGI_graph_processing  # 轴向振动，六螺栓松动程度识别 1NM
import Bolt_work6_sensor6_MSDGI_graph_processing  # 轴向振动，单螺栓分别松动定位 2NM

import Bolt4_work4_sensor6_MSDGI_graph_processing  # 轴向振动，四螺栓松动程度识别
import Bolt2_work4_sensor6_MSDGI_graph_processing  # 轴向振动，双螺栓松动程度识别
import Bolt1_work4_sensor6_MSDGI_graph_processing  # 轴向振动，单螺栓松动程度识别
import Bolt_heng_work6_sensor6_MSDGI_graph_processing  # 横向振动，单螺栓分别松动定位

devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dropout_value = 0.6

if __name__ == '__main__':
    order_classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # CWRU数据九、十九工况
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path6 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_六螺栓松动'
    path4 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_四螺栓松动四工况'
    path2 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_二螺栓松动四工况'
    path1 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_单螺栓松动四工况'

    path6_heng = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_六螺栓松动_横向'

    Total_Epochs = 100
    learning_rate = 2e-5
    sensor_num = 6  # 传感器的数量
    num_classes = 6
    batch_size = 8
    train_sample_number = 64
    fig_name = '横向振动_单螺栓分别松动定位2NM'
    figsave = False   # False:不保存图片，True：保存图片
    # figsave = True  # False:不保存图片，True：保存图片
    Task = 1   # 蓝色渐变
    # Task = 2  # 绿色渐变
    # 导入训练、测试数据(原始数据、乱序数据、有序数据)
    train_data0, test_data0, train_label0, test_label0, \
    train_loader_shuffle, test_loader_shuffle, train_loader, test_loader \
        = Bolt_heng_work6_sensor6_MSDGI_graph_processing.data_processing(path6_heng, batch_size, train_sample_number)

    MSCNN_model_path = 'D:/pyCharm_projects/主要代码/六传感全松模型模块作用验证/对比模型/CNN_LSTM/bolt6_location_heng_train64_work6_epoch100_lr0.0001.pt'
    MSCNN = CNN_DRN_LSTM_model(num_classes=num_classes, sensor=sensor_num).to(devices)
    MSCNN.load_state_dict(torch.load(MSCNN_model_path), strict=False)
    # 结果保存
    loss_fun = nn.CrossEntropyLoss().to(devices)  # 包含了softmax函数的
    optimizer = optim.Adam(MSCNN.parameters(), lr=learning_rate)

    MSCNN.eval()
    conf_maxtri = torch.zeros([num_classes, num_classes])  # 混淆矩阵初始化
    test_correct = 0
    total_sum2 = 0
    test_epoch_loss = []
    for step in range(10):
        with torch.no_grad():
            for test_idx, (x_test, label_test) in enumerate(test_loader_shuffle):
                x_test, label_test, = x_test.to(devices), label_test.to(devices)
                [B, S, L] = x_test.size()
                x_test1 = x_test.view(B, 1, S, L)
                output_test, test_hidden_features = MSCNN(x_test1)
                loss = loss_fun(output_test, label_test)
                # 损失值
                test_epoch_loss.append(loss.item())
                # 准确率
                predicted = output_test.argmax(dim=1)
                test_correct += torch.eq(predicted, label_test).float().sum().item()
                total_sum2 += x_test.size(0)
                # 统计混淆矩阵
                pred = F.softmax(output_test, dim=1)
                pred_classes = torch.argmax(pred, 1)
                conf_matrix = Count_confusion_matrix(pred_classes, label_test, conf_maxtri)
            print("step={},  test_acc={},   te_loss={}".format(step, 100 * test_correct / total_sum2,
                                                               np.average(test_epoch_loss)))
    conf_matrix = np.array(conf_maxtri.cpu())
    plot_confusion_matrix(conf_matrix, norm=True, task=Task, cmap1='Blues', cmap2='Greens', figsave=figsave,
                          fig_name=fig_name)
