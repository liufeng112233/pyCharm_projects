import torch
from torch import nn, optim
import numpy as np
import torch.nn.functional as F
import math
import My_TSNE_sensor6
from Confusion_Matrix import Count_confusion_matrix, plot_confusion_matrix
from Com1_CNN_LSTM_model import CNN_DRN_LSTM_model
from Com2_IDMFFN_model import IDMFFN
from Com3_MLPC_CNN_model import MPLC_CNN
from Com4_CMSCNN_LSTM_model import CMSCNN_LSTM
from Com5_SWAE_model import SWAE
from Com6_MDAAN_model import MDAAN

import Bolt_work3_sensor6_MSDGI_graph_processing  # 轴向振动，六螺栓松动程度识别2NM
import Bolt_work5_sensor6_MSDGI_graph_processing  # 轴向振动，六螺栓松动程度识别 1NM
import Bolt_work6_sensor6_MSDGI_graph_processing  # 轴向振动，单螺栓分别松动定位 2NM

import Bolt4_work4_sensor6_MSDGI_graph_processing  # 轴向振动，四螺栓松动程度识别
import Bolt2_work4_sensor6_MSDGI_graph_processing  # 轴向振动，双螺栓松动程度识别
import Bolt1_work4_sensor6_MSDGI_graph_processing  # 轴向振动，单螺栓松动程度识别
import Bolt_heng_work6_sensor6_MSDGI_graph_processing  # 横向振动，单螺栓分别松动定位

devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dropout_value = 0.6

devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 8

if __name__ == '__main__':
    Total_Epochs = 100
    learning_rate = 1e-3
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # CWRU数据九、十九工况
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path6 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_六螺栓松动'
    path4 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_四螺栓松动四工况'
    path2 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_二螺栓松动四工况'
    path1 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_单螺栓松动四工况'

    path6_heng = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_六螺栓松动_横向'

    sensor_num = 6  # 传感器的数量
    num_classes = 6
    num_classes_order = [0, 1, 2, 3, 4]
    batch_size = 8
    train_sample_number = 64
    fig_name = 'MDAAN轴向定位32'
    # figsave = False   # False:不保存图片，True：保存图片
    figsave = True  # False:不保存图片，True：保存图片
    Task = 1  # 表示轴向
    # Task = 2  # 用于横向
    # 导入训练、测试数据(原始数据、乱序数据、有序数据)
    train_data0, test_data0, train_label0, test_label0, \
    train_loader_shuffle, test_loader_shuffle, train_loader, test_loader \
        = Bolt_work6_sensor6_MSDGI_graph_processing.data_processing(path6, batch_size, train_sample_number)

    CNN_LSTM_path = 'D:/pyCharm_projects/主要代码/六传感全松模型模块作用验证/对比模型/PCNN_DRN_LSTM/bolt6_location_zhou_train64_work6_epoch100_lr0.0001.pt'
    IDMFFN_path = 'D:/pyCharm_projects/主要代码/六传感全松模型模块作用验证/对比模型/IDMFFN/bolt6_location_zhou_train64_work6_epoch100_lr0.001.pt'
    MLPC_CNN_path = 'D:/pyCharm_projects/主要代码/六传感全松模型模块作用验证/对比模型/MLPC_CNN/bolt_location_zhou_train64_work6_epoch150_lr0.01.pt'
    CMSCNN_LSTM_path = 'D:/pyCharm_projects/主要代码/六传感全松模型模块作用验证/对比模型/CMSCNN_LSTM/bolt6_location_zhou_train64_work6_epoch150_lr0.00015.pt'
    SWAE_path = 'D:/pyCharm_projects/主要代码/六传感全松模型模块作用验证/对比模型/SWAE/bolt6_location_zhou_train64_work6_epoch150_lr0.0001.pt'
    MDAAN_path = 'D:/pyCharm_projects/主要代码/六传感全松模型模块作用验证/对比模型/MDAAN/bolt6_location_zhou_train64_work6_epoch100_lr0.0002.pt'

    Com_model = MDAAN(num_classes=num_classes, sensor=sensor_num).to(devices)
    Com_model.load_state_dict(torch.load(MDAAN_path), strict=False)

    curr_lr = learning_rate
    loss_fun = nn.CrossEntropyLoss().to(devices)  # 包含了softmax函数的
    optimizer = optim.Adam(Com_model.parameters(), lr=learning_rate)
    # 结果保存
    test_epochs_avgloss = []
    test_accuracy = []

    Com_model.eval()
    test_correct = 0
    total_sum2 = 0
    test_epoch_loss = []
    for step in range(10):
        with torch.no_grad():
            for test_idx, (x_test, label_test) in enumerate(test_loader):
                x_test, label_test, = x_test.to(devices), label_test.to(devices)
                x_test1 = x_test.view(8, 1, sensor_num, 10240)
                output_test, test_hidden_features = Com_model(x_test1)
                loss = loss_fun(output_test, label_test)
                # 损失值
                test_epoch_loss.append(loss.item())
                # 准确率
                predicted = output_test.argmax(dim=1)
                test_correct += torch.eq(predicted, label_test).float().sum().item()
                total_sum2 += x_test.size(0)
            print("step={}, test_acc={},   te_loss={}".format(step, 100 * test_correct / total_sum2,
                                                              np.average(test_epoch_loss)))

    N = test_loader.dataset.data.shape[0]
    length = test_hidden_features.shape[1]
    out_pre_label = np.zeros((N, num_classes))
    out_feature = np.zeros((N, length))
    out_label = np.zeros((N,))
    Com_model.eval()
    conf_maxtri = torch.zeros([num_classes, num_classes])  # 混淆矩阵初始化
    with torch.no_grad():
        for step2, (data2, label2) in enumerate(test_loader):
            x_data2, y_label2 = data2.to(devices), label2.to(devices)
            x_data21 = x_data2.view(8, 1, sensor_num, 10240)
            out_pred, Hidden_Features_test = Com_model(x_data21)
            out_pre_label[step2 * batch_size:(step2 + 1) * batch_size, :] = out_pred.cpu().numpy()
            out_feature[step2 * batch_size:(step2 + 1) * batch_size, :] = Hidden_Features_test.cpu().numpy()
            out_label[step2 * batch_size:(step2 + 1) * batch_size, ] = y_label2.cpu().numpy()
            # 统计混淆矩阵
            pred = F.softmax(out_pred, dim=1)
            pred_classes = torch.argmax(pred, 1)
            conf_matrix = Count_confusion_matrix(pred_classes, y_label2, conf_maxtri)

    My_TSNE_sensor6.myTSNE(out_feature, out_label, num_classes)
    # 绘制混淆矩阵
    conf_matrix = np.array(conf_maxtri.cpu())
    plot_confusion_matrix(conf_matrix, norm=True, task=Task, cmap1='Blues', cmap2='Greens', figsave=figsave,
                          fig_name=fig_name)
