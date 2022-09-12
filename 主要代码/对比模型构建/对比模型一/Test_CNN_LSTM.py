"""
    该代码主要是对于模型进行测试，验证模型的稳定性
"""
import numpy as np
import torch
from torch import nn, optim
from Com1_CNN_LSTM_model import CNN_DRN_LSTM_model

import Data_CWRU_load3_work9_processing as data_CWRU_work9
import Data_CWRU_load3_DEFE_sensor2_work19_processing as data_CWRU_work19

import SE_Bear_Gear_work5_processing as SE_work5
import SE_Bear_Gear_work9_processing as SE_work9

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


dropout_value = 0.5



if __name__ == '__main__':
    batch_size = 8
    sensors_num = 6
    Total_Epochs = 150
    # num_classes = 3
    learning_rate = 1e-3
    order_classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # CWRU数据九、十九工况
    path_load3 = 'D:/pyCharm_projects/数据集汇总/西储轴承数据12K/负载三不同故障'
    path_load3_DEFE = 'D:/pyCharm_projects/数据集汇总/西储轴承数据12K/负载三不同故障风扇端驱动端'
    # 东南轴承齿轮数据集
    path_bear20 = 'D:/pyCharm_projects/数据集汇总/东南轴承齿轮数据/轴承数据五工况/负载20'  # 单一工况
    path_bear30 = 'D:/pyCharm_projects/数据集汇总/东南轴承齿轮数据/轴承数据五工况/负载30'
    path_gear20 = 'D:/pyCharm_projects/数据集汇总/东南轴承齿轮数据/齿轮箱五工况/负载20'
    path_gear30 = 'D:/pyCharm_projects/数据集汇总/东南轴承齿轮数据/齿轮箱五工况/负载30'
    path_gear_bear20 = 'D:/pyCharm_projects/数据集汇总/东南轴承齿轮数据/齿轮轴承混杂工况/负载20'  # 混合工况
    path_gear_bear30 = 'D:/pyCharm_projects/数据集汇总/东南轴承齿轮数据/齿轮轴承混杂工况/负载30'
    # # DIRG数据集
    path_DIRG_S100_L000 = 'D:/pyCharm_projects/数据集汇总/DIRG轴承数据/同速100同负载000不同深度'
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
    MSCNN_model_path = 'D:/pyCharm_projects/主要代码/对比模型构建/对比模型一/模型一保存/CNNLSTM_FB6_work7_epoch100_lr2e-05.pt'
    MSCNN = CNN_DRN_LSTM_model(num_classes=num_classes, sensor=sensors_num).to(devices)
    MSCNN.load_state_dict(torch.load(MSCNN_model_path), strict=False)
    # 结果保存
    loss_fun = nn.CrossEntropyLoss().to(devices)  # 包含了softmax函数的
    optimizer = optim.Adam(MSCNN.parameters(), lr=learning_rate)

    MSCNN.eval()
    test_correct = 0
    total_sum2 = 0
    test_epoch_loss = []
    for step in range(10):
        with torch.no_grad():
            for test_idx, (x_test, label_test) in enumerate(test_loader_shuffle):
                x_test, label_test, = x_test.to(devices), label_test.to(devices)
                x_test1 = x_test.view(8, 1, sensors_num, 2048)
                output_test, test_hidden_features = MSCNN(x_test1)
                loss = loss_fun(output_test, label_test)
                # 损失值
                test_epoch_loss.append(loss.item())
                # 准确率
                predicted = output_test.argmax(dim=1)
                test_correct += torch.eq(predicted, label_test).float().sum().item()
                total_sum2 += x_test.size(0)
            print("step={},  test_acc={},   te_loss={}".format(step, 100 * test_correct / total_sum2, np.average(test_epoch_loss)))

