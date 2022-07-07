"""
    该代码主要是对于模型进行测试，验证模型的稳定性
"""
import numpy as np
import torch
from torch import nn, optim
from multi_MSCNN6_model import MSCNN_model

import Bolt_work3_sensor6_MSDGI_graph_processing
import Bolt_work4_sensor6_MSDGI_graph_processing
import Bolt_work5_sensor6_MSDGI_graph_processing
import Bolt_work6_sensor6_MSDGI_graph_processing
import Bolt_work7_sensor6_MSDGI_graph_processing
import Bolt_work8_sensor6_MSDGI_graph_processing
import Bolt_work9_sensor6_MSDGI_graph_processing

import Bolt4_work4_sensor6_MSDGI_graph_processing


dropout_value = 0.5

if __name__ == '__main__':
    batch_size = 8
    Total_Epochs = 150
    num_classes = 4
    learning_rate = 1e-3
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path6 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_六螺栓松动'
    path4 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_四螺栓松动四工况'

    # 导入训练、测试数据(原始数据、乱序数据、有序数据)
    train_data0, test_data0, train_label0, test_label0, \
    train_loader_shuffle, test_loader_shuffle, train_loader, test_loader \
        = Bolt_work4_sensor6_MSDGI_graph_processing.data_processing(path6, batch_size)

    MSCNN_model_path = 'D:/pyCharm_projects/主要代码/六传感全松模型模块作用验证/多尺度卷积模型/MSCNN_Bolt2_sensor6_work4_epoch300_lr0.0001.pt'
    MSCNN = MSCNN_model(num_classes=num_classes).to(devices)
    MSCNN.load_state_dict(torch.load(MSCNN_model_path), strict=False)
    # 结果保存
    loss_fun = nn.CrossEntropyLoss().to(devices)  # 包含了softmax函数的
    optimizer = optim.Adam(MSCNN.parameters(), lr=learning_rate)

    MSCNN.eval()
    test_correct = 0
    total_sum2 = 0
    test_epoch_loss = []
    with torch.no_grad():
        for test_idx, (x_test, label_test) in enumerate(test_loader_shuffle):
            x_test, label_test, = x_test.to(devices), label_test.to(devices)
            x_test1 = x_test.view(8, 1, 6, 10240)
            output_test, test_hidden_features = MSCNN(x_test1)
            loss = loss_fun(output_test, label_test)
            # 损失值
            test_epoch_loss.append(loss.item())
            # 准确率
            predicted = output_test.argmax(dim=1)
            test_correct += torch.eq(predicted, label_test).float().sum().item()
            total_sum2 += x_test.size(0)
        print("test_acc={},   te_loss={}".format(100 * test_correct / total_sum2, np.average(test_epoch_loss)))

