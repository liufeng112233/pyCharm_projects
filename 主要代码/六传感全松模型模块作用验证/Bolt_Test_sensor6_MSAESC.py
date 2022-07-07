"""
    该代码主要是对于模型进行测试，验证模型的稳定性
"""
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from multi_AE6_SC_model import MySC_AE

import Bolt_work3_sensor6_MSDGI_graph_processing
import Bolt_work4_sensor6_MSDGI_graph_processing
import Bolt_work5_sensor6_MSDGI_graph_processing
import Bolt_work6_sensor6_MSDGI_graph_processing
import Bolt_work7_sensor6_MSDGI_graph_processing
import Bolt_work8_sensor6_MSDGI_graph_processing
import Bolt_work9_sensor6_MSDGI_graph_processing

import Bolt4_work4_sensor6_MSDGI_graph_processing

dropout_value = 0.5

class MSAE(nn.Module):
    def __init__(self, MyAE, num_classes, activate_fun):
        super(MSAE, self).__init__()
        self.layer_AE = MyAE  # 8*64*1*106
        self.layer_conv1 = nn.Conv2d(64, 128, (1, 3), (1, 3))
        self.layer_pool1 = nn.MaxPool2d((1, 2), (1, 2))
        self.fc1 = activate_fun
        self.layer_conv2 = nn.Conv2d(128, 256, (1, 3), (1, 1))
        self.layer_pool2 = nn.MaxPool2d((1, 2), (1, 2))
        self.fc2 = activate_fun
        self.layer_line = nn.Linear(256 * 7, num_classes)

    def forward(self, x):
        encode, decode = self.layer_AE(x)  # 8*128*104
        x1 = self.layer_conv1(encode)
        x2 = self.layer_pool1(x1)
        x3 = self.fc1(x2)
        x3 = F.dropout(x3, p=dropout_value)
        x4 = self.layer_conv2(x3)
        x5 = self.layer_pool2(x4)
        x6 = self.fc2(x5)
        x6 = F.dropout(x6, p=dropout_value)
        x7 = x6.view(8, 256 * 7)
        out = self.layer_line(x7)
        return out, x7


if __name__ == '__main__':
    batch_size = 8
    Total_Epochs = 150
    # num_classes = 4
    learning_rate = 1e-4
    # order_classes = ['1', '2', '3']
    # order_classes = ['1', '2', '3', '4']
    # order_classes = ['1', '2', '3', '4', '5']
    # order_classes = ['1', '2', '3', '4', '5', '6']
    # order_classes = ['1', '2', '3', '4', '5','6','7']
    # order_classes = ['1', '2', '3', '4', '5', '6', '7', '8']
    # order_classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path6 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_六螺栓松动'
    path4 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_四螺栓松动四工况'

    # 导入训练、测试数据(原始数据、乱序数据、有序数据)
    train_data0, test_data0, train_label0, test_label0, \
    train_loader_shuffle, test_loader_shuffle, train_loader, test_loader \
        = Bolt_work9_sensor6_MSDGI_graph_processing.data_processing(path6, batch_size)
    num_classes = 9
    activate_function = nn.Mish()

    AE_model_path = 'D:/pyCharm_projects/主要代码/六传感全松模型模块作用验证/自编码校正模型/Mish_SC2_AE_sensor6_bolt_work9_epoch150_lr0.001.pt'
    AE_model = MySC_AE().to(devices)
    AE_model.load_state_dict(torch.load(AE_model_path), strict=False)
    for p in AE_model.parameters():
        p.requires_grad = False
    MSAE_model = MSAE(MyAE=AE_model, num_classes=num_classes,activate_fun=activate_function).to(devices)
    MSAE_model_path = 'D:/pyCharm_projects/主要代码/六传感全松模型模块作用验证/自编码校正模型/分类模型/SC2_AE_sensor6_bolt6_work9_epoch200_lr0.0001.pt'
    MSAE_model.load_state_dict(torch.load(MSAE_model_path), strict=False)

    loss_fun = nn.CrossEntropyLoss().to(devices)  # 包含了softmax函数的
    optimizer = optim.Adam(MSAE_model.parameters(), lr=learning_rate)

    MSAE_model.eval()
    test_correct = 0
    total_sum2 = 0
    test_epoch_loss = []
    with torch.no_grad():
        for test_idx, (x_test, label_test) in enumerate(test_loader_shuffle):
            x_test, label_test, = x_test.to(devices), label_test.to(devices)
            x_test1 = x_test.view(8, 1, 6, 10240)
            output_test, test_hidden_features = MSAE_model(x_test1)
            loss = loss_fun(output_test, label_test)
            # 损失值
            test_epoch_loss.append(loss.item())
            # 准确率
            predicted = output_test.argmax(dim=1)
            test_correct += torch.eq(predicted, label_test).float().sum().item()
            total_sum2 += x_test.size(0)
        print("test_acc={},   te_loss={}".format(100 * test_correct / total_sum2, np.average(test_epoch_loss)))

