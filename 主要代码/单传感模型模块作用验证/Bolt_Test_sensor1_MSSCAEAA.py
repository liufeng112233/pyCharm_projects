import numpy as np
import torch
import matplotlib.pyplot as plt
from multi_AE1_SC_model import MySC_AE
import My_TSNE_sensor1
from torch import nn, optim
import torch.nn.functional as F
import time
from Bolt_Train1_MSAE_SC_AA_Net import train_test_Net
from multi_AE1_SC_AA_model import Attention_Fusion_Net

import Bolt_work3_sensor1_MSDGI_graph_processing
import Bolt_work4_sensor1_MSDGI_graph_processing
import Bolt_work5_sensor1_MSDGI_graph_processing
import Bolt_work6_sensor1_MSDGI_graph_processing
import Bolt_work7_sensor1_MSDGI_graph_processing
import Bolt_work8_sensor1_MSDGI_graph_processing
import Bolt_work9_sensor1_MSDGI_graph_processing

import Bolt_heng_work6_sensor1_MSDGI_graph_processing

devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dropout_value = 0.6


class MSSCAEAA(nn.Module):
    def __init__(self, SC_AE_AA_model, num_classes, Activate_fun):
        super(MSSCAEAA, self).__init__()
        self.layer_SC_AE_AA = SC_AE_AA_model  # 8*64*1*106
        self.layer_conv1 = nn.Conv1d(64, 128, 3, 2)  # 8*128*52
        self.layer_pool1 = nn.MaxPool1d(2, 2)  # 8*128*26
        self.fc1 = Activate_fun
        self.layer_conv2 = nn.Conv1d(128, 256, 3, 1)
        self.layer_pool2 = nn.MaxPool1d(2, 2)
        self.fc2 = Activate_fun
        self.layer_line1 = nn.Linear(256 * 12, 256 * 6)
        self.fc3 = Activate_fun
        self.layer_line2 = nn.Linear(256 * 6, num_classes)

    def forward(self, x):
        AAF_out1, AAF_out2, AAF_out3, AAF_out4, weight = self.layer_SC_AE_AA(x)  # 8*64*1*106
        x01 = AAF_out1.view(AAF_out1.size()[0], AAF_out1.size()[1], AAF_out1.size()[3])
        x02 = AAF_out2.view(AAF_out2.size()[0], AAF_out2.size()[1], AAF_out2.size()[3])
        x03 = AAF_out3.view(AAF_out3.size()[0], AAF_out3.size()[1], AAF_out3.size()[3])
        x04 = AAF_out4.view(AAF_out4.size()[0], AAF_out4.size()[1], AAF_out4.size()[3])  # 融合模块
        x1 = self.layer_conv1(x03)  # 方法六具有最佳效果
        x2 = self.layer_pool1(x1)
        x3 = self.fc1(x2)
        x3 = F.dropout(x3, p=dropout_value)
        x4 = self.layer_conv2(x3)
        x5 = self.layer_pool2(x4)
        x6 = self.fc2(x5)
        x6 = F.dropout(x6, p=dropout_value)
        x7 = x6.view(8, 256 * 12)
        x8 = self.fc3(self.layer_line1(x7))
        out = self.layer_line2(x8)
        return out, x8, weight


if __name__ == '__main__':
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path6 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_六螺栓松动'  # 2,3,4,5,7,8
    path6_heng = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_六螺栓松动_横向'   # 2,3,4,5,7,8
    Total_Epochs = 150
    learning_rate = 1e-3
    sensor_order = 8
    num_classes = 6
    batch_size = 8
    train_sample_number = 64
    Activate_function = nn.Mish()
    # 导入训练、测试数据(原始数据、乱序数据、有序数据)
    train_data0, test_data0, train_label0, test_label0, \
    train_loader_shuffle, test_loader_shuffle, train_loader, test_loader \
        = Bolt_heng_work6_sensor1_MSDGI_graph_processing.data_processing(path6_heng, batch_size, sensor_order)

    SC_AE_model_path = 'D:/pyCharm_projects/主要代码/单传感模型模块作用验证/预训练模型/SCAE_heng_channel8_work6_epoch150_lr0.001.pt'
    SC_AE_model = MySC_AE().to(devices)
    SC_AE_model.load_state_dict(torch.load(SC_AE_model_path), strict=False)
    for p in SC_AE_model.parameters():
        p.requires_grad = False

    SC_AE_AA_model_path = 'D:/pyCharm_projects/主要代码/单传感模型模块作用验证/预训练模型/分类模型/SCAEAA_zhou_channel2_work6_epoch150_lr0.001.pt'
    SC_AE_AA_model = Attention_Fusion_Net(SC_AE_model, 1, 8, Activate_fun=Activate_function).to(devices)
    SC_AE_AA_model_classification = MSSCAEAA(SC_AE_AA_model, num_classes, Activate_fun=Activate_function).to(devices)

    SC_AE_AA_model_classification.load_state_dict(torch.load(SC_AE_AA_model_path))
    # 打印参数
    # for k in SC_AE_AA_model_classification.parameters():
    #     print(k)

    # test
    curr_lr = learning_rate
    loss_fun = nn.CrossEntropyLoss().to(devices)  # 包含了softmax函数的
    optimizer = optim.Adam(SC_AE_AA_model_classification.parameters(), lr=learning_rate)
    # 结果保存
    test_epochs_avgloss = []
    test_accuracy = []

    SC_AE_AA_model_classification.eval()
    test_correct = 0
    total_sum2 = 0
    test_epoch_loss = []
    for step in range(10):
        with torch.no_grad():
            for test_idx, (x_test, label_test) in enumerate(test_loader):
                x_test, label_test, = x_test.to(devices), label_test.to(devices)
                x_test1 = x_test.view(8, 1, 1, 10240)
                output_test, test_hidden_features, _ = SC_AE_AA_model_classification(x_test1)
                loss = loss_fun(output_test, label_test)
                # 损失值
                test_epoch_loss.append(loss.item())
                # 准确率
                predicted = output_test.argmax(dim=1)
                test_correct += torch.eq(predicted, label_test).float().sum().item()
                total_sum2 += x_test.size(0)
            print("step={}, test_acc={},   te_loss={}".format(step, 100 * test_correct / total_sum2, np.average(test_epoch_loss)))

