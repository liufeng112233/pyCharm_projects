import numpy as np
import torch
import matplotlib.pyplot as plt
from multi_AE6_SC_model import MySC_AE
from multi_Beam_AE3_SC_model import MySC_AE as MySC_AE3
import My_TSNE_sensor6
from torch import nn
import torch.nn.functional as F
import time
from Bolt_Train6_MSAE_SC_AA_Net import train_test_Net
from multi_AE6_SC_AA_model import Attention_Fusion_Net

import Bolt_work3_sensor6_MSDGI_graph_processing
import Bolt_work4_sensor6_MSDGI_graph_processing
import Bolt_work5_sensor6_MSDGI_graph_processing   # 轴向振动，六螺栓松动程度识别
import Bolt_work6_sensor6_MSDGI_graph_processing    # 轴向振动，单螺栓分别松动定位
import Bolt_work7_sensor6_MSDGI_graph_processing
import Bolt_work8_sensor6_MSDGI_graph_processing
import Bolt_work9_sensor6_MSDGI_graph_processing

import Beam_work3_sensor6_MSDGI_graph_processing
import Beam_work5_sensor6_MSDGI_graph_processing

import Bolt4_work4_sensor6_MSDGI_graph_processing   # 轴向振动，四螺栓松动程度识别
import Bolt2_work4_sensor6_MSDGI_graph_processing   # 轴向振动，双螺栓松动程度识别
import Bolt1_work4_sensor6_MSDGI_graph_processing   # 轴向振动，单螺栓松动程度识别
import Bolt_heng_work6_sensor6_MSDGI_graph_processing   # 横向振动，单螺栓分别松动定位


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
        [b,c,s,L] = x.size()
        AAF_out1, AAF_out2, AAF_out3, AAF_out4, weight = self.layer_SC_AE_AA(x)  # 8*64*1*106
        x01 = AAF_out1.view(AAF_out1.size()[0], AAF_out1.size()[1], AAF_out1.size()[3])
        x02 = AAF_out2.view(AAF_out2.size()[0], AAF_out2.size()[1], AAF_out2.size()[3])
        x03 = AAF_out3.view(AAF_out3.size()[0], AAF_out3.size()[1], AAF_out3.size()[3])
        x04 = AAF_out4.view(AAF_out4.size()[0], AAF_out4.size()[1], AAF_out4.size()[3])  # 融合模块
        x1 = self.layer_conv1(x03)  # 方法三具有最佳效果
        x2 = self.layer_pool1(x1)
        x3 = self.fc1(x2)
        x3 = F.dropout(x3, p=dropout_value)
        x4 = self.layer_conv2(x3)
        x5 = self.layer_pool2(x4)
        x6 = self.fc2(x5)
        x6 = F.dropout(x6, p=dropout_value)
        x7 = x6.view(b, 256 * 12)
        x8 = self.fc3(self.layer_line1(x7))
        out = self.layer_line2(x8)
        return out, x8, weight


if __name__ == '__main__':
    Total_Epochs = 150
    learning_rate = 1e-4
    # order_classes = ['1', '2', '3']
    # order_classes = ['1', '2', '3', '4']
    # order_classes = ['1', '2', '3', '4', '5']
    # order_classes = ['1', '2', '3', '4', '5','6']
    # order_classes = ['1', '2', '3', '4', '5','6','7']
    # order_classes = ['1', '2', '3', '4', '5', '6', '7', '8']
    order_classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path6 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_六螺栓松动'
    path4 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_四螺栓松动四工况'
    path2 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_二螺栓松动四工况'
    path1 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_单螺栓松动四工况'

    path6_heng = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_法兰_六螺栓松动_横向'

    path_beam_work5 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_横梁_双螺栓松动'
    path_beam_work3 = 'D:/pyCharm_projects/数据集汇总/螺栓振动数据/振动_横梁_单螺栓松动'
    # 导入训练、测试数据(原始数据、乱序数据、有序数据)
    sensor_num = 6    # 传感器的数量
    num_classes = 6
    batch_size = 8
    train_sample_number = 64
    Activate_function = nn.Mish()

    train_data0, test_data0, train_label0, test_label0, \
    train_loader_shuffle, test_loader_shuffle, train_loader, test_loader \
        = Bolt_heng_work6_sensor6_MSDGI_graph_processing.data_processing(path6_heng, batch_size, train_sample_number)

    SC_AE_model_path = 'D:/pyCharm_projects/主要代码/六传感全松模型模块作用验证/激活函数影响模型/SCAE_heng_train64_work6_epoch150_lr0.001.pt'
    SC_AE_model = MySC_AE().to(devices)    # 法兰管道编码器
    # SC_AE_model = MySC_AE3(sensor_num).to(devices)    # 梁编码器
    SC_AE_model.load_state_dict(torch.load(SC_AE_model_path), strict=False)
    # for p in SC_AE_model.parameters():
    #     p.requires_grad = False

    SC_AE_AA_model = Attention_Fusion_Net(SC_AE_model, 1, 8, Activate_function, sensor_num).to(devices)
    print(SC_AE_AA_model)
    # 检查参数是否可以学习
    for name, parms in SC_AE_AA_model.named_parameters():
        print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
              ' -->grad_value:', parms.grad)

    SC_AE_AA_model_classification = MSSCAEAA(SC_AE_AA_model, num_classes, Activate_fun=Activate_function).to(devices)
    start = time.time()  # 训练开始时间

    train_epochs_avgloss, test_epochs_avgloss, test_accuracy, hidden_out_features, hidden_out_label, Att_weight = \
        train_test_Net(SC_AE_AA_model_classification,
                       learning_rate,
                       Total_Epochs,
                       num_classes,
                       batch_size,
                       train_sample_number,
                       sensor_num,
                       train_loader_shuffle,
                       test_loader_shuffle)
    end = time.time()  # 训练结束时间
    Total_time = end - start
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
    # plt.savefig('E:/桌面/训练数据记录/准确损失图.jpg', dpi=3600)
    plt.show()
    # fig_name = '松动程度四_4_32'
    # My_TSNE_sensor6.My_TSNE(hidden_out_features, hidden_out_label, num_classes,fig_name)
    print('模型时间=%.8f' % Total_time)
    #

    # test_train_loss_acc = np.hstack(
    #     [train_epochs_avgloss.reshape(300, 1), test_epochs_avgloss.reshape(300, 1), test_accuracy.reshape(300, 1)])
    # a_pd = pd.DataFrame(test_train_loss_acc)
    # writer = pd.ExcelWriter('E:/桌面/训练数据记录/Bolt6_work9_MSSCAEAA_test.xlsx')
    # a_pd.to_excel(writer, 'loss_acc', float_format='%.6f')
    # writer.save()
    # writer.close()
    threshold_acc = np.average(test_accuracy[140:149])
    order_error=np.zeros((1,140))
    for i in range(140):
        order_error[0, i] = np.average(test_accuracy[i:(i+10)])-threshold_acc
        if np.abs(order_error[0,i])<=0.5:
            print('模型收敛',i)
