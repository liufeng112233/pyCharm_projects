"""
    该代码主要是对于模型进行测试，验证模型的稳定性
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import nn, optim
import My_TSNE_sensor6
from Bolt_Train6_MSAE_SC_AA_Net import train_test_Net
from multi_AE6_SC_AA_model import Attention_Fusion_Net
from multi_AE6_SC_model import MySC_AE
from Confusion_Matrix import Count_confusion_matrix, plot_confusion_matrix
from Macro_ROC_AUC import Macro_roc_auc

import Bolt_work3_sensor6_MSDGI_graph_processing
import Bolt_work4_sensor6_MSDGI_graph_processing
import Bolt_work5_sensor6_MSDGI_graph_processing
import Bolt_work6_sensor6_MSDGI_graph_processing
import Bolt_work7_sensor6_MSDGI_graph_processing
import Bolt_work8_sensor6_MSDGI_graph_processing
import Bolt_work9_sensor6_MSDGI_graph_processing

import Bolt4_work4_sensor6_MSDGI_graph_processing
import Bolt2_work4_sensor6_MSDGI_graph_processing
import Bolt1_work4_sensor6_MSDGI_graph_processing
import Bolt_heng_work6_sensor6_MSDGI_graph_processing

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
    Total_Epochs = 150
    learning_rate = 1e-3
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
    fig_name = '松动定位样本32_轴向_六工况'
    # figsave = False   # False:不保存图片，True：保存图片
    figsave = True  # False:不保存图片，True：保存图片
    Task = 1  # 程度
    # Task = 2  # 定位
    # 导入训练、测试数据(原始数据、乱序数据、有序数据)
    train_data0, test_data0, train_label0, test_label0, \
    train_loader_shuffle, test_loader_shuffle, train_loader, test_loader \
        = Bolt_work6_sensor6_MSDGI_graph_processing.data_processing(path6, batch_size, train_sample_number)

    Activate_function = nn.Mish()
    SC_AE_model_path = 'D:/pyCharm_projects/主要代码/六传感全松模型模块作用验证/小样本模型/SCAE_sensor6_train64_work6_epoch150_lr0.001.pt'
    SC_AE_model = MySC_AE().to(devices)  # 法兰管道编码器
    # SC_AE_model = MySC_AE3(sensor_num).to(devices)    # 梁编码器
    SC_AE_model.load_state_dict(torch.load(SC_AE_model_path), strict=False)
    # for p in SC_AE_model.parameters():
    #     p.requires_grad = False

    SC_AE_AA_model_path = 'D:/pyCharm_projects/主要代码/六传感全松模型模块作用验证/小样本模型/分类模型/SCAEAA_train64_work6_epoch300_lr0.001.pt'
    SC_AE_AA_model = Attention_Fusion_Net(SC_AE_model, 1, 8, Activate_function, sensor_num).to(devices)
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
                x_test1 = x_test.view(8, 1, sensor_num, 10240)
                output_test, test_hidden_features, _ = SC_AE_AA_model_classification(x_test1)
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
    SC_AE_AA_model_classification.eval()
    conf_matrix = torch.zeros([num_classes, num_classes])  # 混淆矩阵初始化
    with torch.no_grad():
        for step2, (data2, label2) in enumerate(test_loader):
            x_data2, y_label2 = data2.to(devices), label2.to(devices)
            x_data21 = x_data2.view(8, 1, sensor_num, 10240)
            out_pred, Hidden_Features_test, weight = SC_AE_AA_model_classification(x_data21)
            out_pre_label[step2 * batch_size:(step2 + 1) * batch_size, :] = out_pred.cpu().numpy()
            out_feature[step2 * batch_size:(step2 + 1) * batch_size, :] = Hidden_Features_test.cpu().numpy()
            out_label[step2 * batch_size:(step2 + 1) * batch_size, ] = y_label2.cpu().numpy()
            # 统计混淆矩阵
            pred = F.softmax(out_pred, dim=1)
            pred_classes = torch.argmax(pred, 1)
            conf_matrix = Count_confusion_matrix(pred_classes, y_label2, conf_matrix)

    # My_TSNE_sensor6.My_TSNE(out_feature, out_label, num_classes, fig_name)
    # 绘制混淆矩阵
    conf_matrix = np.array(conf_matrix.cpu())
    plot_confusion_matrix(conf_matrix, norm=True, task=Task, cmap1='Blues', cmap2='Greens', figsave=figsave,
                          fig_name=fig_name)
    # 计算ROC_AUC
    # Micro, Macro, AUC = Macro_roc_auc(out_pre_label, out_label, num_classes_order)
