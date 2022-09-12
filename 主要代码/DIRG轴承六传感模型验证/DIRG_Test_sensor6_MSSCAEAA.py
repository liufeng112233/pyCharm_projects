"""
    该代码主要是对于模型进行测试，验证模型的稳定性
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import nn, optim
from DIRG_Train6_MSAE_SC_AA_Net import train_test_Net
from multi_AE6_SC_AA_model import Attention_Fusion_Net
from multi_AE6_SC_model import MySC_AE

import DIRG_work7_MSDGI_graph_processing as DIRG_work7

dropout_value = 0.5


class MSSCAEAA(nn.Module):
    def __init__(self, SC_AE_AA_model, num_classes, activate_F):
        super(MSSCAEAA, self).__init__()
        self.layer_SC_AE_AA = SC_AE_AA_model  # b*64*1*106
        self.layer_conv1 = nn.Conv1d(64, 128, 3, 2)  # 8*128*52
        self.layer_pool1 = nn.MaxPool1d(2, 2)  # 8*128*26
        self.fc1 = activate_F
        self.layer_conv2 = nn.Conv1d(128, 256, 3, 1)
        self.layer_pool2 = nn.MaxPool1d(2, 2)
        self.fc2 = activate_F
        self.layer_line1 = nn.Linear(256 * 12, 256 * 6)
        self.fc3 = activate_F
        self.layer_line2 = nn.Linear(256 * 6, num_classes)

    def forward(self, x):
        [b, C, S, L] = x.size()
        AAF_out1, AAF_out2, AAF_out3, AAF_out4, _ = self.layer_SC_AE_AA(x)  # 8*64*1*106
        x01 = AAF_out1.view(AAF_out1.size()[0], AAF_out1.size()[1], AAF_out1.size()[3])
        x02 = AAF_out2.view(AAF_out2.size()[0], AAF_out2.size()[1], AAF_out2.size()[3])
        x03 = AAF_out3.view(AAF_out3.size()[0], AAF_out3.size()[1], AAF_out3.size()[3])
        x04 = AAF_out4.view(AAF_out4.size()[0], AAF_out4.size()[1], AAF_out4.size()[3])
        x1 = self.layer_conv1(x03)  # 方法六具有最佳效果
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
        return out, x8


if __name__ == '__main__':

    Total_Epochs = 150
    num_classes = 7
    learning_rate = 1e-3
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
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

    batch_size = 2
    train_sample_number = 24
    # 导入训练、测试数据(原始数据、乱序数据、有序数据)
    train_data0, test_data0, train_label0, test_label0, \
    train_loader_shuffle, test_loader_shuffle, train_loader, test_loader \
        = DIRG_work7.data_processing(path_DIRG_S500_L000, batch_size, train_sample_number)

    SC_AE_model_path = 'D:/pyCharm_projects/主要代码/DIRG轴承六传感模型验证/小样本模型/SCAE_DIRG_S500_L000_TrainNUM24_work7_epoch150_lr0.001.pt'
    SC_AE_AA_model_path = 'D:/pyCharm_projects/主要代码/DIRG轴承六传感模型验证/小样本模型/分类模型/SCAEAA_DIRG_S500_L000_24_work7_epoch300_lr0.001.pt'
    activate_function = nn.Mish()
    SC_AE_model = MySC_AE().to(devices)
    SC_AE_model.load_state_dict(torch.load(SC_AE_model_path), strict=False)
    # for p in SC_AE_model.parameters():
    #     p.requires_grad = False

    # SC_AE_AA_model_path = 'D:/pyCharm_projects/主要代码/DIRG轴承六传感模型验证/轴注意力和自校准编码器分类模型/SCAEAA_DIRG_S100_L000_work7_epoch300_lr0.001.pt'
    SC_AE_AA_model = Attention_Fusion_Net(SC_AE_model, 1, 8, activate_function).to(devices)
    SC_AE_AA_model_classification = MSSCAEAA(SC_AE_AA_model, num_classes, activate_function).to(devices)

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
    for step in range(20):
        SC_AE_AA_model_classification.eval()
        test_correct = 0
        total_sum2 = 0
        test_epoch_loss = []
        with torch.no_grad():
            for test_idx, (x_test, label_test) in enumerate(test_loader):
                x_test, label_test, = x_test.to(devices), label_test.to(devices)
                x_test1 = x_test.view(batch_size, 1, 6, 10240)
                output_test, test_hidden_features = SC_AE_AA_model_classification(x_test1)
                loss = loss_fun(output_test, label_test)
                # 损失值
                test_epoch_loss.append(loss.item())
                # 准确率
                predicted = output_test.argmax(dim=1)
                test_correct += torch.eq(predicted, label_test).float().sum().item()
                total_sum2 += x_test.size(0)
            print("step={},  test_acc={},   te_loss={}".format(step, 100 * test_correct / total_sum2,
                                                               np.average(test_epoch_loss)))
