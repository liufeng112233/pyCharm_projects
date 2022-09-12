import numpy as np
import torch
import matplotlib.pyplot as plt
from multi_AE6_SC_model import MySC_AE
import My_TSNE_sensor3
from torch import nn
import torch.nn.functional as F
from Bear_Gear_Train6_MSAE_SC_Net import train_test_Net

import Bear_Gear_work5_MSDGI_graph_processing as Bear_Gear_work5
import Bear_Gear_work9_MSDGI_graph_processing as Bear_Gear_work9

devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dropout_value = 0.4


class MSAE(nn.Module):
    def __init__(self, MyAE, num_classes):
        super(MSAE, self).__init__()
        self.layer_AE = MyAE  # 8*64*1*106
        self.layer_conv1 = nn.Conv2d(64, 128, (1, 3), (1, 3))
        self.layer_pool1 = nn.MaxPool2d((1, 2), (1, 2))
        self.fc1 = nn.ReLU()
        self.layer_conv2 = nn.Conv2d(128, 256, (1, 3), (1, 1))
        self.layer_pool2 = nn.MaxPool2d((1, 2), (1, 2))
        self.fc2 = nn.ReLU()
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
    Total_Epochs = 300
    num_classes = 5
    learning_rate = 1e-4
    # order_classes = ['1', '2', '3']
    # order_classes = ['1', '2', '3', '4']
    # order_classes = ['1', '2', '3', '4', '5']
    # order_classes = ['1', '2', '3', '4', '5', '6']
    # order_classes = ['1', '2', '3', '4', '5','6','7']
    # order_classes = ['1', '2', '3', '4', '5', '6', '7', '8']
    # order_classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 单一工况
    path_bear20 = 'D:/pyCharm_projects/数据集汇总/东南轴承齿轮数据/轴承数据五工况/负载20'
    path_bear30 = 'D:/pyCharm_projects/数据集汇总/东南轴承齿轮数据/轴承数据五工况/负载30'
    path_gear20 = 'D:/pyCharm_projects/数据集汇总/东南轴承齿轮数据/齿轮箱五工况/负载20'
    path_gear30 = 'D:/pyCharm_projects/数据集汇总/东南轴承齿轮数据/齿轮箱五工况/负载30'
    # 混合工况
    path_gear_bear20 = 'D:/pyCharm_projects/数据集汇总/东南轴承齿轮数据/齿轮轴承混杂工况/负载20'
    path_gear_bear30 = 'D:/pyCharm_projects/数据集汇总/东南轴承齿轮数据/齿轮轴承混杂工况/负载30'

    # 导入训练、测试数据(原始数据、乱序数据、有序数据)
    train_data0, test_data0, train_label0, test_label0, \
    train_loader_shuffle, test_loader_shuffle, train_loader, test_loader \
        = Bear_Gear_work5.data_processing(path_bear20, batch_size)

    AE_model_path = 'D:/pyCharm_projects/主要代码/东南轴承齿轮三传感模型验证/自编码校正模型/分类模型/SCAE_bear20_work9_epoch150_lr0.001.pt'
    AE_model = MySC_AE().to(devices)
    AE_model.load_state_dict(torch.load(AE_model_path), strict=False)
    for p in AE_model.parameters():
        p.requires_grad = False

    MSAE_model = MSAE(MyAE=AE_model, num_classes=num_classes).to(devices)
    print(MSAE_model)
    # 检查参数是否可以学习
    for name, parms in MSAE_model.named_parameters():
        print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
              ' -->grad_value:', parms.grad)
    train_epochs_avgloss, test_epochs_avgloss, test_accuracy, hidden_out_features, hidden_out_label = \
        train_test_Net(MSAE_model,
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
    plt.plot(test_accuracy, color='b')
    plt.title("Test_acc")
    plt.xlabel("EPOCH")
    plt.savefig('E:/桌面/训练数据记录/东南轴承数据/准确率损失.png', dpi=3600)
    plt.show()

    My_TSNE_sensor3.myTSNE(hidden_out_features, hidden_out_label, num_classes)

    import numpy as np
    import pandas as pd

    test_train_loss_acc = np.hstack(
        [train_epochs_avgloss.reshape(Total_Epochs, 1), test_epochs_avgloss.reshape(Total_Epochs, 1),
         test_accuracy.reshape(Total_Epochs, 1)])

    a_pd = pd.DataFrame(test_train_loss_acc)
    writer = pd.ExcelWriter('E:/桌面/训练数据记录/东南轴承数据/MSSCAE损失准确记录.xlsx')
    a_pd.to_excel(writer, 'loss_acc', float_format='%.6f')
    writer.save()
    writer.close()
