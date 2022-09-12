import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

import Bear_Gear_work5_MSDGI_graph_processing as Bear_Gear_work5
import Bear_Gear_work9_MSDGI_graph_processing as Bear_Gear_work9

from multi_AE6_SC_model import MySC_AE

devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 主函数
if __name__ == '__main__':
    # 超参数
    batch_size = 8
    sensors_num = 6
    Total_Epochs = 150
    num_classes = 9
    learning_rate = 1e-3
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

    AE_model = MySC_AE().to(devices)
    print(AE_model)
    optimizer = optim.Adam(AE_model.parameters(), learning_rate)
    Loss_func = nn.MSELoss().to(devices)
    # 平均损失保存
    train_epochs_aveloss = []
    test_epochs_aveloss = []


    # 更新学习率
    def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    curr_lr = learning_rate
    # 训练模型
    for epoch in range(Total_Epochs):
        AE_model.train()
        train_epoch_loss = []
        for step, (data, label) in enumerate(train_loader_shuffle):
            x_data, x_label = data.to(devices), label.to(devices)
            x_data1 = x_data.view(8, 1, sensors_num, 10240)
            encoded_output_train, decoded_output_train = AE_model(x_data1)
            # 损失
            loss = Loss_func(x_data, decoded_output_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 5 == 0:
                print('Epoch :', epoch, '|', 'train_loss:%.8f' % loss.data, '|', 'lr_rate', curr_lr)
            train_epoch_loss.append(loss.item())
        # 更新优化参数的学习率
        # curr_lr = learning_rate * 0.95 ** epoch
        # update_lr(optimizer, curr_lr)
        if (epoch + 1) % 100 == 0:  # each 20 epoch, decay the learning rate
            curr_lr /= 2
            update_lr(optimizer, curr_lr)
        # if (epoch + 1) % 125 == 0:  # each 20 epoch, decay the learning rate
        #     curr_lr /= 2
        #     update_lr(optimizer, curr_lr)
        # 训练集平均损失
        train_epochs_aveloss.append(np.average(train_epoch_loss))

    # 测试模型
    AE_model.eval()
    with torch.no_grad():
        test_epoch_loss = []
        for step1, (data1, label1) in enumerate(test_loader_shuffle):
            input_data1, input_label1 = data1.to(devices), label1.to(devices)
            input_data = input_data1.view(8, 1, sensors_num, 10240)

            encoded_output_test, decoded_output_test = AE_model(input_data)
            # 损失
            loss_test = Loss_func(input_data1, decoded_output_test)
            if step1 % 5 == 0:
                print('test_loss:%.4f' % loss_test.data)
            test_epoch_loss.append(loss_test.item())
        # 训练集平均损失
        test_epochs_aveloss = test_epoch_loss
    # 保存模型
    PATH_name = "D:/pyCharm_projects/主要代码/东南轴承齿轮三传感模型验证/自编码校正模型/分类模型/SCAE_bear20_work{}_epoch{}_lr{}.pt".format(
        num_classes,
        Total_Epochs,
        learning_rate)
    torch.save(AE_model.state_dict(), PATH_name)
    # 加载模型
    # AE_model1 = MyAE().to(devices)
    # AE_model1.load_state_dict(torch.load(PATH_name))
    # AE_model1.eval()

    plt.subplot(2, 1, 1)
    plt.plot(train_epochs_aveloss, label='train_loss', color='r', )
    plt.title("Training loss curve")
    # 训练重构数据
    encode_data_train = x_data.to('cpu').detach().numpy()
    decode_data_train = decoded_output_train.to('cpu').detach().numpy()
    plt.subplot(2, 1, 2)
    plt.plot(encode_data_train[0][0, 0:256], label='train_Original_data', color='b')
    plt.plot(decode_data_train[0][0, 0:256], label='train_Reconstruct_data', color='r')
    plt.title("Data comparison before and after training code")
    plt.legend(loc=2)
    plt.subplots_adjust(hspace=0.5)
    # plt.savefig('E:/桌面/训练数据记录/训练图片记录/MSCAE_train_epoch{}_lr{}.jpg'.format(Total_Epochs, learning_rate), dpi=3600)
    plt.show()

    # 测试损失
    plt.subplot(2, 1, 1)
    plt.plot(test_epochs_aveloss, label='train_loss', color='r', )
    plt.title("Test loss curve")
    # 测试重构数据
    encode_data_test = input_data1.to('cpu').detach().numpy()
    decode_data_test = decoded_output_test.to('cpu').detach().numpy()
    plt.subplot(2, 1, 2)
    plt.plot(encode_data_test[0][0, 0:256], label='test_Original_data', color='b')
    plt.plot(decode_data_test[0][0, 0:256], label='test_Reconstruct_data', color='r')
    plt.title("Data comparison before and after testing code")
    plt.legend(loc=2)
    plt.subplots_adjust(hspace=0.5)
    # plt.savefig('E:/桌面/训练数据记录/训练图片记录/MSCAE_test_epoch{}_lr{}.jpg'.format(Total_Epochs, learning_rate), dpi=3600)
    plt.show()

    # import numpy as np
    # import pandas as pd
    # a_pd = pd.DataFrame(np.array(test_epochs_aveloss))
    # writer = pd.ExcelWriter('E:/桌面/训练数据记录/不同自己校准重构损失.xlsx')
    # a_pd.to_excel(writer, 'sheet1', float_format='%.6f')
    # writer.save()
    # writer.close()
