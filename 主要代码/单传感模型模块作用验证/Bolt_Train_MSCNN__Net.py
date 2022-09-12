import torch
from torch import nn, optim
import numpy as np

devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 8


def train_test_Net(model, learning_rate, total_epochs, num_classes, train_loader, test_loader):
    # 更新学习率
    def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    curr_lr = learning_rate
    loss_fun = nn.CrossEntropyLoss().to(devices)  # 包含了softmax函数的
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 结果保存
    train_epochs_avgloss = []
    test_epochs_avgloss = []
    test_accuracy = []

    for epoch in range(total_epochs):
        model.train()
        train_epoch_loss = []
        for step, (x, label) in enumerate(train_loader):
            x, label = x.to(devices), label.to(devices)
            output_train, _ = model(x)
            # output_train, x1, x2, x3, x4 = model(x)
            # torch.tensor[10*4]
            loss = loss_fun(output_train, label)
            optimizer.zero_grad()
            loss.backward()  # 反向传递
            optimizer.step()

            if step % 5 == 0:
                print('Epoch :', epoch, '|', 'train_loss:%.4f' % loss.data)
            # 保存给个batch的损失数值
            train_epoch_loss.append(loss.item())
        # curr_lr = learning_rate*0.95**epoch
        # update_lr(optimizer, curr_lr)
        # 更新学习率
        if (epoch + 1) % 50 == 0:
            curr_lr /= 2
            update_lr(optimizer, curr_lr)
        # 训练集平均损失
        train_epochs_avgloss.append(np.average(train_epoch_loss))

        # test
        model.eval()
        test_correct = 0
        total_sum2 = 0
        test_epoch_loss = []
        with torch.no_grad():
            for test_idx, (x_test, label_test) in enumerate(train_loader):
                x_test, label_test, = x_test.to(devices), label_test.to(devices)
                output_test, test_hidden_features = model(x_test)
                loss = loss_fun(output_test, label_test)
                # 损失值
                test_epoch_loss.append(loss.item())
                # 准确率
                predicted = output_test.argmax(dim=1)
                test_correct += torch.eq(predicted, label_test).float().sum().item()
                total_sum2 += x_test.size(0)
            print("测试集epoch={}/{},   test_acc={},   te_loss={}".format(
                epoch + 1, total_epochs, 100 * test_correct / total_sum2, np.average(test_epoch_loss)))
            # 平均损失
            test_accuracy.append(100 * test_correct / total_sum2)
            # 验证集平均损失
            test_epochs_avgloss.append(np.average(test_epoch_loss))
    # MSCNN_PATH_name = "D:/pyCharm_projects/主要代码/单传感模型模块作用验证/模型一保存/MSCNN_heng_sensor1_work{}_epoch{}_lr{}.pt". \
    #     format(num_classes, total_epochs, learning_rate)
    # torch.save(model.state_dict(), MSCNN_PATH_name)

    # 输出中间层特增进行可视化
    out_feature = np.zeros((32 * num_classes, 9216))
    out_label = np.zeros((32 * num_classes,))
    model.eval()
    with torch.no_grad():
        for step2, (data2, label2) in enumerate(test_loader):
            x_data2, y_label2 = data2.to(devices), label2.to(devices)
            _, Hidden_Features_test = model(x_data2)
            out_feature[step2 * batch_size:(step2 + 1) * batch_size, :] = Hidden_Features_test.cpu().numpy()
            out_label[step2 * batch_size:(step2 + 1) * batch_size, ] = y_label2.cpu().numpy()

    return np.array(train_epochs_avgloss), np.array(test_epochs_avgloss), \
           np.array(test_accuracy), out_feature, out_label
