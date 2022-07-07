'''
    纪录训练信息，包括：
    1. train loss
    2. test loss
    3. test accuracy
'''
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet34
from utils import LoadData, write_result

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    avg_total = 0.0
    # 从数据加载器中读取batch（一次读取多少张，即批次数），X(图片数据)，y（图片真实标签）。
    for batch, (X, y) in enumerate(dataloader):
        # 将数据存到显卡
        X, y = X.cuda(), y.cuda()
        # 得到预测的结果pred
        pred = model(X)
        # 计算预测的误差
        loss = loss_fn(pred, y)
        avg_total = avg_total+loss.item()

        # 反向传播，更新模型参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每训练10次，输出一次当前信息
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>5f}  [{current:>5d}/{size:>5d}]")

    avg_loss = f"{(avg_total % batch_size):>5f}"
    return avg_loss

def test(dataloader, model):
    size = len(dataloader.dataset)
    # 将模型转为验证模式
    model.eval()
    # 初始化test_loss 和 correct， 用来统计每次的误差
    test_loss, correct = 0, 0
    # 测试时模型参数不用更新，所以no_gard()
    # 非训练， 推理期用到
    with torch.no_grad():
        # 加载数据加载器，得到里面的X（图片数据）和y(真实标签）
        for X, y in dataloader:
            # 将数据转到GPU
            X, y = X.cuda(), y.cuda()
            # 将图片传入到模型当中就，得到预测的值pred
            pred = model(X)
            # 计算预测值pred和真实值y的差距
            test_loss += loss_fn(pred, y).item()
            # 统计预测正确的个数
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    accuracy = f"{(100*correct):>0.1f}"
    avg_loss = f"{test_loss:>8f}"
    print(f"correct = {correct}, Test Error: \n Accuracy: {accuracy}%, Avg loss: {avg_loss} \n")
    # 增加数据写入功能
    return accuracy, avg_loss

if __name__ == '__main__':
    batch_size = 32

    # # 给训练集和测试集分别创建一个数据集加载器
    train_data = LoadData("train.txt", True)
    valid_data = LoadData("test.txt", False)

    train_dataloader = DataLoader(dataset=train_data, num_workers=4, pin_memory=True, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=valid_data, num_workers=4, pin_memory=True, batch_size=batch_size)

    # 如果显卡可用，则用显卡进行训练
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    '''
            修改ResNet34模型的最后一层
    '''
    pretrain_model = resnet34(pretrained=False)
    num_ftrs = pretrain_model.fc.in_features    # 获取全连接层的输入
    pretrain_model.fc = nn.Linear(num_ftrs, 5)  # 全连接层改为不同的输出

    # 预先训练好的参数， 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
    pretrained_dict = torch.load('./resnet34_pretrain.pth')

    # # 弹出fc层的参数
    pretrained_dict.pop('fc.weight')
    pretrained_dict.pop('fc.bias')

    # # 自己的模型参数变量，在开始时里面参数处于初始状态，所以很多0和1
    model_dict = pretrain_model.state_dict()

    # # 去除一些不需要的参数
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # # 模型参数列表进行参数更新，加载参数
    model_dict.update(pretrained_dict)

    # 改进过的预训练模型结构，加载刚刚的模型参数列表
    pretrain_model.load_state_dict(model_dict)

    '''
        冻结部分层
    '''
    # 将满足条件的参数的 requires_grad 属性设置为False
    for name, value in pretrain_model.named_parameters():
        if (name != 'fc.weight') and (name != 'fc.bias'):
            value.requires_grad = False
    #
    # filter 函数将模型中属性 requires_grad = True 的参数选出来
    params_conv = filter(lambda p: p.requires_grad, pretrain_model.parameters())    # 要更新的参数在parms_conv当中

    model = pretrain_model.to(device)

    # 定义损失函数，计算相差多少，交叉熵，
    loss_fn = nn.CrossEntropyLoss()

    '''   控制优化器只更新需要更新的层  '''
    optimizer = torch.optim.SGD(params_conv, lr=1e-3)  # 初始学习率
    #
    # 一共训练5次
    epochs = 50
    best = 0.0
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss = train(train_dataloader, model, loss_fn, optimizer)
        accuracy, avg_loss = test(test_dataloader, model)
        write_result("model_traindata.txt", t+1, train_loss, avg_loss, accuracy)

        if (t+1) % 10 == 0:
            torch.save(model.state_dict(), "resnet_epoch_"+str(t+1)+"_acc_"+str(accuracy)+".pth")

        if float(accuracy) > best:
            best = float(accuracy)
            torch.save(model.state_dict(), "BEST_resnet_epoch_" + str(t+1) + "_acc_" + str(accuracy) + ".pth")

    print("Train PyTorch Model Success!")