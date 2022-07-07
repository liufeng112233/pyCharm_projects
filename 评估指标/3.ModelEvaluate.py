'''
    1.单幅图片验证
    2.多幅图片验证
'''
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet34
from utils import LoadData, write_result
import pandas as pd
from tqdm import tqdm

def eval(dataloader, model):
    label_list = []
    likelihood_list = []
    pred_list = []
    model.eval()
    with torch.no_grad():
        # 加载数据加载器，得到里面的X（图片数据）和y(真实标签）
        for X, y in tqdm(dataloader, desc="Model is predicting, please wait"):
            # 将数据转到GPU
            X = X.cuda()
            # 将图片传入到模型当中就，得到预测的值pred
            pred = model(X)

            pred_softmax = torch.softmax(pred,1).cpu().numpy()
            # 获取可能性最大的标签
            label = torch.softmax(pred,1).cpu().numpy().argmax()
            label_list.append(label)
            # 获取可能性最大的值（即概率）
            likelihood = torch.softmax(pred,1).cpu().numpy().max()
            likelihood_list.append(likelihood)
            pred_list.append(pred_softmax.tolist()[0])

        return label_list,likelihood_list, pred_list


if __name__ == "__main__":

    '''
        加载预训练模型
    '''
    # 1. 导入模型结构
    model = resnet34(pretrained=False)
    num_ftrs = model.fc.in_features    # 获取全连接层的输入
    model.fc = nn.Linear(num_ftrs, 5)  # 全连接层改为不同的输出
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. 加载模型参数
    # model_loc = "./BEST_resnet_epoch_49_acc_85.2.pth"
    # model_loc = "./BEST_resnet_epoch_1_acc_40.8.pth"
    # model_loc = "./BEST_resnet_epoch_2_acc_57.7.pth"
    model_loc = "./BEST_resnet_epoch_4_acc_70.8.pth"
    # model_loc = "./BEST_resnet_epoch_49_acc_85.2.pth"
    # model_loc = "./BEST_resnet_epoch_1_acc_39.3.pth"

    model_dict = torch.load(model_loc)
    model.load_state_dict(model_dict)
    model = model.to(device)

    '''
       加载需要预测的图片
    '''
    valid_data = LoadData("test.txt", train_flag=False)
    test_dataloader = DataLoader(dataset=valid_data, num_workers=4, pin_memory=True, batch_size=1)


    '''
      获取结果
    '''
    # 获取模型输出
    label_list, likelihood_list, pred =  eval(test_dataloader, model)

    # 将输出保存到exel中，方便后续分析
    label_names = ["daisy", "dandelion","rose","sunflower","tulip"]     # 可以把标签写在这里
    df_pred = pd.DataFrame(data=pred, columns=label_names)



    df_pred.to_csv('pred_result_3.csv', encoding='gbk', index=False)
    print("Done!")

