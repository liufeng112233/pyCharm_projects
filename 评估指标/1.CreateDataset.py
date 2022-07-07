'''
    生成训练集和测试集，保存在txt文件中
'''
##相当于模型的输入。后面做数据加载器dataload的时候从里面读他的数据
import os
import random#打乱数据用的

def CreateTrainingSet():
    # 百分之80用来当训练集
    train_ratio = 0.8

    rootdata = r"data"  # 数据的根目录

    train_list, test_list = [],[]   # 读取里面每一类的类别
    data_list = []

    #生产train.txt和test.txt
    class_flag = -1
    for a,b,c in os.walk(rootdata):
        print(a)
        for i in range(len(c)):
            data_list.append(os.path.join(a,c[i]))

        for i in range(0, int(len(c)*train_ratio)):
            train_data = os.path.join(a, c[i])+'\t'+str(class_flag)+'\n'
            train_list.append(train_data)

        for i in range(int(len(c) * train_ratio), len(c)):
            test_data = os.path.join(a, c[i]) + '\t' + str(class_flag)+'\n'
            test_list.append(test_data)

        class_flag += 1

    print(train_list)
    random.shuffle(train_list)#打乱次序
    random.shuffle(test_list)

    with open('train.txt','w',encoding='UTF-8') as f:
        for train_img in train_list:
            f.write(str(train_img))

    with open('test.txt','w',encoding='UTF-8') as f:
        for test_img in test_list:
            f.write(test_img)

def CreateEvalData():
    data_list = []
    test_root = r".\testdata"
    for a, b, c in os.walk(test_root):
        for i in range(len(c)):
            data_list.append(os.path.join(a, c[i]))
    print(data_list)
    with open('eval.txt', 'w', encoding='UTF-8') as f:
        for test_img in data_list:
            f.write(test_img + '\t' + "0" + '\n')

if __name__ == "__main__":
    CreateTrainingSet()
    # CreateEvalData()



