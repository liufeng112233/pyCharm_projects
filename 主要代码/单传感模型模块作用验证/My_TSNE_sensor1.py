import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def myTSNE(features,labels,num_classes):
    '''

    :param feature:   输入特征：行：一个样本，列表示特征[252,9216]
    :param label:     对应的样本特征[1,252],一行诗句
    :param num_classes:   分类数量
    :return:
    '''
    pca = PCA(n_components=num_classes)  # 总的类别
    pca_result = pca.fit_transform(features)
    print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))

    test_samples = features.shape[0]
    X_embedded = TSNE(n_components=2, verbose=1).fit_transform(pca_result)  # testing_feature是提取出来的特征
    testing_label = labels.reshape(-1, 1)
    X_embeddedd = np.hstack((X_embedded, testing_label))  # testing_label是分类器输出的标签
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(15, 15))
    type0_x = []
    type0_y = []

    type1_x = []
    type1_y = []

    type2_x = []
    type2_y = []

    type3_x = []
    type3_y = []
    #
    type4_x = []
    type4_y = []

    type5_x = []
    type5_y = []

    type6_x = []
    type6_y = []

    type7_x = []
    type7_y = []

    type8_x = []
    type8_y = []
    for k in range(test_samples):  # test_samples是测试样本数
        if X_embeddedd[k, 2] == 0:
            type0_x.append(X_embeddedd[k, 0])
            type0_y.append(X_embeddedd[k, 1])
        elif X_embeddedd[k, 2] == 1:
            type1_x.append(X_embeddedd[k, 0])
            type1_y.append(X_embeddedd[k, 1])
        elif X_embeddedd[k, 2] == 2:
            type2_x.append(X_embeddedd[k, 0])
            type2_y.append(X_embeddedd[k, 1])
        elif X_embeddedd[k, 2] == 3:
            type3_x.append(X_embeddedd[k, 0])
            type3_y.append(X_embeddedd[k, 1])
        elif X_embeddedd[k, 2] == 4:
            type4_x.append(X_embeddedd[k, 0])
            type4_y.append(X_embeddedd[k, 1])
        elif X_embeddedd[k, 2] == 5:
            type5_x.append(X_embeddedd[k, 0])
            type5_y.append(X_embeddedd[k, 1])
        elif X_embeddedd[k, 2] == 6:
            type6_x.append(X_embeddedd[k, 0])
            type6_y.append(X_embeddedd[k, 1])
        elif X_embeddedd[k, 2] == 7:
            type7_x.append(X_embeddedd[k, 0])
            type7_y.append(X_embeddedd[k, 1])
        elif X_embeddedd[k, 2] == 8:
            type8_x.append(X_embeddedd[k, 0])
            type8_y.append(X_embeddedd[k, 1])
    type0_x = np.array(type0_x)
    type0_y = np.array(type0_y)

    type1_x = np.array(type1_x)
    type1_y = np.array(type1_y)

    type2_x = np.array(type2_x)
    type2_y = np.array(type2_y)
    #
    type3_x = np.array(type3_x)
    type3_y = np.array(type3_y)
    #
    type4_x = np.array(type4_x)
    type4_y = np.array(type4_y)

    type5_x = np.array(type5_x)
    type5_y = np.array(type5_y)

    type6_x = np.array(type6_x)
    type6_y = np.array(type6_y)

    type7_x = np.array(type7_x)
    type7_y = np.array(type7_y)

    type8_x = np.array(type8_x)
    type8_y = np.array(type8_y)

    s_size = 100
    al = 1
    plt.scatter(type0_x, type0_y, s=s_size, cmap=0, marker='o', label='1', edgecolors=(0, 0, 0), alpha=al)
    plt.scatter(type1_x, type1_y, s=s_size, cmap=1, marker='s', label='2', edgecolors=(0, 0, 0), alpha=al)
    plt.scatter(type2_x, type2_y, s=s_size, cmap=2, marker='<', label='3', edgecolors=(0, 0, 0), alpha=al)
    plt.scatter(type3_x, type3_y, s=s_size, cmap=3, marker='o', label='4', edgecolors=(0, 0, 0), alpha=al)
    plt.scatter(type4_x, type4_y, s=s_size, cmap=4, marker='s', label='5', edgecolors=(0, 0, 0), alpha=al)
    plt.scatter(type5_x, type5_y, s=s_size, cmap=5, marker='<', label='6', edgecolors=(0, 0, 0), alpha=al)
    plt.scatter(type6_x, type6_y, s=s_size, cmap=6, marker='o', label='7', edgecolors=(0, 0, 0), alpha=al)
    plt.scatter(type7_x, type7_y, s=s_size, cmap=7, marker='s', label='8', edgecolors=(0, 0, 0), alpha=al)
    plt.scatter(type8_x, type8_y, s=s_size, cmap=8, marker='<', label='9', edgecolors=(0, 0, 0), alpha=al)

    #plt.scatter(type10_x, type10_y, s=s_size, c=(0.8, 0.6, 1), marker='s', label='SC-2', edgecolors=(0, 0, 0), alpha=al)
    #plt.scatter(type11_x, type11_y, s=s_size, c=(0.8, 0.6, 1), marker='<', label='SC-3', edgecolors=(0, 0, 0), alpha=al)'''
    plt.legend(loc='upper left')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel("xlable", fontsize=27)
    plt.ylabel("ylable", fontsize=27)
    plt.rcParams.update({'font.size': 22})
    plt.xlabel('Dimension-1')
    plt.ylabel('Dimension-2')
    plt.savefig('E:/桌面/训练数据记录/单传感模型训练/MSCNN_work{}.jpg'.format(num_classes), dpi=1200)
    plt.show()
