import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
import os


# 混淆矩阵定义
def Count_confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def plot_confusion_matrix(conf_matrix, norm=True, task=1, cmap1='Blues', cmap2='Greens', figsave=False,fig_name='test'):
    """
    :param cmap1:
    :param cmap2:
    :param task:
    :param y_ture: (n,)
    :param y_pred: (n,)
    :param norm: 是否归一化
    :param fig_name: 图片名
    :return:
    """
    font = {'family': 'Times New Roman',
            'weight': 'light',
            'size': 12,
            }
    plt.rc('font', **font)
    # plt.rc('font', family='Times New Roman', style='normal', weight='light', size='1')
    f, ax = plt.subplots()
    font1 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 12,
             }

    if norm:  # # 归一化,可显示准确率,默认显示
        conf_matrix = conf_matrix.astype('float32') / (conf_matrix.sum(axis=1)[:, np.newaxis])
        if task == 1:
            sns.heatmap(conf_matrix, annot=True, ax=ax, cmap=cmap1, fmt='.2f',
                        linewidths=0.02, linecolor="w", vmin=0, vmax=1)
            # ax.set_xlabel('Predicted label ($T_1$)', fontdict=font1)
            # ax.set_ylabel('True label ($T_1$)', fontdict=font1)
            ax.set_xlabel('Predicted label', fontdict=font1)
            ax.set_ylabel('True label', fontdict=font1)
        elif task == 2:
            sns.heatmap(conf_matrix, annot=True, ax=ax, cmap=cmap2, fmt='.2f',
                        linewidths=0.02, linecolor="w", vmin=0, vmax=1)
            # ax.set_xlabel('Predicted label ($T_2$)', fontdict=font1)
            # ax.set_ylabel('True label ($T_2$)', fontdict=font1)
            ax.set_xlabel('Predicted label', fontdict=font1)
            ax.set_ylabel('True label', fontdict=font1)
        # cmap如: cividis, Purples, PuBu, viridis, magma, inferno; fmt: default=>'.2g'
    else:
        if task == 1:
            sns.heatmap(conf_matrix, annot=True, ax=ax, cmap=cmap1, fmt='d')
            ax.set_xlabel('Predicted label ($T_1$)', fontdict=font1)
            ax.set_ylabel('True label ($T_1$)', fontdict=font1)
        elif task == 2:
            sns.heatmap(conf_matrix, annot=True, ax=ax, cmap=cmap2, fmt='d')
            ax.set_xlabel('Predicted label ($T_2$)', fontdict=font1)
            ax.set_ylabel('True label ($T_2$)', fontdict=font1)
        # cmap如: plasma, viridis, magma, inferno, Pastel1_r; fmt: default=>'.2g'
    # ax.set_title('FTI-task', fontdict=font1)  # 标题
    if figsave:    # 判断是否进行图片保存
        root = r'E:/桌面/小论文/图片记录'
        f = os.path.join(root, fig_name)
        plt.savefig(f, dpi=600)
    # print(f'Save at\n{f}')

    plt.show()

# def plot_confusion_matrix(matrix, classes_name, cmap=plt.cm.Blues, normalize=False):
#     '''
#
#     :param matrix:    混淆矩阵
#     :param classes:   list是name = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
#     :param normalize:    图中显示，True:显示百分比，False:显示个数
#     :param title:      图表题
#     :param cmap:      矩阵背景颜色
#     :return:    图
#     '''
#     if normalize:
#         matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#     print(matrix)
#     plt.imshow(matrix, interpolation='nearest', cmap=cmap)
#     plt.title('Confusion_matrix')
#     plt.colorbar()
#     tick_marks = np.arange(len(classes_name))
#     plt.xticks(tick_marks, classes_name, rotation=90)
#     plt.yticks(tick_marks, classes_name)
#
#     # 数理留白问题
#     # x,y轴长度一致(问题1解决办法）
#     plt.axis("equal")
#     # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
#     ax = plt.gca()  # 获得当前axis
#     left, right = plt.xlim()  # 获得x轴最大最小值
#     ax.spines['left'].set_position(('data', left))
#     ax.spines['right'].set_position(('data', right))
#     for edge_i in ['top', 'bottom', 'right', 'left']:
#         ax.spines[edge_i].set_edgecolor("white")
#     # 。。。。。。。。。。。。新增代码结束处。。。。。。。。。。。。。。。。
#
#     thresh = matrix.max() / 2.
#     for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
#         num = '{:.2f}'.format(matrix[i, j]) if normalize else int(matrix[i, j])
#         plt.text(j, i, num,
#                  verticalalignment='center',
#                  horizontalalignment="center",
#                  color="white" if float(num) > thresh else "black")
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()
