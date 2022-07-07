import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from numpy import interp


def Macro_roc_auc(pre_data, label, num_classes):
    '''

    :param pre_data:
    :param label:  array
    :param num_classes: [0,1,2,3,4]
    :return:
    '''
    y = label  # 原始标签一维
    y_test = label_binarize(y, classes=num_classes)  # 转化为二值化
    n_classes = y_test.shape[1]
    y_score = pre_data
    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # 获取两个输出值
    micro_x, micro_y = fpr["micro"], tpr["micro"]
    macro_x, macro_y = fpr["macro"], tpr["macro"]
    micro_auc, macro_auc = roc_auc["micro"], roc_auc["macro"]

    micro = np.hstack([micro_x.reshape(micro_x.shape[0], 1), micro_y.reshape(micro_y.shape[0], 1)])
    macro = np.hstack([macro_x.reshape(macro_x.shape[0], 1), macro_y.reshape(macro_y.shape[0], 1)])
    AUC = np.zeros([2, 1])
    AUC[0, 0], AUC[1, 0] = micro_auc, macro_auc
    return micro, macro_auc, AUC
