'''
    模型性能度量
'''
from sklearn.metrics import *  # pip install scikit-learn
import matplotlib.pyplot as plt # pip install matplotlib
import numpy as np  # pip install numpy
from numpy import interp
from sklearn.preprocessing import label_binarize
import pandas as pd # pip install pandas

'''
读取数据

需要读取模型输出的标签（predict_label）以及原本的标签（true_label）

'''
target_loc = "test.txt"
target_data = pd.read_csv(target_loc, sep="\t", names=["loc","type"])
true_label = [i for i in target_data["type"]]

# print(true_label)

# predict_loc = "pred_result_1.csv"
# predict_loc = "pred_result_2.csv"
predict_loc = "pred_result_3.csv"
# predict_loc = "pred_result_5.csv"
predict_data = pd.read_csv(predict_loc)#,index_col=0)

# print(predict_data)


# print("predict_data.to_numpy() = ",predict_data.to_numpy())

# # print(np.argmax(predict_data,axis=1))
#
#
predict_label = predict_data.to_numpy().argmax(axis=1)
#
# print("predict_label = ")
# print(predict_label)
# np.argmax(array,axis=1)


predict_score = predict_data.to_numpy().max(axis=1)
print("predict_score = ",predict_score )

# predict_label = [i for i in predict_data["label"]]



'''
    常用指标：精度，查准率，召回率，F1-Score
'''
# 精度，准确率， 预测正确的占所有样本种的比例
accuracy = accuracy_score(true_label, predict_label)
print("精度: ",accuracy)

# 查准率P（准确率），precision(查准率)=TP/(TP+FP)
precision = precision_score(true_label, predict_label, labels=None, pos_label=1, average='micro') # 'micro', 'macro', 'weighted'
print("查准率P: ",precision)

# 查全率R（召回率），原本为对的，预测正确的比例；recall(查全率)=TP/(TP+FN)
recall = recall_score(true_label, predict_label, average='micro') # 'micro', 'macro', 'weighted'
print("召回率: ",recall)

# F1-Score
f1 = f1_score(true_label, predict_label, average='micro')     # 'micro', 'macro', 'weighted'
print("F1 Score: ",f1)


'''
混淆矩阵
'''
label_names = ["daisy", "dandelion","rose","sunflower","tulip"]
confusion = confusion_matrix(true_label, predict_label, labels=[i for i in range(len(label_names))])

# print("混淆矩阵: \n",confusion)

plt.matshow(confusion, cmap=plt.cm.Oranges)   # Greens, Blues, Oranges, Reds
plt.colorbar()
for i in range(len(confusion)):
    for j in range(len(confusion)):
        plt.annotate(confusion[i,j], xy=(i, j), horizontalalignment='center', verticalalignment='center')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.xticks(range(len(label_names)), label_names)
plt.yticks(range(len(label_names)), label_names)
plt.title("Confusion Matrix")
plt.show()


'''
ROC曲线（多分类）
'''
n_classes = len(label_names)
binarize_predict = label_binarize(predict_label, classes=[i for i in range(n_classes)])

# 读取预测结果

predict_score = predict_data.to_numpy()

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(binarize_predict[:,i], [socre_i[i] for socre_i in predict_score])
    roc_auc[i] = auc(fpr[i], tpr[i])
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
# Plot all ROC curves
lw = 2
plt.figure()
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)


for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve of {0} (area = {1:0.2f})'.format(label_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class receiver operating characteristic ')
plt.legend(loc="lower right")
plt.show()