import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def My_TSNE(input_data, input_label, classes, fig_name=None, labels=None, n_dim=2):
    input_label = input_label.astype(dtype=int)
    da = TSNE(n_components=n_dim, init='pca', random_state=0, angle=0.3).fit_transform(input_data)
    da = MinMaxScaler().fit_transform(da)  # (n, n_dim)

    figs = plt.figure()
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 10}
    mark = ['o', 'v', 's', 'p', '*', 'h', '8', '.', '4', '^', '+', 'x', '1', '2']
    #  实心圆，正三角，正方形，五角，星星，六角，八角，点，tri_right, 倒三角...

    if labels is None:
        if classes == 4:
            labels = ['8Nm', '9Nm', '10Nm', '20Nm']  # three types
        elif classes == 5:
            labels = ['7Nm', '8Nm', '9Nm', '10Nm', '20Nm']  # four types
        elif classes == 6:
            labels = [' ', ' ', ' ', ' ', ' ', ' ']  # four types
    assert len(labels) == classes

    plt.rc('font', family='Times New Roman', style='normal', weight='light', size=10)
    ax = figs.add_subplot(111)
    # "husl", "muted"
    palette = np.array(sns.color_palette(palette="husl", n_colors=classes))  # [classes, 3]
    # print(palette.shape)
    sample_numbers = len(input_label)
    if classes == 4:
        type0 = np.empty((0, n_dim), dtype=np.float32)
        type1 = np.empty((0, n_dim), dtype=np.float32)
        type2 = np.empty((0, n_dim), dtype=np.float32)
        type3 = np.empty((0, n_dim), dtype=np.float32)
        for i in range(sample_numbers):
            if input_label[i] == 0:
                type0 = np.vstack((type0, da[i]))
            elif input_label[i] == 1:
                type1 = np.vstack((type1, da[i]))
            elif input_label[i] == 2:
                type2 = np.vstack((type2, da[i]))
            elif input_label[i] == 3:
                type3 = np.vstack((type3, da[i]))
        ax.scatter(type0[:, 0], type0[:, 1], s=100, color=palette[0], alpha=0.8, marker=mark[0], label=labels[0])
        ax.scatter(type1[:, 0], type1[:, 1], s=100, color=palette[1], alpha=0.8, marker=mark[1], label=labels[1])
        ax.scatter(type2[:, 0], type2[:, 1], s=100, color=palette[2], alpha=0.8, marker=mark[2], label=labels[2])
        ax.scatter(type3[:, 0], type3[:, 1], s=100, color=palette[3], alpha=0.8, marker=mark[3], label=labels[3])
        ax.set_xlim(-0.1, 1.3)
        ax.set_ylim(-0.1, 1.3)
        ax.legend(loc='upper right', prop=font, labelspacing=1)
    elif classes == 5:
        type0 = np.empty((0, n_dim), dtype=np.float32)
        type1 = np.empty((0, n_dim), dtype=np.float32)
        type2 = np.empty((0, n_dim), dtype=np.float32)
        type3 = np.empty((0, n_dim), dtype=np.float32)
        type4 = np.empty((0, n_dim), dtype=np.float32)
        for i in range(sample_numbers):
            if input_label[i] == 0:
                type0 = np.vstack((type0, da[i]))
            elif input_label[i] == 1:
                type1 = np.vstack((type1, da[i]))
            elif input_label[i] == 2:
                type2 = np.vstack((type2, da[i]))
            elif input_label[i] == 3:
                type3 = np.vstack((type3, da[i]))
            elif input_label[i] == 4:
                type4 = np.vstack((type4, da[i]))
        ax.scatter(type0[:, 0], type0[:, 1], s=100, color='#FFB90F', alpha=0.8, marker=mark[0], label=labels[0])
        ax.scatter(type1[:, 0], type1[:, 1], s=100, color='#FF83FA', alpha=0.8, marker=mark[1], label=labels[1])
        ax.scatter(type2[:, 0], type2[:, 1], s=100, color='#33ffff', alpha=0.8, marker=mark[2], label=labels[2])
        ax.scatter(type3[:, 0], type3[:, 1], s=100, color='#FF3E96', alpha=0.8, marker=mark[3], label=labels[3])
        ax.scatter(type4[:, 0], type4[:, 1], s=100, color='#BF3EFF', alpha=0.8, marker=mark[4], label=labels[4])
        ax.set_xlim(-0.1, 1.3)
        ax.set_ylim(-0.1, 1.3)
        ax.legend(loc='upper right', prop=font, labelspacing=1)
    elif classes == 6:
        type0 = np.empty((0, n_dim), dtype=np.float32)
        type1 = np.empty((0, n_dim), dtype=np.float32)
        type2 = np.empty((0, n_dim), dtype=np.float32)
        type3 = np.empty((0, n_dim), dtype=np.float32)
        type4 = np.empty((0, n_dim), dtype=np.float32)
        type5 = np.empty((0, n_dim), dtype=np.float32)
        for i in range(sample_numbers):
            if input_label[i] == 0:
                type0 = np.vstack((type0, da[i]))
            elif input_label[i] == 1:
                type1 = np.vstack((type1, da[i]))
            elif input_label[i] == 2:
                type2 = np.vstack((type2, da[i]))
            elif input_label[i] == 3:
                type3 = np.vstack((type3, da[i]))
            elif input_label[i] == 4:
                type4 = np.vstack((type4, da[i]))
            elif input_label[i] == 5:
                type5 = np.vstack((type5, da[i]))
        ax.scatter(type0[:, 0], type0[:, 1], s=100, color='#2a9d8f', alpha=0.8, marker=mark[0], label=labels[0])
        ax.scatter(type1[:, 0], type1[:, 1], s=100, color='#457b9d', alpha=0.8, marker=mark[1], label=labels[1])
        ax.scatter(type2[:, 0], type2[:, 1], s=100, color='#f4a261', alpha=0.8, marker=mark[2], label=labels[2])
        ax.scatter(type3[:, 0], type3[:, 1], s=100, color='#d62828', alpha=0.8, marker=mark[3], label=labels[3])
        ax.scatter(type4[:, 0], type4[:, 1], s=100, color='#FF00FF', alpha=0.8, marker=mark[4], label=labels[4])
        ax.scatter(type5[:, 0], type5[:, 1], s=100, color='#EEA2AD', alpha=0.8, marker=mark[5], label=labels[5])
        ax.set_xlim(-0.1, 1.3)
        ax.set_ylim(-0.1, 1.3)
        ax.legend(loc='upper right', prop=font, labelspacing=1)

    root = r'E:/桌面/小论文/图片记录/TSNE图'
    path = os.path.join(root, fig_name)
    plt.savefig(path, dpi=1200)
    # print('Save t-SNE to \n', path)
    plt.show()

    # title = 't-SNE embedding of %s (time %.2fs)' % (name, (time() - t0))
    # plt.title(title)
    print('t-SNE Done!')
    return figs

