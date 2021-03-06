# -*- coding: utf-8 -*-
# @Time : 2021/2/20 14:58
# @Author : CHT
# @Site : 
# @File : plot_feature_distribution.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Function:

import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import manifold
from scipy.io import loadmat
import torch
import torch.utils.data as data
import data_path
from sklearn.cluster import KMeans, SpectralClustering
from Distance import distance_functions
from sklearn import metrics

cmap_names = ["viridis", "RdBu", "Set1", "jet"]  # 定义色板，方便使用！这里一共是4种风格
cmap = mpl.cm.get_cmap('rainbow', 7)
colors = cmap(np.linspace(0, 1, 2))

colors_src = ['b', 'r', 'g', 'c', 'y', '#9370DB', '#FFFAFA', '#8B0000', '#90EE90', 'orange', '#FF00FF', '#90EE90']
colors_tgt = ['o', '#708090', '#FFFAFA', 'c', 'y', '#9370DB', '#FFFAFA', '#8B0000', '#90EE90', 'orange', '#FF00FF', '#90EE90']
markers = ['o', '*', '^', '<', '>', 'x', '+', 'd']

font_text = {'family' : 'Times New Roman',
'weight' : 'normal', 'color':'black',
'size'  : 20,
}


def get_feas_labels(root_path, domain, fea_type='Resnet50'):
    # 得到原始特征
    path = os.path.join(root_path, domain)
    if fea_type == 'Resnet50':
        with open(path, encoding='utf-8') as f:
            imgs_data = np.loadtxt(f, delimiter=",")
            features = imgs_data[:, :-1]
            labels = imgs_data[:, -1]

    elif fea_type =='MDS':
        # dict_keys(['__header__', '__version__', '__globals__', 'fts', 'labels'])
        domain_data = loadmat(path)
        features = np.asarray(domain_data['fts'])
        labels = np.asarray(domain_data['labels']).squeeze()

    else: # DeCAF6
        domain_data = loadmat(path)
        features = np.asarray(domain_data['feas'])
        labels = np.asarray(domain_data['labels']).squeeze() - 1  # start from 0
    return features, labels


def list_to_numpy(data_list, list_mode=1):
    data_numpy = 0
    if list_mode == 1:
        for i in range(len(data_list)):
            if i == 0:
                data_numpy = data_list[0]
            else:
                data_numpy = np.concatenate((data_numpy, data_list[i]), 0)
    elif list_mode == 2:
        # list中间还是list，所以取第一个元素进行处理
        data_numpy = np.asarray([data[0] for data in data_list])
        data_numpy = data_numpy.reshape([-1, 2048])
    return data_numpy


def get_src_dataloader_by_domain_path(root_path, domain, batch_size, drop_last=False, fea_type='Resnet50'):
    feas, labels = get_feas_labels(root_path, domain, fea_type)

    dataset = data.TensorDataset(torch.tensor(feas), torch.tensor(labels))
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,

        shuffle=True,
        drop_last=drop_last,
    )

    return dataloader


def clustering(feas, labels, n_clusters):
    k = KMeans(n_clusters=n_clusters)
    y_pred = k.fit_predict(feas)

    # print(k.labels_)

    # return list
    feas_list = [[feas[i] for i in range(len(feas)) if y_pred[i] == n] for n in range(n_clusters)]
    labels_list = [[labels[i] for i in range(len(labels)) if y_pred[i] == n] for n in range(n_clusters)]

    print('calinski_harabasz_score:', metrics.calinski_harabasz_score(feas, k.labels_))

    return feas_list, labels_list, k.labels_


def get_pairs(d_matrix):
    n_Ds, n_Dt = d_matrix.shape
    index_pairs_Ds = np.argmin(d_matrix, 0)  # 像下取 0纵1横
    pairs = [(src, tgt) for tgt, src in enumerate(index_pairs_Ds)]

    return pairs


def TSNE(data):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    data_tsne = tsne.fit_transform(data)
    print(data_tsne.min)
    x_min, x_max = data_tsne.min(0), data_tsne.max(0)
    X_norm = (data_tsne - x_min) / (x_max - x_min)  # 归一化
    return X_norm


def plot_original_distribution_pre_train(root_path, domain_src, domain_tgt, fig, figure_index, fea_type):
    # 利用不同颜色进行区分
    feas_src, labels_src = get_feas_labels(root_path, domain_src, fea_type)
    feas_tgt, labels_tgt = get_feas_labels(root_path, domain_tgt, fea_type)

    data_tsne = TSNE(np.concatenate((feas_src, feas_tgt), 0))

    ax = fig.add_subplot(figure_index)
    ax.set_title('None-adapted')

    # Ds
    ax.scatter(data_tsne[:feas_src.shape[0]][:, 0], data_tsne[:feas_src.shape[0]][:, 1], s=10, alpha=0.4,
                color='red', marker='x', )  # s是大小size
    # Dt
    ax.scatter(data_tsne[feas_src.shape[0]:][:, 0], data_tsne[feas_src.shape[0]:][:, 1], s=10, alpha=0.4,
                color='blue', marker='x', )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(['Ds', 'Dt'])


def plot_transfer_distribution_pre_train(pth_path, fig, figure_index):
    state = torch.load(pth_path)
    feas_src_f = list_to_numpy(state['feas_src_f'])
    feas_tgt_f = list_to_numpy(state['feas_src_f'])

    data_tsne = TSNE(np.concatenate((feas_src_f, feas_tgt_f), 0))
    ax = fig.add_subplot(figure_index)
    ax.set_title('Adapted')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


    # Ds
    ax.scatter(data_tsne[:feas_src_f.shape[0]][:, 0], data_tsne[:feas_src_f.shape[0]][:, 1], s=10, alpha=0.4,
               color='red', marker='x', )  # s是大小size
    # Dt
    ax.scatter(data_tsne[feas_tgt_f.shape[0]:][:, 0], data_tsne[feas_tgt_f.shape[0]:][:, 1], s=10, alpha=0.4,
               color='blue', marker='x', )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(['Ds', 'Dt'])


def plot_feas_distribution_pre_train(root_path, domain_src, domain_tgt, feas_f_pth_path, fea_type='Resnet50'):
    fig = plt.figure(figsize=(12, 6))
    # plt.axis('off')
    plt.title('Pre-train', fontdict={'weight':'normal','size': 50})
    plt.xticks([])
    plt.yticks([])
    # 原始分布
    plot_original_distribution_pre_train(
        root_path, domain_src, domain_tgt, fig, figure_index=121, fea_type=fea_type,
    )
    # 对抗迁移后的分布
    # plot_transfer_distribution_pre_train(feas_f_pth_path, fig, figure_index=122)

    plt.savefig('./PNG/Pre_train_C_I.jpg', dpi=400)
    plt.show()


def plot_original_distribution_fine_tune(feas_src_list, feas_tgt_list, clusters_pairs, clustering_labels_src,
                                         clustering_labels_tgt, fig, figure_index):

    feas_src = list_to_numpy(feas_src_list)
    feas_tgt = list_to_numpy(feas_tgt_list)

    data_tsne = TSNE(np.concatenate((feas_src, feas_tgt), 0))
    ax = fig.add_subplot(figure_index)
    ax.set_title('Adapted')

    # 因为使用greedy，所以Ds可能会重复，需要处理。
    ds_index = []
    legend_markers = []
    legend_labels = []
    for pair in clusters_pairs:
        dt = ax.scatter(data_tsne[feas_src.shape[0]:][clustering_labels_tgt == pair[1]][:, 0],
                   data_tsne[feas_src.shape[0]:][clustering_labels_tgt == pair[1]][:, 1],
                   s=10, alpha=0.4, color='blue', marker=markers[pair[0]], )  # s是大小size
        if pair[0] not in ds_index:
            ds = ax.scatter(data_tsne[:feas_src.shape[0]][clustering_labels_src == pair[0]][:, 0],
                       data_tsne[:feas_src.shape[0]][clustering_labels_src == pair[0]][:, 1],
                       s=10, alpha=0.4, color='red', marker=markers[pair[0]], )  # s是大小size
            ds_index.append(pair[0])

            legend_markers += [ds, dt]
            legend_labels += ['Ds', 'Dt']

    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(legend_markers, legend_labels)


def plot_transfer_distribution_fine_tune(
        feas_src_f_list, feas_tgt_f_list, feas_src_list, feas_tgt_list, clusters_pairs, fig, figure_index=None):

    print(np.asarray(feas_src_list[1]).shape, feas_src_f_list[1].shape)
    # 选2堆出来展示
    current_i = [clusters_pairs[1][0], clusters_pairs[1][1]]
    data_tsne = TSNE(
        np.concatenate((feas_src_list[current_i[0]], feas_tgt_list[current_i[1]]), 0))
    ax = fig.add_subplot(121)
    ax.set_title('None-adapted')
    ds = ax.scatter(data_tsne[:np.asarray(feas_src_list[current_i[0]]).shape[0]][:, 0],
                    data_tsne[:np.asarray(feas_src_list[current_i[0]]).shape[0]][:, 1],
                    s=10, alpha=0.4, color='red', marker='o', )  # s是大小size
    dt = ax.scatter(data_tsne[np.asarray(feas_src_list[current_i[1]]).shape[0]:][:, 0],
                    data_tsne[np.asarray(feas_src_list[current_i[1]]).shape[0]:][:, 1],
                    s=10, alpha=0.4, color='blue', marker='o', )  # s是大小size

    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(['Ds', 'Dt'])


    data_tsne = TSNE(
        np.concatenate((feas_src_f_list[current_i[0]], feas_tgt_f_list[current_i[1]]), 0))
    ax = fig.add_subplot(122)
    ax.set_title('Aadapted')

    ds = ax.scatter(data_tsne[:feas_src_f_list[current_i[0]].shape[0]][:, 0],
                    data_tsne[:feas_src_f_list[current_i[0]].shape[0]][:, 1],
                    s=10, alpha=0.4, color='red', marker='o', )  # s是大小size
    dt = ax.scatter(data_tsne[feas_src_f_list[current_i[1]].shape[0]:][:, 0],
                    data_tsne[feas_src_f_list[current_i[1]].shape[0]:][:, 1],
                    s=10, alpha=0.4, color='blue', marker='o', )  # s是大小size

    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(['Ds', 'Dt'])



def plot_feas_distribution_fine_tune(feas_f_pth_path, ):
    # state = {'feas_src_list': feas_src_list, 'feas_tgt_list': feas_tgt_list,
    #          'clusters_pairs': clusters_pairs, 'feas_src_f_all': feas_src_f_all,
    #          'feas_tgt_f_all': feas_tgt_f_all, 'clustering_labels_src': clustering_labels_src,
    #          'clustering_labels_tgt': clustering_labels_tgt}
    state = torch.load(feas_f_pth_path)
    feas_src_list = state['feas_src_list']
    feas_tgt_list = state['feas_tgt_list']
    feas_src_f_list = state['feas_src_f_all']
    feas_tgt_f_list = state['feas_tgt_f_all']
    clusters_pairs = state['clusters_pairs']
    clustering_labels_src = state['clustering_labels_src']
    clustering_labels_tgt = state['clustering_labels_tgt']

    fig = plt.figure(figsize=(12, 6))
    plt.title('Fine-tune', fontdict={'weight':'normal','size': 50})
    plt.xticks([])
    plt.yticks([])
    # plot_original_distribution_fine_tune(
    #     feas_src_list, feas_tgt_list, clusters_pairs, clustering_labels_src, clustering_labels_tgt, fig, 111)
    plot_transfer_distribution_fine_tune(
        feas_src_f_list, feas_tgt_f_list, feas_src_list, feas_tgt_list, clusters_pairs, fig)

    plt.savefig('./PNG/fine_tune.jpg', dpi=500)
    plt.show()


def plot_clusters_distribution(root_path, domain_src, fea_type='Resnet50', nC_Ds=5):
    feas_src, labels_src = get_feas_labels(root_path, domain_src, fea_type)
    data_tsne_src = TSNE(feas_src)
    # 不能够重新运行? 不能够单独用每一个类别进行运行，不然会覆盖掉

    feas_src_list, labels_src_list, labels = clustering(feas_src, labels_src, nC_Ds)

    plt.figure(figsize=(12, 12))
    plt.spines['top'].set_visible(False)
    plt.spines['right'].set_visible(False)
    plt.spines['bottom'].set_visible(False)
    plt.spines['left'].set_visible(False)

    for i in range(nC_Ds):
        plt.scatter(data_tsne_src[labels==i][:, 0], data_tsne_src[labels==i][:, 1],
                    marker=markers[i], color=colors_src[i], s=50, alpha=0.5,)

    plt.xticks([])
    plt.yticks([])


    plt.legend(['Ds', 'Dt'], loc='best')
    plt.savefig('./clusters_kmeans.png')
    plt.show()



def scatter_none_adapter(root_path, domain_src, domain_tgt, pth_path, fea_type='Resnet50'):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # 第一张图
    feas_src, labels_src = get_feas_labels(root_path, domain_src, fea_type)
    feas_tgt, labels_tgt = get_feas_labels(root_path, domain_tgt, fea_type)
    data_tsne_0 = TSNE(np.concatenate((feas_src, feas_tgt), 0))

    # 去掉边框
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[0].set_title('(a) None-adapted', y=-0.1, fontdict=font_text)
    # Ds
    ax[0].scatter(data_tsne_0[:feas_src.shape[0]][:, 0], data_tsne_0[:feas_src.shape[0]][:, 1], s=8, alpha=0.8,
               color='red', marker='x', )  # s是大小size
    # Dt
    ax[0].scatter(data_tsne_0[feas_src.shape[0]:][:, 0], data_tsne_0[feas_src.shape[0]:][:, 1], s=8, alpha=0.8,
               color='blue', marker='x', )
    ax[0].legend(['Ds', 'Dt'], loc='best')

    # 第二张图
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    # 读取数据
    state = torch.load(pth_path)
    feas_src_f = list_to_numpy(state['feas_src_f'])
    feas_tgt_f = list_to_numpy(state['feas_tgt_f'])
    data_tsne_1 = TSNE(np.concatenate((feas_src, feas_tgt, feas_src_f, feas_tgt_f), 0))

    ax[1].set_title('(b) Adapted', y=-0.1, fontdict=font_text)
    # Ds
    ax[1].scatter(data_tsne_0[:feas_src.shape[0]][:, 0], data_tsne_0[:feas_src.shape[0]][:, 1], s=8, alpha=0.5,
                  color='red', marker='x', )  # s是大小size
    # Dt
    ax[1].scatter(data_tsne_0[feas_src.shape[0]:][:, 0], data_tsne_0[feas_src.shape[0]:][:, 1], s=8, alpha=0.5,
                  color='blue', marker='x', )
    # Ds
    ax[1].scatter(data_tsne_1[:feas_src_f.shape[0]][:, 0], data_tsne_1[:feas_src_f.shape[0]][:, 1], s=8, alpha=0.5,
                  marker='o', c='none', edgecolors='red')  #
    # Dt
    ax[1].scatter(data_tsne_1[feas_src_f.shape[0]:][:, 0], data_tsne_1[feas_src_f.shape[0]:][:, 1], s=8, alpha=0.5,
                  marker='o', c='none', edgecolors='blue')  # 画空心图的意思就是只有边框颜色

    # 这里是一起降维，上面是分开降维
    # # Ds
    # ax[1].scatter(
    #     data_tsne_0[:feas_src.shape[0]][:, 0],
    #     data_tsne_0[:feas_src.shape[0]][:, 1],
    #     s=8, alpha=0.8, color='red', marker='x', )  # s是大小size
    # # Dt
    # ax[1].scatter(
    #     data_tsne_0[feas_src.shape[0]:feas_src.shape[0] + feas_tgt.shape[0]][:, 0],
    #     data_tsne_0[feas_src.shape[0]:feas_src.shape[0] + feas_tgt.shape[0]][:, 1],
    #     s=8, alpha=0.8, color='blue', marker='x', )
    #
    # # Ds_f
    # ax[1].scatter(
    #     data_tsne_0[feas_src.shape[0] + feas_tgt.shape[0]:feas_src.shape[0] + feas_tgt.shape[0] + feas_src_f.shape[0]][:, 0],
    #     data_tsne_0[feas_src.shape[0] + feas_tgt.shape[0]:feas_src.shape[0] + feas_tgt.shape[0] + feas_src_f.shape[0]][:, 1],
    #     s=8, alpha=0.5, color='red', marker='o', )  # s是大小size firebrick
    #
    # # Dt_f
    # ax[1].scatter(
    #     data_tsne_0[feas_src.shape[0] + feas_tgt.shape[0] + feas_src_f.shape[0]:][:, 0],
    #     data_tsne_0[feas_src.shape[0] + feas_tgt.shape[0] + feas_src_f.shape[0]:][:, 1],
    #     s=8, alpha=0.5, color='blue', marker='o', )  # s是大小size aqua

    ax[1].legend(['Ds', 'Dt', 'Ds_f', 'Dt_f'], loc='best')
    # ax[1].legend()
    plt.tight_layout(w_pad=4)
    plt.savefig('./PNG/none-adapted_all.jpg', dpi=200)
    plt.show()


def Figure(root_path, domain_src, domain_tgt, fea_type='Resnet50'):
    pass


if __name__ == '__main__':
    # plot_original_distribution(
    #     data_path.Image_CLEF_root_path, data_path.domain_c, data_path.domain_ci, fea_type='Resnet50'
    # )
    # plot_transfer_distribution(
    #     r'E:\cht_project\Kmeans_Transfer_Learning\TAT_Kmeans\pth\Image_CLEF_Resnet50\adversarial_feas\C_I\0.03\C_I_0.815.pth')

    pth_path = r'E:\cht_project\Kmeans_Transfer_Learning\TAT_Kmeans\pth\adversarial_feas\pre_train\C_I_0.815.pth'
    scatter_none_adapter(
        data_path.Image_CLEF_root_path, data_path.domain_c, data_path.domain_ci, pth_path
    )
    # pth_path = r'E:\cht_project\Kmeans_Transfer_Learning\TAT_Kmeans\pth\adversarial_feas\fine_tune\C_I_2_2.pth'
    # plot_feas_distribution_fine_tune(
    #     feas_f_pth_path=pth_path
    # )