# -*- coding: utf-8 -*-
# @Time : 2021/2/16 8:34
# @Author : CHT
# @Site : 
# @File : cluster_analysis.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Function:

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


colors_src = ['b', 'r', 'g', 'c', 'y', '#9370DB', '#FFFAFA', '#8B0000', '#90EE90', 'orange', '#FF00FF', '#90EE90']
colors_tgt = ['o', '#708090', '#FFFAFA', 'c', 'y', '#9370DB', '#FFFAFA', '#8B0000', '#90EE90', 'orange', '#FF00FF', '#90EE90']
marker_src = ['o', '*', '^', '<', '>']
marker_tgt = ['x', '+', 'd']


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


def plot_original_distribution(root_path, domain_src, domain_tgt, fea_type):
    feas_src, labels_src = get_feas_labels(root_path, domain_src, fea_type)
    feas_tgt, labels_tgt = get_feas_labels(root_path, domain_tgt, fea_type)

    data_tsne_src = TSNE(feas_src)
    data_tsne_tgt = TSNE(feas_tgt)

    plt.figure(figsize=(12, 12))
    for i in range(data_tsne_src.shape[0]):
        # plt.text(data_tsne_src[i, 0], data_tsne_src[i, 1], str(int(image_clef_c_label[i])),
        #          color=colors_src[int(image_clef_c_label[i])],
        #          fontdict={'weight': 'bold', 'size': 9})
        plt.scatter(data_tsne_src[i, 0], data_tsne_src[i, 1], s=50, alpha=0.5,
                    color=colors_src[int(labels_src[i])], marker='^', )
        plt.scatter(data_tsne_tgt[i, 0], data_tsne_tgt[i, 1], s=50, alpha=0.5,
                    color=colors_src[int(labels_tgt[i])], marker='*')

    plt.xticks([])
    plt.yticks([])
    plt.legend(['Ds', 'Dt'])
    plt.savefig('./clusters1.png')
    plt.show()


def plot_clusters_distribution(root_path, domain_src, fea_type='Resnet50', nC_Ds=5):
    feas_src, labels_src = get_feas_labels(root_path, domain_src, fea_type)
    data_tsne_src = TSNE(feas_src)
    # 不能够重新运行? 不能够单独用每一个类别进行运行，不然会覆盖掉

    feas_src_list, labels_src_list, labels = clustering(feas_src, labels_src, nC_Ds)

    plt.figure(figsize=(12, 12))
    for i in range(nC_Ds):
        plt.scatter(data_tsne_src[labels==i][:, 0], data_tsne_src[labels==i][:, 1],
                    marker=marker_src[i], color=colors_src[i], s=50, alpha=0.5,)

    plt.xticks([])
    plt.yticks([])
    plt.legend(['Ds', 'Dt'])
    plt.savefig('./clusters_kmeans.png')
    plt.show()


def plot_clusters_pairs_distribution_mode1(root_path, domain_src, domain_tgt, fea_type='Resnet50', nC_Ds=3, nC_Dt=3):
    # 一样的颜色就是属于一个pair，形状不同就是不同domain
    feas_src, labels_src = get_feas_labels(root_path, domain_src, fea_type)
    feas_tgt, labels_tgt = get_feas_labels(root_path, domain_tgt, fea_type)

    data_src_tsne = TSNE(feas_src)
    data_tgt_tsne = TSNE(feas_tgt)

    feas_src_list, labels_src_list, pred_labels_src = clustering(feas_src, labels_src, nC_Ds)
    feas_tgt_list, labels_gtg_list, pred_labels_tgt = clustering(feas_tgt, labels_tgt, nC_Dt)

    distance_matrix = distance_functions.get_distance_matrix(feas_src_list, feas_tgt_list, distance_method='MMD')

    pairs = get_pairs(distance_matrix)
    src_plot_index_list = []
    legend_src = []
    legend_tgt = []

    plt.figure(figsize=(12, 12))
    for i, pair in enumerate(pairs):
        print(pair)
        t = plt.scatter(data_tgt_tsne[pred_labels_tgt == pair[1]][:, 0],
                        data_tgt_tsne[pred_labels_tgt == pair[1]][:, 1],
                        color=colors_src[pair[0]], marker=marker_tgt[pair[1]])
        # greedy 重复选取数据
        if pair[0] not in src_plot_index_list:
            print(pair[0], src_plot_index_list)
            s = plt.scatter(data_src_tsne[pred_labels_src==pair[0]][:, 0], data_src_tsne[pred_labels_src==pair[0]][:, 1],
                        color=colors_src[pair[0]], marker=marker_src[pair[0]])
            src_plot_index_list.append(pair[0])
            legend_src.append(s)
            legend_tgt += ['Ds', 'Dt']
        else:
            legend_tgt.append('Dt')
        legend_src.append(t)


    plt.xticks([])
    plt.yticks([])
    plt.legend(legend_src, legend_tgt)
    plt.savefig('./clusters2.png')
    plt.show()


def plot_clusters_pairs_distribution_mode2(root_path, domain_src, domain_tgt, fea_type='Resnet50', nC_Ds=3, nC_Dt=3):
    # 不同颜色不同domain，一样的形状就是一个pair
    feas_src, labels_src = get_feas_labels(root_path, domain_src, fea_type)
    feas_tgt, labels_tgt = get_feas_labels(root_path, domain_tgt, fea_type)

    data_src_tsne = TSNE(feas_src)
    data_tgt_tsne = TSNE(feas_tgt)

    feas_src_list, labels_src_list, pred_labels_src = clustering(feas_src, labels_src, nC_Ds)
    feas_tgt_list, labels_gtg_list, pred_labels_tgt = clustering(feas_tgt, labels_tgt, nC_Dt)

    distance_matrix = distance_functions.get_distance_matrix(feas_src_list, feas_tgt_list, distance_method='MMD')
    pairs = get_pairs(distance_matrix)

    src_plot_index_list = []
    legend_src = []
    legend_tgt = []

    plt.figure(figsize=(12, 12))
    for i, pair in enumerate(pairs):
        print(pair)
        t = plt.scatter(data_tgt_tsne[pred_labels_tgt == pair[1]][:, 0],
                        data_tgt_tsne[pred_labels_tgt == pair[1]][:, 1],
                        color='orange', marker=marker_src[pair[0]])
        # greedy 重复选取数据
        if pair[0] not in src_plot_index_list:
            print(pair[0], src_plot_index_list)
            s = plt.scatter(data_src_tsne[pred_labels_src==pair[0]][:, 0], data_src_tsne[pred_labels_src==pair[0]][:, 1],
                        color='blue', marker=marker_src[pair[0]])
            src_plot_index_list.append(pair[0])
            legend_src.append(s)
            legend_tgt += ['Ds', 'Dt']
        else:
            legend_tgt.append('Dt')
        legend_src.append(t)


    plt.xticks([])
    plt.yticks([])
    plt.legend(legend_src, legend_tgt)
    plt.savefig('./clusters2_1.png')
    plt.show()


if __name__ == '__main__':
    # plot_original_distribution(data_path.Image_CLEF_root_path, data_path.domain_c, data_path.domain_i, fea_type='Resnet50')
    plot_clusters_pairs_distribution_mode2(
        data_path.Image_CLEF_root_path, data_path.domain_c, data_path.domain_i
    )
    # plot_clusters_distribution(
    #     data_path.Image_CLEF_root_path, data_path.domain_c
    # )
