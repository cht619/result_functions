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


colors_src = ['b', 'r', 'g', 'c', 'y', '#9370DB', '#FFFAFA', '#8B0000', '#90EE90', 'orange', '#FF00FF', '#90EE90']


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

    # return list
    feas_list = [[feas[i] for i in range(len(feas)) if y_pred[i] == n] for n in range(n_clusters)]
    labels_list = [[labels[i] for i in range(len(labels)) if y_pred[i] == n] for n in range(n_clusters)]

    return feas_list, labels_list



def TSNE(data):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    data_tsne = tsne.fit_transform(data)
    print(data_tsne.min)
    x_min, x_max = data_tsne.min(0), data_tsne.max(0)
    X_norm = (data_tsne - x_min) / (x_max - x_min)  # 归一化
    return X_norm


def plot_(root_path, domain_src, domain_tgt, fea_type):
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
    plt.savefig('./clusters.png')
    plt.show()


def analysis():
    pass


if __name__ == '__main__':
    # image_clef_c_data, image_clef_c_label = get_feas_labels(
    #     data_path.Image_CLEF_root_path, data_path.domain_c, fea_type='Resnet50')
    #
    # tsne_data = TSNE(image_clef_c_data)
    #
    # plt.figure(figsize=(8, 8))
    # for i in range(tsne_data.shape[0]):
    #     plt.text(tsne_data[i, 0], tsne_data[i, 1], str(int(image_clef_c_label[i])), color=colors_src[int(image_clef_c_label[i])],
    #              fontdict={'weight': 'bold', 'size': 9})
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    plot_(data_path.Image_CLEF_root_path, data_path.domain_c, data_path.domain_i, fea_type='Resnet50')
