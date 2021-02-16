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
import re
import csv
from sklearn import manifold


def TSNE(data):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    data_tsne = tsne.fit_transform(data)
    print(data_tsne.min)
    x_min, x_max = data_tsne.min(0), data_tsne.max(0)
    X_norm = (data_tsne - x_min) / (x_max - x_min)  # 归一化
    return X_norm

def analysis():
    pass


if __name__ == '__main__':
    svhn_data = svhn_data.reshape(svhn_data.shape[0], -1)
    mnist_data = mnist_data.reshape(mnist_data.shape[0], -1)
    m_label = np.ones_like(mnist_label)
    s_label = np.zeros_like(svhn_label)
    label = np.concatenate((m_label[:1000], s_label[:1000]))
    data = np.concatenate((mnist_data[:1000], svhn_data[:1000]), axis=0)