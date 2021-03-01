# -*- coding: utf-8 -*-
# @Time : 2021/2/28 13:47
# @Author : CHT
# @Site : 
# @File : plot_components_plot.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Function:

import matplotlib as mpl
from analysis_figure import plot_components_analysis
import matplotlib.pyplot as plt
import numpy as np
import os


font1 = {'family' : 'Times New Roman',
'weight' : 'normal', 'color':'black',
'size'  : 23,
}

font_text = {'family' : 'Times New Roman',
'weight' : 'normal', 'color':'black',
'size'  : 10,
}

cmap_names = ["viridis", "RdBu", "Set1", "jet"]  # 定义色板，方便使用！这里一共是4种风格
cmap = mpl.cm.get_cmap('jet', 7)
colors = cmap(np.linspace(0, 1, 7))  # 获取7种颜色

def data_preprocess(accuracy_list):
    # 取每一个聚类的最大出来显示
    accuracy_list = np.asarray(accuracy_list).reshape(7, 7)
    # 0纵1横
    max_acc_in_Dt_index = np.argmax(accuracy_list, 1)  # index就是Dt的堆数
    max_acc = np.max(accuracy_list, 1)

    return max_acc_in_Dt_index, max_acc


def plot(plt, max_acc_in_Dt_index, max_acc, components_in_Ds_list, color, label, marker):
    plt.plot(list(set(components_in_Ds_list)), max_acc, alpha=0.5, color=color, label=label,
             marker=marker, markersize=10)
    for i, (x, y) in enumerate(zip(list(set(components_in_Ds_list)), max_acc)):
        # 注意从2开始，所以是加2
        plt.text(x, y, '{}:{:.4}'.format(max_acc_in_Dt_index[i]+2, y))

def Figure():
    fig = plt.figure(figsize=(12, 12))
    # 二维图，把最高Accuracy突出来
    # get data
    accuracy_list, components_in_Ds_list, components_in_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='A_C',
    )
    max_acc_in_Dt_index, max_acc = data_preprocess(accuracy_list)
    plot(plt, max_acc_in_Dt_index, max_acc, components_in_Ds_list, colors[0], label='A-C', marker='o')

    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='C_A'
    )
    max_acc_in_Dt_index, max_acc = data_preprocess(accuracy_list)
    plot(plt, max_acc_in_Dt_index, max_acc, components_in_Ds_list, colors[1], label='C-A', marker='x')

    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='C_I'
    )
    max_acc_in_Dt_index, max_acc = data_preprocess(accuracy_list)
    plot(plt, max_acc_in_Dt_index, max_acc, components_in_Ds_list, colors[2], label='C-I', marker='*')

    # accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
    #     root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
    #     domain_name='Ar_Cl'
    # )
    # max_acc_in_Dt_index, max_acc = data_preprocess(accuracy_list)
    # plot(plt, max_acc_in_Dt_index, max_acc, components_in_Ds_list, colors[3], label='Ar-Cl')

    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='B_K'
    )
    max_acc_in_Dt_index, max_acc = data_preprocess(accuracy_list)
    plot(plt, max_acc_in_Dt_index, max_acc, components_in_Ds_list, colors[4], label='B-K', marker='+')

    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='train_vali'
    )
    max_acc_in_Dt_index, max_acc = data_preprocess(accuracy_list)
    plot(plt, max_acc_in_Dt_index, max_acc, components_in_Ds_list, colors[5], label='train-vali', marker='D')

    # plt.title('Multi Component Analysis', fontdict={'weight':'normal','size': 50})
    plt.xlabel('Number of components in Ds')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')

    # Plot
    # plt.plot(list(set(components_in_Ds_list)), max_acc, alpha=0.5,
    #          marker='x', markersize=20)
    # for i, (x, y) in enumerate(zip(list(set(components_in_Ds_list)), max_acc)):
    #     plt.text(x, y, '{}'.format(max_acc_in_Dt_index[i]), fontdict=font_text)

    plt.savefig('./PNG/plot1.jpg')
    plt.show()



    # print(accuracy_list, components_in_Ds_list, components_in_Dt_list)


if __name__ == '__main__':
    Figure()
