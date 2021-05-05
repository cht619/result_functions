# -*- coding: utf-8 -*-
# @Time : 2021/5/3 13:33
# @Author : CHT
# @Site : 
# @File : line_chart_kl_acc_sum.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Function:


import matplotlib as mpl
from analysis_figure import plot_components_analysis
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

font1 = {'family' : 'Times New Roman',
'weight' : 'normal', 'color':'black',
'size'  : 23,
}

font_text = {'family' : 'Times New Roman',
'weight' : 'bold', 'color':'black',
'size'  : 20,
}

cmap_names = ["viridis", "RdBu", "Set1", "jet"]  # 定义色板，方便使用！这里一共是4种风格
cmap = mpl.cm.get_cmap('RdBu', 7)
# colors = cmap(np.linspace(0, 1, 7))  # 获取7种颜色
colors = ['black', 'orange', 'green', 'blue', 'blueviolet', 'red']


def data_preprocess_KL(root_path, domain):
    # KL N个堆的话进行一个累加的操作
    # 取7个结果出来
    kl_list_sum = []
    with open(r'{}/{}'.format(root_path, domain), 'r') as f:
        csv_reader = list(csv.reader(f))

        # 取7个结果
        for i in range(1, 8):
            i = i*7 - 1
            # 取第一个元素，划分
            kl_list = csv_reader[i][0].split('\t')
            # 然后去掉标准差，直接用均值
            kl_list = [float(kl[:-4]) for kl in kl_list]
            kl_list_sum.append(np.around(sum(kl_list), 2))
    return kl_list_sum


def data_preprocess_Acc(accuracy_list):
    # 取每一个聚类的最大出来显示
    # 本来有7*7=49种结果，根据Dt选一个最高的出来
    accuracy_list = [data * 100 for data in accuracy_list]
    accuracy_list = np.asarray(accuracy_list).reshape(7, 7)
    # 0纵1横
    max_acc_in_Dt_index = np.argmax(accuracy_list, 1)  # index就是Dt的堆数
    max_acc = np.max(accuracy_list, 1)

    return max_acc_in_Dt_index, max_acc


def plot_KL_Acc(plt, max_acc_list, kl_list, colors, label, marker):
    index = np.argsort(kl_list)  # 记住第一个点是第几堆的
    kl_list = np.sort(kl_list)
    plt.plot(kl_list,  # x轴数据
             max_acc_list,  # y轴数据
             color=colors,  # 折线颜色
             marker=marker,  # 点的形状
             markersize=6,  # 点的大小
             label=label,)
             # markeredgecolor='black',  # 点的边框色
             # markerfacecolor='brown')  # 点的填充色
             #    linestyle = '-',  # 折线类型
             #    linewidth = 2,  # 折线宽度
    # for i, (x, y) in enumerate(zip(list(set(kl_list)), max_acc_list)):
    #     plt.text(x - 0.5, y + 0.01, '{}'.format(index[i] + 2))



def figure_ImageCLEF_kl_sum():
    # 对所有clusters的kl求和，然后展示最终的Acc
    kl_list = data_preprocess_KL(r'E:\cht_project\Experimental_Result\ER\Image_CLEF_Resnet50\clusters_distance_analysis', 'C_I_kl.csv')
    accuracy_list, components_in_Ds_list, components_in_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='C_I', n_data_mean=10, all_data=12
    )
    max_acc_in_Dt_index, max_acc_list = data_preprocess_Acc(accuracy_list)
    plot_KL_Acc(plt, max_acc_list=max_acc_list, kl_list=kl_list, colors=colors[0], label='C-I', marker='o')

    kl_list = data_preprocess_KL(
        r'E:\cht_project\Experimental_Result\ER\Image_CLEF_Resnet50\clusters_distance_analysis', 'P_C_kl.csv')
    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='P_C', n_data_mean=5, all_data=7
    )
    max_acc_in_Dt_index, max_acc_list = data_preprocess_Acc(accuracy_list)
    plot_KL_Acc(plt, max_acc_list=max_acc_list, kl_list=kl_list, colors=colors[1], label='P-C', marker='x')

    kl_list = data_preprocess_KL(
        r'E:\cht_project\Experimental_Result\ER\Image_CLEF_Resnet50\clusters_distance_analysis', 'I_P_kl.csv')
    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='I_P', n_data_mean=5, all_data=7
    )
    max_acc_in_Dt_index, max_acc_list = data_preprocess_Acc(accuracy_list)
    plot_KL_Acc(plt, max_acc_list=max_acc_list, kl_list=kl_list, colors=colors[2], label='I-P', marker='*')

    kl_list = data_preprocess_KL(
        r'E:\cht_project\Experimental_Result\ER\Image_CLEF_Resnet50\clusters_distance_analysis', 'I_C_kl.csv')
    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='I_C', n_data_mean=5, all_data=7
    )
    max_acc_in_Dt_index, max_acc_list = data_preprocess_Acc(accuracy_list)
    plot_KL_Acc(plt, max_acc_list=max_acc_list, kl_list=kl_list, colors=colors[3], label='I-C', marker='+')

    kl_list = data_preprocess_KL(
        r'E:\cht_project\Experimental_Result\ER\Image_CLEF_Resnet50\clusters_distance_analysis', 'C_P_kl.csv')
    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='C_P', n_data_mean=5, all_data=7
    )
    max_acc_in_Dt_index, max_acc_list = data_preprocess_Acc(accuracy_list)
    plot_KL_Acc(plt, max_acc_list=max_acc_list, kl_list=kl_list, colors=colors[4], label='C-P', marker='^')

    kl_list = data_preprocess_KL(
        r'E:\cht_project\Experimental_Result\ER\Image_CLEF_Resnet50\clusters_distance_analysis', 'P_I_kl.csv')
    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='P_I', n_data_mean=5, all_data=7
    )
    max_acc_in_Dt_index, max_acc_list = data_preprocess_Acc(accuracy_list)
    plot_KL_Acc(plt, max_acc_list=max_acc_list, kl_list=kl_list, colors=colors[5], label='P-I', marker='^')


def Figure():
    fig = plt.figure(figsize=(12, 8))
    # plt.axis([2, 8, 75.0, 95])
    # 二维图，把最高Accuracy突出来

    # get data
    # Office-Caltech
    # figure_OfficeCaltech()

    # ImageCLEF
    figure_ImageCLEF_kl_sum()


    plt.xlabel('KL', font_text)
    plt.ylabel('Accuracy (%)', font_text)
    plt.grid(linestyle='--', linewidth=2)
    # bbox_to_anchor=[x轴位置， y轴位置]， 大于1就是突出去
    plt.legend(bbox_to_anchor=(0.5,1.04), loc="center", ncol=6)  # 多少个legend就有多个
    # 设置x轴的范围为[a, b]，y轴的范围为[c, d]
    # plt.axis([2, 8, 75.0, 95])


    plt.savefig('./PNG/plot_ImageCLEF_KL.jpg')
    plt.show()


if __name__ == '__main__':
    Figure()