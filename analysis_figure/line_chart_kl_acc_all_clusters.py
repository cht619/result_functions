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
colors = ['black', 'orange', 'green', 'blue', 'blueviolet', 'red', 'yellow']
markers = ['o', 'x', '+', '*', '^', 'D', 's']


def data_preprocess_KL(root_path, domain):
    # 第一行是每一个堆的kl，
    # 第二行是每一个堆的acc，并且最后一行是总的Acc，用来筛选最好的情况
    with open(r'{}/{}'.format(root_path, domain), 'r') as f:
        csv_reader = list(csv.reader(f))

        # 筛选出最高的情况, 一共是7种情况，七条曲线
        acc_best_list = []
        kl_best_list = []
        for i in range(0, 98, 14):  # 一共有49个结果，并且每2*7是一组
            # Dt为N的Ds的各种情况
            kl_clusters_list = [csv_reader[i+j] for j in range(0, 14, 2)]  # 取出所有
            acc_clusters_list = [csv_reader[i+j+1] for j in range(0, 14, 2)]  # 多一行
            kl_clusters_list = np.asarray([[float(kl) for kl in kl_list] for kl_list in kl_clusters_list])
            acc_clusters_list = np.asarray([[float(acc) for acc in acc_list] for acc_list in acc_clusters_list])
            # 最后一列代表的是最终的Acc，所以取出来
            max_acc_index = np.argmax(acc_clusters_list, axis=0)[-1]
            acc_best_list.append(acc_clusters_list[max_acc_index])
            kl_best_list.append(kl_clusters_list[max_acc_index])
    return acc_best_list, kl_best_list


def data_preprocess_Acc(accuracy_list):
    # 调成百分比的形式，并且不要最后一列
    accuracy_list = [[data_list[i] * 100 for i in range(len(data_list) - 1)] for data_list in accuracy_list]

    return accuracy_list


def plot_KL_Acc(plt, acc_best_list, kl_best_list):
    # index = np.argsort(kl_list)  # 记住第一个点是第几堆的
    # kl_list = np.sort(kl_list)
    # plt.plot(kl_list,  # x轴数据
    #          max_acc_list,  # y轴数据
    #          color=colors,  # 折线颜色
    #          marker=marker,  # 点的形状
    #          markersize=6,  # 点的大小
    #          label=label,)
             # markeredgecolor='black',  # 点的边框色
             # markerfacecolor='brown')  # 点的填充色
             #    linestyle = '-',  # 折线类型
             #    linewidth = 2,  # 折线宽度
    for i in range(len(acc_best_list)):
        # 同样对KL排序,升序
        # if True:
        if i in [ 1, 2]:
            sort_index = np.argsort(kl_best_list[i])
            kl_list = np.asarray(kl_best_list[i])[sort_index]
            acc_list = np.asarray(acc_best_list[i])[sort_index]
            plt.plot(kl_list,  # x轴数据
                     acc_list,  # y轴数据
                     color=colors[i],  # 折线颜色
                     marker=markers[i],  # 点的形状
                     markersize=6,  # 点的大小
                     label='N={}'.format(i+2), )



def figure_ImageCLEF_kl_sum():
    # 对所有clusters的kl求和，然后展示最终的Acc
    # acc_best_list, kl_best_list = data_preprocess_KL(r'E:\cht_project\Experimental_Result\ER\Image_CLEF_Resnet50\Clustering_Train\KL_acc\5.5', 'C_I.csv')
    # acc_best_list = data_preprocess_Acc(acc_best_list)
    # plot_KL_Acc(plt, acc_best_list=acc_best_list, kl_best_list=kl_best_list)

    # acc_best_list, kl_best_list = data_preprocess_KL(
    #     r'E:\cht_project\Experimental_Result\ER\Image_CLEF_Resnet50\Clustering_Train\KL_acc\5.5', 'C_P.csv')
    # acc_best_list = data_preprocess_Acc(acc_best_list)
    # plot_KL_Acc(plt, acc_best_list=acc_best_list, kl_best_list=kl_best_list)

    acc_best_list, kl_best_list = data_preprocess_KL(
        r'E:\cht_project\Experimental_Result\ER\Image_CLEF_Resnet50\Clustering_Train\KL_acc\5.5', 'I_C.csv')
    acc_best_list = data_preprocess_Acc(acc_best_list)
    plot_KL_Acc(plt, acc_best_list=acc_best_list, kl_best_list=kl_best_list)


def Figure():
    fig = plt.figure(figsize=(12, 8))
    # plt.axis([2, 8, 75.0, 95])
    # 二维图，把最高Accuracy突出来

    # get data
    # Office-Caltech
    # figure_OfficeCaltech()

    # ImageCLEF
    figure_ImageCLEF_kl_sum()

    plt.title('I-C')
    plt.xlabel('KL', font_text)
    plt.ylabel('Accuracy (%)', font_text)
    plt.grid(linestyle='--', linewidth=2)
    # bbox_to_anchor=[x轴位置， y轴位置]， 大于1就是突出去
    plt.legend(bbox_to_anchor=(0.5,1.04), loc="center", ncol=7)  # 多少个legend就有多个
    # 设置x轴的范围为[a, b]，y轴的范围为[c, d]
    # plt.axis([2, 8, 75.0, 95])


    plt.savefig('./PNG/KL/KL_I_C.jpg')
    plt.show()


if __name__ == '__main__':
    Figure()