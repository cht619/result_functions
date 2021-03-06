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
'size'  : 20,
}

cmap_names = ["viridis", "RdBu", "Set1", "jet"]  # 定义色板，方便使用！这里一共是4种风格
cmap = mpl.cm.get_cmap('RdBu', 7)
# colors = cmap(np.linspace(0, 1, 7))  # 获取7种颜色
colors = ['black', 'orange', 'green', 'blue', 'blueviolet', 'red']

def data_preprocess(accuracy_list):
    # 取每一个聚类的最大出来显示
    accuracy_list = [data * 100 for data in accuracy_list]
    accuracy_list = np.asarray(accuracy_list).reshape(7, 7)
    # 0纵1横
    max_acc_in_Dt_index = np.argmax(accuracy_list, 1)  # index就是Dt的堆数
    max_acc = np.max(accuracy_list, 1)

    return max_acc_in_Dt_index, max_acc


def plot(plt, max_acc_in_Dt_index, max_acc, components_in_Ds_list, color, label, marker):
    plt.plot(list(set(components_in_Ds_list)), max_acc, alpha=0.8, color=color, label=label,
             marker=marker, markersize=10)
    for i, (x, y) in enumerate(zip(list(set(components_in_Ds_list)), max_acc)):
        # 注意从2开始，所以是加2
        # 上面写堆数，下面写accuracy
        if label == 'I-C':
            if i == 0: plt.text(x - 0.1, y + 0.2, '{}'.format(max_acc_in_Dt_index[i]+2))
            else: plt.text(x - 0.1, y - 0.2, '{}'.format(max_acc_in_Dt_index[i] + 2))
        elif label == 'P-C':
            if i == 0:
                plt.text(x - 0.2, y - 0.4, '{}'.format(max_acc_in_Dt_index[i] + 2))
            else:
                plt.text(x - 0.1, y + 0.2, '{}'.format(max_acc_in_Dt_index[i] + 2))
        elif label == 'A-W':
            plt.text(x - 0.1, y - 0.2, '{}'.format(max_acc_in_Dt_index[i] + 2))
        else:
            plt.text(x - 0.1, y + 0.2, '{}'.format(max_acc_in_Dt_index[i]+2))
        # plt.text(x, y + 0.2, '{}'.format(max_acc_in_Dt_index[i] + 2), fontsize=13)
        # plt.text(x-0.1, y - 0.3, '{:.2f}'.format(y), fontsize=13)
    plt.tick_params(labelsize=20) # 坐标轴字体的大小

def Figure():
    fig = plt.figure(figsize=(12, 8))
    # plt.axis([2, 8, 75.0, 95])
    # 二维图，把最高Accuracy突出来

    # get data
    # Office-Caltech
    figure_OfficeCaltech()

    # ImageCLEF
    # figure_ImageCLEF()


    plt.xlabel('Number of components in Ds', font_text)
    plt.ylabel('Accuracy (%)', font_text)
    plt.grid(linestyle='--', linewidth=2)
    # bbox_to_anchor=[x轴位置， y轴位置]， 大于1就是突出去
    plt.legend(bbox_to_anchor=(0.5,1.04), loc="center", ncol=6)  # 多少个legend就有多个
    # 设置x轴的范围为[a, b]，y轴的范围为[c, d]
    # plt.axis([2, 8, 75.0, 95])


    plt.savefig('./PNG/plot1.jpg')
    plt.show()



def figure_ImageCLEF():
    accuracy_list, components_in_Ds_list, components_in_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='P_I', n_data_mean=5, all_data=7
    )
    max_acc_in_Dt_index, max_acc = data_preprocess(accuracy_list)
    plot(plt, max_acc_in_Dt_index, max_acc, components_in_Ds_list, colors[0], label='P-I', marker='o')

    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='P_C', n_data_mean=5, all_data=7
    )
    max_acc_in_Dt_index, max_acc = data_preprocess(accuracy_list)
    plot(plt, max_acc_in_Dt_index, max_acc, components_in_Ds_list, colors[1], label='P-C', marker='x')

    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='I_P', n_data_mean=5, all_data=7
    )
    max_acc_in_Dt_index, max_acc = data_preprocess(accuracy_list)
    plot(plt, max_acc_in_Dt_index, max_acc, components_in_Ds_list, colors[2], label='I-P', marker='*')

    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='I_C', n_data_mean=5, all_data=7
    )
    max_acc_in_Dt_index, max_acc = data_preprocess(accuracy_list)
    plot(plt, max_acc_in_Dt_index, max_acc, components_in_Ds_list, colors[3], label='I-C', marker='+')

    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='C_P', n_data_mean=5, all_data=7
    )
    max_acc_in_Dt_index, max_acc = data_preprocess(accuracy_list)
    plot(plt, max_acc_in_Dt_index, max_acc, components_in_Ds_list, colors[4], label='C-P', marker='^')

    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='C_I', n_data_mean=10, all_data=12
    )
    max_acc_in_Dt_index, max_acc = data_preprocess(accuracy_list)
    plot(plt, max_acc_in_Dt_index, max_acc, components_in_Ds_list, colors[5], label='C-I', marker='D')


def figure_OfficeCaltech():

    accuracy_list, components_in_Ds_list, components_in_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='C_W',  n_data_mean=10, all_data=12
    )
    max_acc_in_Dt_index, max_acc = data_preprocess(accuracy_list)
    plot(plt, max_acc_in_Dt_index, max_acc, components_in_Ds_list, colors[0], label='C-W', marker='o')

    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='A_W'
    )
    max_acc_in_Dt_index, max_acc = data_preprocess(accuracy_list)
    plot(plt, max_acc_in_Dt_index, max_acc, components_in_Ds_list, colors[1], label='A-W', marker='x')

    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='A_D'
    )
    max_acc_in_Dt_index, max_acc = data_preprocess(accuracy_list)
    plot(plt, max_acc_in_Dt_index, max_acc, components_in_Ds_list, colors[2], label='A-D', marker='*')

    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='W_A'
    )
    max_acc_in_Dt_index, max_acc = data_preprocess(accuracy_list)
    plot(plt, max_acc_in_Dt_index, max_acc, components_in_Ds_list, colors[3], label='C-A', marker='+')

    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='C_A'
    )
    max_acc_in_Dt_index, max_acc = data_preprocess(accuracy_list)
    plot(plt, max_acc_in_Dt_index, max_acc, components_in_Ds_list, colors[4], label='C-A', marker='^')

    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='A_C'
    )
    max_acc_in_Dt_index, max_acc = data_preprocess(accuracy_list)
    plot(plt, max_acc_in_Dt_index, max_acc, components_in_Ds_list, colors[5], label='A-C', marker='D')

    # plt.title('Multi Component Analysis', fontdict={'weight':'normal','size': 50})

if __name__ == '__main__':
    Figure()
