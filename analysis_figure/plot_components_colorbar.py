# -*- coding: utf-8 -*-
# @Time : 2021/2/28 16:15
# @Author : CHT
# @Site : 
# @File : plot_components_colorbar.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Function:


import numpy as np
import matplotlib.pyplot as plt
from analysis_figure import plot_components_analysis


font1 = {'family' : 'Times New Roman',
'weight' : 'normal', 'color':'black',
'size'  : 15,
}

font_text = {'family' : 'Times New Roman',
'weight' : 'normal', 'color':'black',
'size'  : 10,
}


def data_preprocess(accuracy_list):
    # 取每一个聚类的最大出来显示
    accuracy_list = np.asarray(accuracy_list).reshape(7, 7)
    # 0纵1横
    max_acc_in_Dt_index = np.argmax(accuracy_list, 1)  # index就是Dt的堆数
    max_acc = np.max(accuracy_list, 1)

    return max_acc_in_Dt_index, max_acc


def colorbar():
    pass


def Figure():
    accuracy_list, components_in_Ds_list, components_in_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='A_C',
    )
    max_acc_in_Dt_index, max_acc = data_preprocess(accuracy_list)
    plot(plt, max_acc_in_Dt_index, max_acc, components_in_Ds_list, colors[0], label='A-C')

    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='C_A'
    )
    max_acc_in_Dt_index, max_acc = data_preprocess(accuracy_list)
    plot(plt, max_acc_in_Dt_index, max_acc, components_in_Ds_list, colors[1], label='C-A')

    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='C_I'
    )
    max_acc_in_Dt_index, max_acc = data_preprocess(accuracy_list)
    plot(plt, max_acc_in_Dt_index, max_acc, components_in_Ds_list, colors[2], label='C-I')

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
    plot(plt, max_acc_in_Dt_index, max_acc, components_in_Ds_list, colors[4], label='B-K')

    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='train_vali'
    )
    max_acc_in_Dt_index, max_acc = data_preprocess(accuracy_list)
    plot(plt, max_acc_in_Dt_index, max_acc, components_in_Ds_list, colors[5], label='train-vali')


if __name__ == '__main__':
    accuracy_list, components_in_Ds_list, components_in_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='A_C',
    )


    fig = plt.figure(figsize=(12, 12))

    x = np.asarray(accuracy_list).reshape(7, 7)
    plt.matshow(x, cmap=plt.cm.cool, vmin=0.88, vmax=0.889)
    plt.colorbar()
    plt.xticks([0, 1, 2, 3, 4, 5, 6], list(set(components_in_Dt_list)))
    plt.yticks([0, 1, 2, 3, 4, 5, 6], list(set(components_in_Dt_list)))
    # plt.tight_layout()

    plt.title('Multi Component Analysis', pad=20, fontdict=font1)
    plt.xlabel('Number of components in Ds', fontdict=font1)
    plt.ylabel('Number of components in Dt', fontdict=font1)
    plt.savefig('./PNG/colorbar.png', bbox_inches='tight')
    plt.show()

    # print(plt.cm.cmap_d)
