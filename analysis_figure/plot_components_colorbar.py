# -*- coding: utf-8 -*-
# @Time : 2021/2/28 16:15
# @Author : CHT
# @Site : 
# @File : plot_components_colorbar.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Function: 这里绘制热力图


import numpy as np
import matplotlib.pyplot as plt
from analysis_figure import plot_components_analysis
import seaborn as sns



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


def colorbar(accuracy_list, components_in_Ds_list, components_in_Dt_list, fig, figure_index, title):
    ax = fig.add_subplot(figure_index)
    min_acc = np.min(accuracy_list)
    max_acc = np.max(accuracy_list)
    x = np.asarray(accuracy_list).reshape(7, 7)
    ax.set_title(title)
    # ax.set_xticks([0, 1, 2, 3, 4, 5, 6], list(set(components_in_Ds_list)))
    # ax.set_yticks([0, 1, 2, 3, 4, 5, 6], list(set(components_in_Dt_list)))
    img = ax.imshow(x, cmap=plt.cm.jet, vmin=min_acc, vmax=max_acc)
    # ax.colorbar()
    plt.colorbar(img, ax=ax)


def Figure():
    fig = plt.figure(figsize=(12, 12), dpi=100) # 定义大小
    # print(help(plt.cm))

    accuracy_list, components_in_Ds_list, components_in_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='A_C',
    )
    colorbar(accuracy_list, components_in_Ds_list, components_in_Dt_list, fig, figure_index=221, title='A-C')
    plt.xticks([0, 1, 2, 3, 4, 5, 6], list(set(components_in_Dt_list)))
    plt.yticks([0, 1, 2, 3, 4, 5, 6], list(set(components_in_Dt_list)).reverse())

    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='C_A'
    )
    colorbar(accuracy_list, components_in_Ds_list, components_in_Dt_list, fig, figure_index=222, title='C-A')
    plt.xticks([0, 1, 2, 3, 4, 5, 6], list(set(components_in_Dt_list)))
    plt.yticks([0, 1, 2, 3, 4, 5, 6], list(set(components_in_Dt_list)).reverse())

    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='C_I'
    )
    colorbar(accuracy_list, components_in_Ds_list, components_in_Dt_list, fig, figure_index=223, title='C-I')
    plt.xticks([0, 1, 2, 3, 4, 5, 6], list(set(components_in_Dt_list)))
    plt.yticks([0, 1, 2, 3, 4, 5, 6], list(set(components_in_Dt_list)).reverse())

    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='Ar_Cl'
    )
    colorbar(accuracy_list, components_in_Ds_list, components_in_Dt_list, fig, figure_index=224, title='Ar-Cl')
    plt.xticks([0, 1, 2, 3, 4, 5, 6], list(set(components_in_Dt_list)))
    plt.yticks([0, 1, 2, 3, 4, 5, 6], list(set(components_in_Dt_list)).reverse())

    # accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
    #     root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
    #     domain_name='B_K'
    # )
    # colorbar(accuracy_list, components_in_Ds_list, components_in_Dt_list, fig, figure_index=224, title='B-K')
    # plt.xticks([0, 1, 2, 3, 4, 5, 6], list(set(components_in_Dt_list)))
    # plt.yticks([0, 1, 2, 3, 4, 5, 6], list(set(components_in_Dt_list)))
    # #
    # accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
    #     root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
    #     domain_name='train_vali'
    # )
    # colorbar(accuracy_list, components_in_Ds_list, components_in_Dt_list, fig, figure_index=224, title='Ar-Cl')
    # plt.xticks([0, 1, 2, 3, 4, 5, 6], list(set(components_in_Dt_list)))
    # plt.yticks([0, 1, 2, 3, 4, 5, 6], list(set(components_in_Dt_list)))
    #

    plt.savefig('./PNG/colorbar.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    Figure()
    # accuracy_list, components_in_Ds_list, components_in_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
    #     root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
    #     domain_name='A_C',
    # )
    # print(accuracy_list)
    # min_acc = np.min(accuracy_list)
    # max_acc = np.max(accuracy_list)
    #
    #
    # fig = plt.figure(figsize=(12, 12))
    #
    # x = np.asarray(accuracy_list).reshape(7, 7)
    # plt.matshow(x, cmap=plt.cm.cool, vmin=min_acc, vmax=max_acc)
    # plt.colorbar()
    # plt.xticks([0, 1, 2, 3, 4, 5, 6], list(set(components_in_Dt_list)))
    # plt.yticks([0, 1, 2, 3, 4, 5, 6], list(set(components_in_Dt_list)))
    # # plt.tight_layout()
    #
    # plt.title('Multi Component Analysis', pad=20, fontdict=font1)
    # plt.xlabel('Number of components in Ds', fontdict=font1)
    # plt.ylabel('Number of components in Dt', fontdict=font1)
    # plt.savefig('./PNG/colorbar.png', bbox_inches='tight')
    # plt.show()

    # print(plt.cm.cmap_d)
