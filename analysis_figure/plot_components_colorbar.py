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
'weight' : 'bold', # 字体加速
 'color':'black',
'size'  : 30,
}

font_text = {'family' : 'Times New Roman',
'weight' : 'normal', 'color':'black',
'size'  : 10,
}


def data_preprocess(accuracy_list):
    # 把数据交换一下，2本来在最上面改成在最下面
    accuracy_list = np.asarray(accuracy_list).reshape(7, 7)
    for i in range(len(accuracy_list) // 2):
        accuracy_list[i] = accuracy_list[-(i+1)]
    pass


def colorbar(accuracy_list, components_in_Ds_list, components_in_Dt_list, fig, figure_index, title):
    ax = fig.add_subplot(figure_index)
    min_acc = np.min(accuracy_list)
    max_acc = np.max(accuracy_list)
    x = np.asarray(accuracy_list).reshape(7, 7)
    # ax.set_title(title)
    # ax.set_xticklabels([0, 1, 2, 3, 4, 5, 6], list(set(components_in_Ds_list)))
    # ax.set_yticklabels([0, 1, 2, 3, 4, 5, 6], list(set(components_in_Dt_list)))
    img = ax.matshow(x, cmap=plt.cm.cool, vmin=min_acc, vmax=max_acc)
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
    plt.yticks([0, 1, 2, 3, 4, 5, 6], [8, 7, 6, 5, 4, 3, 2].reverse())

    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='C_A'
    )
    colorbar(accuracy_list, components_in_Ds_list, components_in_Dt_list, fig, figure_index=222, title='C-A')
    plt.xticks([0, 1, 2, 3, 4, 5, 6], list(set(components_in_Dt_list)))
    plt.yticks([0, 1, 2, 3, 4, 5, 6], [8, 7, 6, 5, 4, 3, 2])

    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='C_I'
    )
    colorbar(accuracy_list, components_in_Ds_list, components_in_Dt_list, fig, figure_index=223, title='C-I')
    plt.xticks([0, 1, 2, 3, 4, 5, 6], list(set(components_in_Dt_list)))
    plt.yticks([0, 1, 2, 3, 4, 5, 6], [8, 7, 6, 5, 4, 3, 2])

    accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='Ar_Cl'
    )
    colorbar(accuracy_list, components_in_Ds_list, components_in_Dt_list, fig, figure_index=224, title='Ar-Cl')
    plt.xticks([0, 1, 2, 3, 4, 5, 6], list(set(components_in_Dt_list)))
    plt.yticks([0, 1, 2, 3, 4, 5, 6], [8, 7, 6, 5, 4, 3, 2])

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


def Figure_1(data_path, domain_name, title, n_data_mean=5, ):
    # n_data_mean就是每一个情况有多少个个数据，然后取平均。
    # plt.figure(figsize=(12, 12), dpi=100)  # 定义大小
    fig, ax = plt.subplots(figsize=(12, 12))
    # print(help(plt.cm))

    accuracy_list, components_in_Ds_list, components_in_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=data_path, domain_name=domain_name, n_data_mean=n_data_mean, all_data=n_data_mean+2
    )
    min_acc = np.min(accuracy_list)
    max_acc = np.max(accuracy_list)
    # xlim(min, max)就是现实的范围
    accuracy_list = np.asarray(accuracy_list).reshape(7, 7)

    # plt.set_xticks([2, 3, 4, 5, 6, 7, 8])
    # ax.set_yticks([0, 1, 2, 3, 4, 5, 6], list(set(components_in_Dt_list)))
    # ax.set_yticks([2, 3, 4, 5, 6, 7, 8])
    ax.set_title(title, y=-0.05, fontdict=font1)
    cbar = sns.heatmap(
        accuracy_list, vmin=min_acc, vmax=max_acc, ax=ax, cmap=plt.cm.cool,
        annot=False,  # 每个格子要不要显示数据
        cbar=True, # 要不要右边那个柱子
        # 设置cbar的各类属性
        cbar_kws={'format': '%.2f'},
        xticklabels=[2, 3, 4, 5, 6, 7, 8], yticklabels=list(set(components_in_Dt_list)))
    # plt.xticks([0, 1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7, 8])
    # ax.set_yticks(list(set(components_in_Dt_list)))
    # ax.invert_yaxis() # y轴数据调转

    #
    print(len(ax.figure.axes))
    cbar_axes = ax.figure.axes[1]  # 获取最后一列
    cbar_axes.tick_params(labelsize=30)
    # 设置图例大小
    # cb = cbar.figure.colorbar(cbar.collections[0])  # 显示colorbar
    # cb.ax.tick_params(labelsize=28)  # 设置colorbar刻度字体大小。

    # 坐标轴设置
    ax.xaxis.tick_top()  # x轴放到最上面
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)



    # ax.set_xticks([2, 3, 4, 5, 6, 7, 8])
    plt.savefig('./PNG/colorbar/{}.png'.format(domain_name), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    # Figure()
    # Figure_1(data_path=r'E:\cht_project\Experimental_Result\ER\Image_CLEF_Resnet50\MMD\3.30',
    #     domain_name='C_I', title='C-I', n_data_mean=3)
    Figure_1(data_path=r'E:\cht_project\Experimental_Result\ER\Image_CLEF_Resnet50\MMD\3.30',
             domain_name='C_P', title='C-P', n_data_mean=3)
    # Figure_1(data_path=r'E:\cht_project\Experimental_Result\ER\Image_CLEF_Resnet50\MMD\3.30',
    #          domain_name='I_P', title='I-P', n_data_mean=3)
    # Figure_1(data_path=r'E:\cht_project\Experimental_Result\ER\Image_CLEF_Resnet50\MMD\3.30',
    #          domain_name='I_C', title='I-C', n_data_mean=3)
    # Figure_1(data_path=r'E:\cht_project\Experimental_Result\ER\Image_CLEF_Resnet50\MMD\3.30',
    #          domain_name='P_C', title='P-C', n_data_mean=3)
    # Figure_1(data_path=r'E:\cht_project\Experimental_Result\ER\Image_CLEF_Resnet50\MMD\3.30',
    #          domain_name='P_I', title='P-I', n_data_mean=3)
    # Figure_1(data_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
    #          domain_name='I_C', title='I-C')
    # Figure_1(data_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
    #          domain_name='I_P', title='I_P')
    # Figure_1(data_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
    #          domain_name='P_C', title='P-C')
    # Figure_1(data_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
    #          domain_name='P_I', title='P-I')
    # Figure_1(data_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
    #          domain_name='Ar_Cl', title='Ar-Cl', n_data_mean=10)
    # Figure_1(data_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
    #          domain_name='Pr_Rw', title='Pr-Rw', n_data_mean=10)

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
