# -*- coding: utf-8 -*-
# @Time : 2021/2/19 20:56
# @Author : CHT
# @Site : 
# @File : plot_components_bar.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Function:

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from analysis_figure import plot_components_analysis

# plt.style.use("ggplot")  # 改变风格，可以尝试一下

# color = ['dodgerblue', 'turquoise', 'darkcyan', 'chartreuse', 'lightpink', 'mediumslateblue', 'darkred']
color = ['orange', 'blue', 'red', 'yellow', 'blue', 'white', 'green']
alpha = np.linspace(0.2, 0.95, 49, endpoint=True)

cmap_names = ["viridis", "RdBu", "Set1", "jet"]  # 定义色板，方便使用！这里一共是4种风格
cmap = mpl.cm.get_cmap('RdBu', 7)
colors = cmap(np.linspace(0, 1, 7))




def get_bar(accuracy_list, components_of_Ds_list, components_of_Dt_list, fig, figure_index, title, fea_type=None,
            enlarge=4):
    # 设置x轴取值
    xedges = np.array(components_of_Ds_list) * enlarge
    # 设置y轴取值
    yedges = np.array(components_of_Dt_list) * enlarge
    # 设置X,Y对应点的值。即原始数据。
    accuracy = np.array([np.around(data * 100, 2) for data in accuracy_list])

    # 显示最高的？
    # max_accuracy_list = np.asarray(accuracy).reshape(7, 7)
    # max_accuracy_index = np.argmax(max_accuracy_list, 1)
    # # for i in range(len(max_accuracy_index)):
    # #     if i > 0:
    # #         max_accuracy_index[i] += 7 * i
    # print(max_accuracy_index)
    # max_accuracy_index = [max_accuracy_index[i] + i*7 for i in range(len(max_accuracy_index))]

    # 生成图表对象。
    ax = fig.add_subplot(figure_index, projection='3d')
    ax.set_title(title)

    zpos = np.zeros_like(xedges)

    # 设置柱形图大小
    dz = accuracy.flatten()
    # 设置坐标轴标签
    ax.set_xlabel('Components of Ds')
    ax.set_ylabel('Components of Dt')
    ax.set_zlabel('Accuracy')
    # 设置刻度值
    ax.set_xticklabels(set(components_of_Ds_list))
    ax.set_yticklabels(set(components_of_Dt_list))

    if fea_type == 'DeCAF6':
        min_accuracy, max_accuracy = np.min(accuracy), np.max(accuracy)
        start = math.modf(min_accuracy)[1]  # 相当于np.floor
        end = np.ceil(max_accuracy)
        # start = min_accuracy // 0.01 / 100
        dz = [data - start for data in dz]
        print(start, dz, start, end)
        # ax.set_zticks()  # 这个是刻度范围
        ax.set_zticklabels(['{:.1f}'.format(step) for step in np.arange(start, end, 0.2)])

    elif fea_type == 'Resnet50':
        min_accuracy, max_accuracy = np.min(accuracy), np.max(accuracy)
        start = math.modf(min_accuracy)[1]  # 相当于np.floor
        end = np.ceil(max_accuracy)
        # start = min_accuracy // 0.01 / 100
        dz = [data - start for data in dz]
        print(start, dz, start, end)
        # ax.set_zticks()  # 这个是刻度范围
        ax.set_zticklabels(['{:.1f}'.format(step) for step in np.arange(start, end, 0.2)])

    elif fea_type == 'MDS400':
        min_accuracy, max_accuracy = np.min(accuracy), np.max(accuracy)
        start = math.modf(min_accuracy)[1]  # 相当于np.floor
        end = np.ceil(max_accuracy)
        # start = min_accuracy // 0.01 / 100
        dz = [data - start for data in dz]
        print(start, dz, start, end)
        # ax.set_zticks()  # 这个是刻度范围
        ax.set_zticklabels(['{:.1f}'.format(step) for step in np.arange(start, end, 0.2)])

    ax.view_init(elev=45., azim=45)

    for i in range(accuracy.shape[0]):

        bar = ax.bar3d(xedges[i], yedges[i], zpos, dx=1, dy=1, dz=dz[i], color=colors[xedges[i] // enlarge - 2],
                 alpha=0.5, linewidth=2)
        # if i in max_accuracy_index :
        #     print(accuracy[i])
        #     ax.text(xedges[i], yedges[i], dz[i] + 0.2, '{:.2f}'.format(accuracy[i]), ha='center',va='bottom',
        #             fontsize=5


def Figure():
    fig = plt.figure(figsize=(16, 16))
    plt.title('Multi Component Analysis')

    plt.axis('off')  # 在一开始进行不显示

    if True:
        accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
            root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
            domain_name='A_C',
        )
        get_bar(accuracy_list, components_of_Ds_list, components_of_Dt_list, fig, 231, '(a)A-C', fea_type='DeCAF6')

        accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
            root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
            domain_name='C_A'
        )
        get_bar(accuracy_list, components_of_Ds_list, components_of_Dt_list, fig, 232, '(b)D-A', fea_type='DeCAF6')

        accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
            root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
            domain_name='C_I'
        )
        get_bar(accuracy_list, components_of_Ds_list, components_of_Dt_list, fig, 233, '(c)C-I', fea_type='Resnet50')

        accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
            root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
            domain_name='Ar_Cl'
        )
        get_bar(accuracy_list, components_of_Ds_list, components_of_Dt_list, fig, 234, '(d)Ar_Cl', fea_type='Resnet50')

        accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
            root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
            domain_name='B_K'
        )
        get_bar(accuracy_list, components_of_Ds_list, components_of_Dt_list, fig, 235, '(e)B-K', fea_type='MDS400')

        accuracy_list, components_of_Ds_list, components_of_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
            root_path=r'E:\cht_project\Experimental_Result\ER\VisDA_Resnet50',
            domain_name='train_vali'
        )
        get_bar(accuracy_list, components_of_Ds_list, components_of_Dt_list, fig, 236, '(f)train_vali')

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    # 保存时候的dpi影响最后的效果
    plt.savefig('./PNG/Multi_Component_Analysis.jpg', dpi=500)
    plt.show()


if __name__ == '__main__':
    Figure()
    # print(np.arange(85, 88, 0.2))