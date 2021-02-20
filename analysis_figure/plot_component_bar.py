# -*- coding: utf-8 -*-
# @Time : 2021/2/19 20:56
# @Author : CHT
# @Site : 
# @File : plot_component_bar.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Function:

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from analysis_figure import plot_components_analysis

# plt.style.use("ggplot")  # 改变风格，可以尝试一下

# color = ['dodgerblue', 'turquoise', 'darkcyan', 'chartreuse', 'lightpink', 'mediumslateblue', 'darkred']
color = ['orange', 'blue', 'red', 'yellow', 'blue', 'white', 'green']
alpha = np.linspace(0.2, 0.95, 49, endpoint=True)

cmap_names = ["viridis", "RdBu", "Set1", "jet"]  # 定义色板，方便使用！这里一共是4种风格
cmap = mpl.cm.get_cmap('RdBu', 7)
colors = cmap(np.linspace(0, 1, 7))




def get_bar(accuracy_list, components_of_Ds_list, components_of_Dt_list, fig, figure_index, title, fea_type=None,
            enlarge=3):
    # 设置x轴取值
    xedges = np.array(components_of_Ds_list) * enlarge
    # 设置y轴取值
    yedges = np.array(components_of_Dt_list) * enlarge
    # 设置X,Y对应点的值。即原始数据。
    accuracy = np.array(accuracy_list)

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

    if fea_type == 'DeCAF6':
        print(accuracy_list)  # 883 884
        min_accuracy = np.min(accuracy_list)
        low = min_accuracy // 0.01 / 101
        # ax.set_zlim(low, 1.0)
        dz = [data - low for data in dz]
        print(low, dz)
        # ax.set_zticks()  # 这个是刻度范围
        ax.set_zticklabels(['{:.2f}'.format(low + step) for step in np.arange(0.01, 0.1, 0.01)])

    elif fea_type == 'Resnet50':
        print(accuracy_list)
        min_accuracy = np.min(accuracy_list)
        # ax.set_zlim(0.7, 1.0)
    ax.view_init(elev=45., azim=45)

    for i in range(accuracy.shape[0]):

        ax.bar3d(xedges[i], yedges[i], zpos, dx=1, dy=1, dz=dz[i], color=colors[xedges[i] // enlarge - 2],
                 alpha=0.5, linewidth=2)

    # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    # plt.show()


def Figure():
    fig = plt.figure(figsize=(16, 16))
    plt.title('Analysis')

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
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig('./PNG/analysis_components_1.jpg', dpi=500)
    plt.show()


if __name__ == '__main__':
    Figure()