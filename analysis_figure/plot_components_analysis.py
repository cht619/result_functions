# -*- coding: utf-8 -*-
# @Time : 2021/2/8 20:29
# @Author : CHT
# @Site : 
# @File : plot_components_analysis.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Function: 根据不同聚类的影响来画柱状图


from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import csv

# plt.style.use("ggplot")  # 改变风格，可以尝试一下

# color = ['dodgerblue', 'turquoise', 'darkcyan', 'chartreuse', 'lightpink', 'mediumslateblue', 'darkred']
color = ['orange', 'blue', 'red', 'yellow', 'blue', 'white', 'green']
alpha = np.linspace(0.2, 0.95, 49, endpoint=True)

cmap_names = ["viridis", "RdBu", "Set1", "jet"]  # 定义色板，方便使用！这里一共是4种风格
cmap = mpl.cm.get_cmap('RdBu', 7)
colors = cmap(np.linspace(0, 1, 7))  # 获取7种颜色


def get_bar(accuracy_list, components_of_Ds_list, components_of_Dt_list, fig, figure_index, title, fea_type=None):
    #设置x轴取值
    xedges = np.array(components_of_Ds_list)
    #设置y轴取值
    yedges = np.array(components_of_Dt_list)
    #设置X,Y对应点的值。即原始数据。
    accuracy =np.array(accuracy_list)

    #生成图表对象。
    # fig = plt.figure()
    #生成子图对象，类型为3d
    ax = fig.add_subplot(figure_index,projection='3d')
    ax.set_title(title)

    #设置作图点的坐标
    # xpos, ypos = np.meshgrid(xedges[:] , yedges[:])  # 获取坐标
    # print(xpos.shape, ypos.shape)
    #
    # xpos = xpos.flatten('F')
    # ypos = ypos.flatten('F')
    zpos = np.zeros_like(xedges)

    #设置柱形图大小
    dz = accuracy.flatten()
    #设置坐标轴标签
    ax.set_xlabel('Components of Ds')
    ax.set_ylabel('Components of Dt')
    ax.set_zlabel('Accuracy')
    # ax.zaix.set_ticks_position('right')

    # xticks是坐标的名字
    # ax.set_xlim(0, 10)  # 同样设置刻度值
    # ax.set_ylim(0, 10)  # 同样设置刻度值
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
    # ax.view_init(elev=45., azim=45)

    # x, y, z: array - like
    # The coordinates of the anchor point of the bars.
    # dx, dy, dz: scalar or array - like
    # The width, depth, and height of the bars, respectively.
    # minx = np.min(x)
    # maxx = np.max(x + dx)
    # miny = np.min(y)
    # maxy = np.max(y + dy)
    # minz = np.min(z)
    # maxz = np.max(z + dz)
    # ax.bar3d(xpos, ypos, zpos, dx, dy, dz,color='b',zsort='average')
    # ax.bar3d(xpos, ypos, zpos, 1, 1, dz, shade=True)
    for i in range(accuracy.shape[0]):
        # print(xedges[i], yedges[i], dz[i], color[xedges[i]-2])
        # cmap=plt.cm.gist_rainbow
        # ax.bar3d(xedges[i], yedges[i], zpos, 1, 1, dz[i], color=color[xedges[i]-2], zsort='average',
        #          alpha=0.0, linewidth=1)
        ax.bar3d(xedges[i], yedges[i], zpos, dx=1, dy=1, dz=dz[i], color=colors[xedges[i] - 2],
                 alpha=0.5, linewidth=2)

    # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    # plt.show()


def Figure():
    fig = plt.figure(figsize=(16, 16))
    plt.title('Analysis')

    plt.axis('off')  # 在一开始进行不显示

    if True:
        accuracy_list, components_of_Ds_list, components_of_Dt_list = get_mean_clustering_train_plot(
            root_path = r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
            domain_name='A_C',
        )
        get_bar(accuracy_list, components_of_Ds_list, components_of_Dt_list, fig, 231, '(a)A-C', fea_type='DeCAF6')

        accuracy_list, components_of_Ds_list, components_of_Dt_list = get_mean_clustering_train_plot(
            root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
            domain_name='C_A'
        )
        get_bar(accuracy_list, components_of_Ds_list, components_of_Dt_list, fig, 232, '(b)D-A', fea_type='DeCAF6')

        # accuracy_list, components_of_Ds_list, components_of_Dt_list = get_mean_clustering_train_plot(
        #     root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        #     domain_name='C_I'
        # )
        # get_bar(accuracy_list, components_of_Ds_list, components_of_Dt_list, fig, 233, '(c)C-I')
        #
        # accuracy_list, components_of_Ds_list, components_of_Dt_list = get_mean_clustering_train_plot(
        #     root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        #     domain_name='Ar_Cl'
        # )
        # get_bar(accuracy_list, components_of_Ds_list, components_of_Dt_list, fig, 234, '(d)Ar_Cl', fea_type='Resnet50')
        #
        # accuracy_list, components_of_Ds_list, components_of_Dt_list = get_mean_clustering_train_plot(
        #     root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        #     domain_name='B_K'
        # )
        # get_bar(accuracy_list, components_of_Ds_list, components_of_Dt_list, fig, 235, '(e)B-K')
        #
        # accuracy_list, components_of_Ds_list, components_of_Dt_list = get_mean_clustering_train_plot(
        #     root_path=r'E:\cht_project\Experimental_Result\ER\VisDA_Resnet50',
        #     domain_name='train_vali'
        # )
        # get_bar(accuracy_list, components_of_Ds_list, components_of_Dt_list, fig, 236, '(f)train_vali')

    # plt.gca().get_xaxis().set_visible(False)
    # plt.gca().get_yaxis().set_visible(False)
    # plt.xticks([])
    # plt.yticks([])
    # plt.axis('off')

    # plt.tight_layout(pad=0.4, w_pad=1, h_pad=1.0)
    # fig.tight_layout(pad=1.0, w_pad=10.0, h_pad=10.0)
    plt.savefig('./PNG/analysis_components.jpg')
    plt.show()


def get_mean_clustering_train_plot(root_path, domain_name, n_data_mean=5, all_data=7):

    # 第一行是参数情况说明
    # 第二行是
    pattern = re.compile(r'\d\|\|\d')

    accuracy_list = []
    components_of_Ds_list = []
    components_of_Dt_list = []
    csv_file = '{}.csv'.format(domain_name)
    with open(r'{}/{}'.format(root_path, csv_file), 'r') as f:
        csv_reader = list(csv.reader(f))

        # print(' {} The M0 Accuracy: {:.3f}'.format(csv_file[:5], float(csv_reader[1][0])), end=' ')

        for i in range(len(csv_reader) // all_data):  # 一个文件
            i = all_data*i
            data_list = csv_reader[i+2 : i+2+n_data_mean]
            data_list = [float(data[0]) for data in data_list]
            # get clusters
            information = csv_reader[i][0]
            clusters = pattern.findall(information)[0].split('||')
            components_of_Ds_list.append(int(clusters[0]))
            components_of_Dt_list.append(int(clusters[1]))
            accuracy_list.append(np.mean(data_list))
    # accuracy_list.sort(key=lambda x: x[0], reverse=True)
    return accuracy_list, components_of_Ds_list, components_of_Dt_list


def data_preprocess(accuracy_list):
    # 取每一个聚类的最大出来显示
    accuracy_list = np.asarray(accuracy_list).reshape(7, 7)
    # 0纵1横
    max_acc_in_Dt_index = np.argmax(accuracy_list, 1)  # index就是Dt的堆数
    max_acc = np.max(accuracy_list, 1)

    return max_acc_in_Dt_index, max_acc



if __name__ == '__main__':
    # accuracy_list, components_of_Ds_list, components_of_Dt_list =   get_mean_clustering_train_plot(
    #     r'E:\cht_project\Experimental_Result\ER\Office_Caltech_DeCAF6\Clustering_Train',
    #                                domain_name='D_A')
    # get_bar(accuracy_list, components_of_Ds_list, components_of_Dt_list)
    Figure()
    # s = np.min(accuracy_list)
    # print(s, s//0.1 / 10)
    # print(np.arange(0.01, 0.1, 0.01))

    accuracy_list = [0.8837763511606083, 0.8809349212823694, 0.8847846004722413, 0.8841429872739294, 0.8845096233872504, 0.8837763511606083, 0.8834097150472869, 0.8848762595005715, 0.8839596692172688, 0.8856095317272139, 0.8841429872739294, 0.8853345546422231, 0.8852428956138928, 0.8861594858971955, 0.8846012824155809, 0.883684692132278, 0.8856095317272139, 0.8868927581238379, 0.8863428039538561, 0.8835013740756175, 0.8852428956138928, 0.8853345546422231, 0.8826764428206448, 0.8855178726988836, 0.8844179643589202, 0.883684692132278, 0.8873510532654892, 0.884692941443911, 0.883684692132278, 0.8825847837923145, 0.8857011907555441, 0.884967918528902, 0.8821264886506631, 0.8867094400671773, 0.8846012824155807, 0.8844179643589202, 0.8820348296223328, 0.8829514199056356, 0.8847846004722413, 0.8856095317272139, 0.8823098067073236, 0.8803849671123876, 0.8816681935090116, 0.882768101848975, 0.8862511449255258, 0.8840513282455991, 0.8833180560189569, 0.8848762595005717, 0.8794683768290849]
    low = np.min(accuracy_list)
    accuracy_list_1 = [(data  - low // 0.01 / 100) * 100 for data in accuracy_list]
    print(accuracy_list_1)
    print(low, low / 0.01, low // 0.1, low // 0.01 / 100)
