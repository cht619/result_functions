# -*- coding: utf-8 -*-
# @Time : 2021/2/7 20:15
# @Author : CHT
# @Site : 
# @File : plot1.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Function:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

color = ['b', 'r', 'g', 'c', 'y', 'w', '#FFFAFA']

#设置x轴取值
xedges = np.array([2, 3, 4, 5])
print(xedges[:-1])
#设置y轴取值
yedges = np.array([2, 3, 4, 5, 6, 7, 8])
#设置X,Y对应点的值。即原始数据。
hist =np.random.rand(4 * 7)
# hist = np.array([0.7])

#生成图表对象。
fig = plt.figure()
#生成子图对象，类型为3d
ax = fig.add_subplot(111,projection='3d')

#设置作图点的坐标
xpos, ypos = np.meshgrid(xedges[:] , yedges[:])  # 获取坐标
print(xpos.shape, ypos.shape)

xpos = xpos.flatten('F')
ypos = ypos.flatten('F')
zpos = np.zeros_like(xpos)


#设置柱形图大小
dz = hist.flatten()
print(dz, xpos.shape, ypos.shape)
#设置坐标轴标签
ax.set_xlabel('Components of Ds')
ax.set_ylabel('Components of Ds')
ax.set_zlabel('Accuracy')
# ax.zaix.set_ticks_position('right')

# ax.xlim((2, 5))
ax.set_xlim(0, 10)  # 同样设置刻度值
# ax.set_zlim(0.7, 1.0)
ax.view_init(elev=45., azim=45)

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
j = 0
for i in range(hist.shape[0]):
    print(xpos[i], ypos[i], dz[i], color[xpos[i]-2])

    ax.bar3d(xpos[i], ypos[i], 0, 1, 1, dz[i], color=color[xpos[i]-2], zsort='average')
#
# plt.show()


def get_bar(accuracy_list, components_of_Ds_list, components_of_Dt_list, fig, figure_index, title, fea_type=None):
    # 设置x轴取值
    xedges = np.array(components_of_Ds_list)
    # 设置y轴取值
    yedges = np.array(components_of_Dt_list)
    # 设置X,Y对应点的值。即原始数据。
    accuracy = np.array(accuracy_list)

    # 生成图表对象。
    fig = plt.figure()
    ax = fig.add_subplot(figure_index, projection='3d')
    ax.set_title(title)

    zpos = np.zeros_like(xedges)

    # 设置柱形图大小
    dz = accuracy.flatten()
    # 设置坐标轴标签
    ax.set_xlabel('Components of Ds')
    ax.set_ylabel('Components of Dt')
    ax.set_zlabel('Accuracy')

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

    for i in range(accuracy.shape[0]):
        # print(xedges[i], yedges[i], dz[i], color[xedges[i]-2])
        # cmap=plt.cm.gist_rainbow
        # ax.bar3d(xedges[i], yedges[i], zpos, 1, 1, dz[i], color=color[xedges[i]-2], zsort='average',
        #          alpha=0.0, linewidth=1)
        ax.bar3d(xedges[i], yedges[i], zpos, dx=1, dy=1, dz=dz[i], color=color[xedges[i] - 2],
                 alpha=0.5, linewidth=2)

    # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    # plt.show()


def Figure():
    fig = plt.figure(figsize=(16, 16))
    plt.title('Analysis')

    plt.axis('off')  # 在一开始进行不显示

    if True:
        accuracy_list, components_of_Ds_list, components_of_Dt_list = get_mean_clustering_train_plot(
            root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
            domain_name='A_C',
        )
        get_bar(accuracy_list, components_of_Ds_list, components_of_Dt_list, fig, 231, '(a)A-C', fea_type='DeCAF6')

        accuracy_list, components_of_Ds_list, components_of_Dt_list = get_mean_clustering_train_plot(
            root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
            domain_name='C_A'
        )
        get_bar(accuracy_list, components_of_Ds_list, components_of_Dt_list, fig, 232, '(b)D-A', fea_type='DeCAF6')

    plt.savefig('./PNG/analysis_components.jpg')
    plt.show()

if __name__ == '__main__':
    # x = np.array([0, 1, 2])
    # y = np.array([5, 6])
    #
    # X, Y = np.meshgrid(x, y)
    # print(X)
    # print(Y)  # (0,5) (1,5) (2,5)
    def get_bar(accuracy_list, components_of_Ds_list, components_of_Dt_list, fig, figure_index, title, fea_type=None):

        # 设置x轴取值
        xedges = np.array(components_of_Ds_list)
        # 设置y轴取值
        yedges = np.array(components_of_Dt_list)
        # 设置X,Y对应点的值。即原始数据。
        accuracy = np.array(accuracy_list)

        # 生成图表对象。
        fig = plt.figure()
        ax = fig.add_subplot(figure_index, projection='3d')
        ax.set_title(title)

        zpos = np.zeros_like(xedges)

        # 设置柱形图大小
        dz = accuracy.flatten()
        # 设置坐标轴标签
        ax.set_xlabel('Components of Ds')
        ax.set_ylabel('Components of Dt')
        ax.set_zlabel('Accuracy')

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

        for i in range(accuracy.shape[0]):
            # print(xedges[i], yedges[i], dz[i], color[xedges[i]-2])
            # cmap=plt.cm.gist_rainbow
            # ax.bar3d(xedges[i], yedges[i], zpos, 1, 1, dz[i], color=color[xedges[i]-2], zsort='average',
            #          alpha=0.0, linewidth=1)
            ax.bar3d(xedges[i], yedges[i], zpos, dx=1, dy=1, dz=dz[i], color=color[xedges[i] - 2],
                     alpha=0.5, linewidth=2)

        # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        # plt.show()


    def Figure():
        fig = plt.figure(figsize=(16, 16))
        plt.title('Analysis')

        plt.axis('off')  # 在一开始进行不显示

        if True:
            accuracy_list, components_of_Ds_list, components_of_Dt_list = get_mean_clustering_train_plot(
                root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
                domain_name='A_C',
            )
            get_bar(accuracy_list, components_of_Ds_list, components_of_Dt_list, fig, 231, '(a)A-C', fea_type='DeCAF6')

            accuracy_list, components_of_Ds_list, components_of_Dt_list = get_mean_clustering_train_plot(
                root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
                domain_name='C_A'
            )
            get_bar(accuracy_list, components_of_Ds_list, components_of_Dt_list, fig, 232, '(b)D-A', fea_type='DeCAF6')

        plt.savefig('./PNG/analysis_components.jpg')
        plt.show()