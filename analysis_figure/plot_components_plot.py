# -*- coding: utf-8 -*-
# @Time : 2021/2/28 13:47
# @Author : CHT
# @Site : 
# @File : plot_components_plot.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Function:


from analysis_figure import plot_components_analysis
import matplotlib.pyplot as plt
import numpy as np
import os


font1 = {'family' : 'Times New Roman',
'weight' : 'normal', 'color':'red',
'size'  : 23,
}


def data_preprocess(accuracy_list, components_in_Ds_list, components_in_Dt_list):
    # 取每一个聚类的最大出来显示
    accuracy_list = np.asarray(accuracy_list).reshape(7, 7)
    components_in_Ds_list = np.asarray(components_in_Ds_list).reshape(7, 7)
    components_in_Dt_list = np.asarray(components_in_Dt_list).reshape(7, 7)
    # 0纵1横
    max_acc_in_Dt_index = np.argmax(accuracy_list, 1)
    max_acc = np.max(accuracy_list, 1)



    print(max_acc_in_Dt_index)

def Figure():
    # get data
    accuracy_list, components_in_Ds_list, components_in_Dt_list = plot_components_analysis.get_mean_clustering_train_plot(
        root_path=r'E:\cht_project\Experimental_Result\ER\Figure_analysis',
        domain_name='A_C',
    )
    data_preprocess(accuracy_list, components_in_Ds_list, components_in_Dt_list)

    fig = plt.figure(figsize=(16, 16))
    plt.title('Multi Component Analysis', fontdict={'weight':'normal','size': 50})
    plt.xlabel('Number of components in Ds', fontdict=font1)
    plt.ylabel('Accuracy', fontdict=font1)

    # plt.savefig('./PNG/plot1.jpg')
    # plt.show()



    # print(accuracy_list, components_in_Ds_list, components_in_Dt_list)


if __name__ == '__main__':
    x = [[11,2,3], [4,5,6]]
    print(np.max(x, 0))
    Figure()
