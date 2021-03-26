# -*- coding: utf-8 -*-
# @Time : 2021/3/26 9:24
# @Author : CHT
# @Site : 
# @File : kl_divergence_process.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Function:

import os
import re
import csv
import numpy as np


def data_process(data_path, all_data=7, all_pairs=7):
    # 处理结果，原来是这种形式的，要换一下
    # Ds=2 Dt=2
    # Ds=2 Dt=3
    # Ds=2 Dt=4
    # Ds=2 Dt=5
    # Ds=2 Dt=6
    # Ds=2 Dt=7
    # Ds=2 Dt=8
    # 实现排序的时候就简单处理了！
    pattern = re.compile(r'(\d\d)±')

    with open(r'{}'.format(data_path), 'r') as f:
        csv_reader = list(csv.reader(f))
        print(csv_reader[0])
        for i in range(all_data):  # Dt有7种情况 2开始
            dt_pair_data = []  # Dt一样的情况
            for k in range(all_pairs):  # Dt一样的情况Ds有7种,7种的间隔是49
                ds_pair_data = []
                for j in range(1, 6):  # 5个结果
                    data = csv_reader[j + i * 7 + k * 49][0]
                    # pair_data.append(csv_reader[j][0])
                    data = [float(d) for d in data.split('\t')]
                    ds_pair_data.append(data)
                dt_pair_mean = np.mean(ds_pair_data, axis=0)  # mean
                dt_pair_std = np.std(ds_pair_data, axis=0, ddof=1)  # std
                result = ['{:.2f}±{:.1f}'.format(mean, std) for (mean, std) in zip(dt_pair_mean, dt_pair_std)]
                print('\t'.join(result))  # 输出7个结果
                dt_pair_data.append(ds_pair_data)
            # 计算全局平均
            dt_pair_data = np.asarray(dt_pair_data)
            # global_mean = np.mean(dt_pair_data.reshape(-1, i+2), axis=0)
            # global_std = np.std(dt_pair_data.reshape(-1, i+2), axis=0, ddof=1)
            # result = ['{:.2f}±{:.1f}'.format(mean ,std) for (mean ,std) in zip(global_mean, global_std)]
            # print('全局平均:---','\t'.join(result))
            # 计算全局最小  其实可以都可以在这里设置，但是为了粘贴方便，这里仅仅处理最小值
            # print('全局最小:---',end='')
            for col in range(i + 2):
                minimum_mean = np.mean(dt_pair_data, axis=1)
                minimum_std = np.std(dt_pair_data, axis=1, ddof=1)
                # print(minimum_mean)
                index = np.argmin(minimum_mean[:, col])
                print('{:.2f}+{:.1f}'.format(minimum_mean[index, col], minimum_std[index, col]), end='\t')
            print('\n')
            global_mean = np.mean(dt_pair_data.reshape(-1, i + 2), axis=0)
            global_std = np.std(dt_pair_data.reshape(-1, i + 2), axis=0, ddof=1)
            result = ['{:.2f}±{:.1f}'.format(mean, std) for (mean, std) in zip(global_mean, global_std)]
            print('\t'.join(result))
            print('\n当前已经结束\n')



if __name__ == '__main__':
    # data_process(r'E:\cht_project\Experimental_Result\ER\Image_CLEF_Resnet50\clusters_distance_analysis\I_C_kl.csv')
    # name = []
    # for i in range(1, 9):
    #     s = 'Dt{}'.format(i)
    #     name.append(s)
    # print('\t'.join(name))
    # pattern = re.compile(r'(\d\d)±')
    with open(r'{}'.format(r'E:\cht_project\Experimental_Result\ER\Image_CLEF_Resnet50\clusters_distance_analysis\I_C_kl.csv'), 'r') as f:
        csv_reader = list(csv.reader(f))
        print(csv_reader[0])
        for i in range(7): # Dt有7种情况 2开始
            dt_pair_data = []  # Dt一样的情况
            for k in range(7):  # Dt一样的情况Ds有7种,7种的间隔是49
                ds_pair_data = []
                for j in range(1, 6): # 5个结果
                    data = csv_reader[j+i*7+k*49][0]
                    # pair_data.append(csv_reader[j][0])
                    data = [float(d) for d in data.split('\t')]
                    ds_pair_data.append(data)
                dt_pair_mean = np.mean(ds_pair_data, axis=0)  # mean
                dt_pair_std = np.std(ds_pair_data, axis=0, ddof=1)  # std
                result = ['{:.2f}±{:.1f}'.format(mean, std) for (mean, std) in zip(dt_pair_mean, dt_pair_std)]
                print('\t'.join(result))  # 输出7个结果
                dt_pair_data.append(ds_pair_data)
            # 计算全局平均
            dt_pair_data = np.asarray(dt_pair_data)
            # global_mean = np.mean(dt_pair_data.reshape(-1, i+2), axis=0)
            # global_std = np.std(dt_pair_data.reshape(-1, i+2), axis=0, ddof=1)
            # result = ['{:.2f}±{:.1f}'.format(mean ,std) for (mean ,std) in zip(global_mean, global_std)]
            # print('全局平均:---','\t'.join(result))
            # 计算全局最小  其实可以都可以在这里设置，但是为了粘贴方便，这里仅仅处理最小值
            # print('全局最小:---',end='')
            for col in range(i+2):
                minimum_mean = np.mean(dt_pair_data, axis=1)
                minimum_std = np.std(dt_pair_data, axis=1, ddof=1)
                # print(minimum_mean)
                index = np.argmin(minimum_mean[:, col])
                print('{:.2f}+{:.1f}'.format(minimum_mean[index, col], minimum_std[index, col]), end='\t')
            print('\n')
            global_mean = np.mean(dt_pair_data.reshape(-1, i + 2), axis=0)
            global_std = np.std(dt_pair_data.reshape(-1, i + 2), axis=0, ddof=1)
            result = ['{:.2f}±{:.1f}'.format(mean, std) for (mean, std) in zip(global_mean, global_std)]
            print('\t'.join(result))
            print('\n当前已经结束\n')

            # print(dt_pair_data,)
            # print(np.asarray(dt_pair_data).shape)


