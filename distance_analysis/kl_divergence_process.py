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
    # csv_files = os.listdir(data_path)
    pattern = re.compile(r'[A-Z]_[A-Z]')

    with open(r'{}'.format(r'{}'.format(data_path)), 'r') as f:
        csv_reader = list(csv.reader(f))
        print(csv_reader[0])
        for i in range(all_data):  # 每一组也有七条数据
            result = []
            for j in range(all_pairs): # 7组数据
                data = csv_reader[6 + i * 7 + 49 * j]
                # print(data[0], type(data[0]), data[0].split('±'))
                # print(data[0].split('\\t'))
                # result.append('{}'.format(j + 2))
                result.append(data[0])
                print(data[0])



if __name__ == '__main__':
    data_process(r'E:\cht_project\Experimental_Result\ER\Image_CLEF_Resnet50\clusters_distance_analysis\C_P_kl.csv')
    # name = []
    # for i in range(1, 9):
    #     s = 'Dt{}'.format(i)
    #     name.append(s)
    # print('\t'.join(name))
    # with open(r'{}'.format(r'E:\cht_project\Experimental_Result\ER\Image_CLEF_Resnet50\clusters_distance_analysis\C_I_kl.csv'), 'r') as f:
    #     csv_reader = list(csv.reader(f))
    #     print(csv_reader[0])
    #     for i in range(7):  # 7组数据
    #         result = []
    #         for j in range(7):
    #             data = csv_reader[6+i*7+49*j]
    #             # print(data[0], type(data[0]), data[0].split('±'))
    #             # print(data[0].split('\\t'))
    #             # result.append('{}'.format(j + 2))
    #             result.append(data[0])
    #             print(data[0])

            # print(result)
            # with open(r'{}'.format(
            #         r'E:\cht_project\Experimental_Result\ER\Image_CLEF_Resnet50\clusters_distance_analysis\C_I_kl1.csv'),
            #           'a+') as f:
            #     f_csv = csv.writer(f)
            #     f_csv.writerow(['\t'.join(result)])
            # print('\t'.join(result))
        # for i in range(1, 7):
        #     all_data = []
        #     for j in range(7):
        #         data = csv_reader[i + 49*j]
        #         all_data.append(data)
        #     print(all_data)
        #     print('====')
        # print(len(csv_reader), len(csv_reader)/7, csv_reader[20], csv_reader[20+49])


