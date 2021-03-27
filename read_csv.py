# -*- coding: utf-8 -*-
# @Time : 2021/1/3 8:59
# @Author : CHT
# @Site : 
# @File : read_csv.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Function:

import numpy as np
import csv
import os
import re


def get_mean_standard_deviation(root_path, n_data_mean=14):
    csv_files = os.listdir(root_path)
    all_result = []

    for csv_file in csv_files:
        with open(r'{}/{}'.format(root_path, csv_file), 'r') as f:
            csv_reader = csv.reader(f)
            # n Dtl
            data_list_Dtl_0 = []
            data_list_Dtl_3 = []

            for i, row in enumerate(csv_reader):
                if i < 5:
                    data_list_Dtl_0.append(float(row[0]))
                elif i > 5:
                    data_list_Dtl_3.append(float(row[0]))

        data_list_Dtl_3.sort(reverse=True)

        std_3 = np.std(data_list_Dtl_3[:n_data_mean], ddof=1)* 100
        std_0 = np.std(data_list_Dtl_0[:], ddof=1)* 100

        result = '{:.3}±{:.1f}({:.3}±{:.1f})'.format(np.mean(data_list_Dtl_3[:n_data_mean]) * 100,
                  std_3,
                np.mean(data_list_Dtl_0[:])* 100,
                std_0)
        # TAT
        # result = '{:.3}±{:.1f}'.format(np.mean(data_list_Dtl_0[:]) * 100, std_0)
        # result = '{:.3}±{:.1f}'.format(np.mean(data_list_Dtl_3[:]) * 100, std_3)
        all_result.append(result)

        print('{}'.format(csv_file[:-3]), end=' ')
    print('\n')
    for result in all_result:
        print(result, end='\t')


def get_mean_standard_deviation_no_0dtl(root_path, n_data_mean=5):
    csv_files = os.listdir(root_path)
    all_result = []

    for csv_file in csv_files:
        with open(r'{}/{}'.format(root_path, csv_file), 'r') as f:
            csv_reader = csv.reader(f)
            # n Dtl
            data_list_Dtl_3 = []

            for i, row in enumerate(csv_reader):
                data_list_Dtl_3.append(float(row[0]))

        data_list_Dtl_3.sort(reverse=True)
        # print(data_list_Dtl_3)

        std_3 = np.std(data_list_Dtl_3[:n_data_mean], ddof=1)* 100

        result = '{:.3}±{:.1f}'.format(np.mean(data_list_Dtl_3[:n_data_mean]) * 100,
                  std_3,)
        # result = '{:.3}±{:.1f}'.format(np.mean(data_list_Dtl_0[:])* 100, std_0)  # TAT
        # result = '{:.3}±{:.1f}'.format(np.mean(data_list_Dtl_3[:])* 100, std_3)  # Ours_2
        all_result.append(result)

        print('{}'.format(csv_file[:-3]), end=' ')
    print('\n')
    for result in all_result:
        print(result, end='\t')


def get_mean_standard_deviation_SSDA(root_path, n_data_mean=14):
    csv_files = os.listdir(root_path)
    all_result = []

    for csv_file in csv_files:
        with open(r'{}/{}'.format(root_path, csv_file), 'r') as f:
            csv_reader = csv.reader(f)
            # n Dtl
            data_list_Dtl_3 = []

            for i, row in enumerate(csv_reader):
                data_list_Dtl_3.append(float(row[0]))

        data_list_Dtl_3.sort(reverse=True)

        std_3 = np.std(data_list_Dtl_3[:n_data_mean], ddof=1)* 100

        result = '{:.3}±{:.1f}'.format(np.mean(data_list_Dtl_3[:n_data_mean]) * 100,
                  std_3)
        all_result.append(result)

        print('{}'.format(csv_file[:3]), end=' ')
    print('\n')
    for result in all_result:
        print(result, end='\t')


def get_mean_clustering_train(root_path, n_data_mean=10, all_data=12):

    # 第一行是参数情况说明
    # 第二行是

    csv_files = os.listdir(root_path)
    pattern = re.compile(r'\d\|\|\d')
    pre_train_result = []
    domain_name_list = []

    for csv_file in csv_files: # 一个文件的
        domain_name_list.append(csv_file[:5])
        all_result = []
        all_std = []
        with open(r'{}/{}'.format(root_path, csv_file), 'r') as f:

            csv_reader = list(csv.reader(f))
            # print(' {} The M0 Accuracy: {:.3f}'.format(csv_file[:5], float(csv_reader[1][0])), end=' ')

            for i in range(len(csv_reader) // all_data):  # 一个文件
                i = all_data*i
                data_list = csv_reader[i+2 : i+2+n_data_mean]
                data_list = [float(data[0]) for data in data_list]
                # get clusters
                information = csv_reader[i][0]
                clusters = pattern.findall(information)[0]
                std = np.std(data_list[:n_data_mean], ddof=1) * 100
                # all_result.append([np.mean(data_list), clusters])
                result = np.mean(data_list) * 100
                all_result.append(result)
                all_std.append(std)
            pre_train_result.append(csv_reader[1][0])
            # print(pre_train_result)

        # all_result.sort(key=lambda x: x[0], reverse=True)
        # for i in all_result:
        #     print('{:.3}({})'.format(i[0], i[1]))
        max_index = np.argmax(all_result)
        print('{:.1f}±{:.1f}'.format(all_result[int(max_index)], all_std[int(max_index)]), end='\t')

    print('\n')
    for i in domain_name_list:
        print(i, end='\t')

    print('\n')
    for i in pre_train_result:
        print('{:.1f}'.format(float(i) * 100), end='\t')


        #     for i, row in enumerate(csv_reader):
        #         if i < 0:
        #             data_list_Dtl0.append(float(row[0]))
        #         elif i > 5:
        #             data_list_Dtl3.append(float(row[0]))
        #
        # data_list_Dtl3.sort(reverse=True)
        # # print('{}: {:.3}({:.3})'.format(csv_file[:3], np.mean(data_list_Dtl3[:n_data_mean]), np.mean(data_list_Dtl0)))
        # print('{}:{:.3}'.format(csv_file[:3],  np.mean(data_list_Dtl3[:n_data_mean])))


def get_mean_clustering_train_plot(root_path, n_data_mean=10):

    # 第一行是参数情况说明
    # 第二行是

    csv_files = os.listdir(root_path)
    pattern = re.compile(r'\d\|\|\d')

    for csv_file in csv_files: # 一个文件的
        all_result = []
        with open(r'{}/{}'.format(root_path, csv_file), 'r') as f:
            csv_reader = list(csv.reader(f))

            print(' {} The M0 Accuracy: {:.3f}'.format(csv_file[:5], float(csv_reader[1][0])), end=' ')

            for i in range(len(csv_reader) // 12):  # 一个文件
                i = 12*i
                data_list = csv_reader[i+2 : i+2+n_data_mean]
                data_list = [float(data[0]) for data in data_list]
                # get clusters
                information = csv_reader[i][0]
                clusters = pattern.findall(information)[0]
                all_result.append([np.mean(data_list), clusters])
        # all_result.sort(key=lambda x: x[0], reverse=True)
        for i in all_result:
            print('{:.3}({})'.format(i[0], i[1]))

        #     for i, row in enumerate(csv_reader):
        #         if i < 0:
        #             data_list_Dtl0.append(float(row[0]))
        #         elif i > 5:
        #             data_list_Dtl3.append(float(row[0]))
        #
        # data_list_Dtl3.sort(reverse=True)
        # # print('{}: {:.3}({:.3})'.format(csv_file[:3], np.mean(data_list_Dtl3[:n_data_mean]), np.mean(data_list_Dtl0)))
        # print('{}:{:.3}'.format(csv_file[:3],  np.mean(data_list_Dtl3[:n_data_mean])))

if __name__ == '__main__':
    get_mean_standard_deviation_no_0dtl(r'E:\cht_project\Experimental_Result\DAN\Image_CLEF_Resnet50\MMD_experiment')
    # get_mean_standard_deviation_SSDA(r'E:\cht_project\Experimental_Result\ER\VisDA_Resnet50\SSDA')
    # get_mean_clustering_train(
    # r'E:\cht_project\Experimental_Result\ER\Office_Home_Resnet50\Clustering_Train\greedy\3.8',
    # n_data_mean=5, all_data=7)
    # get_mean_standard_deviation(r'E:\cht_project\Experimental_Result\ER\VisDA_Resnet50\DAN')
    # get_mean_clustering_train_plot(r'E:\cht_project\Experimental_Result\ER\Office_Home_Resnet50\Clustering_Train\greedy\1.28')