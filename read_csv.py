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


def get_mean(root_path, n_data_mean=14):
    csv_files = os.listdir(root_path)

    for csv_file in csv_files:
        with open(r'{}/{}'.format(root_path, csv_file), 'r') as f:
            csv_reader = csv.reader(f)
            data_list_Dtl0 = []
            data_list_Dtl3 = []

            for i, row in enumerate(csv_reader):
                if i < 0:
                    data_list_Dtl0.append(float(row[0]))
                elif i > 5:
                    data_list_Dtl3.append(float(row[0]))

        data_list_Dtl3.sort(reverse=True)
        # print('{}: {:.3}({:.3})'.format(csv_file[:3], np.mean(data_list_Dtl3[:n_data_mean]), np.mean(data_list_Dtl0)))
        print('{}:{:.3}'.format(csv_file[:3],  np.mean(data_list_Dtl3[:n_data_mean])))


def get_mean_clustering_train(root_path, n_data_mean=10):

    # 第一行是参数情况说明
    # 第二行是

    csv_files = os.listdir(root_path)
    pattern = re.compile(r'\d\|\|\d')

    for csv_file in csv_files:
        with open(r'{}/{}'.format(root_path, csv_file), 'r') as f:
            csv_reader = list(csv.reader(f))

            print(' {} The M0 Accuracy: {:.3f}'.format(csv_file[:3], float(csv_reader[1][0])))

            for i in range(len(csv_reader) // 12):
                i = 12*i
                data_list = csv_reader[i+2 : i+2+n_data_mean]
                data_list = [float(data[0]) for data in data_list]
                # get clusters
                information = csv_reader[i][0]
                clusters = pattern.findall(information)[0]
                print('{:.3}'.format(np.mean(data_list)))


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
    # get_mean(r'E:\cht_project\Experimental_Result\ER\Multi_Domain_Sentiment_Dataset\SSDA')
    get_mean_clustering_train(r'E:\cht_project\Experimental_Result\ER\Office_Home_Resnet50\Clustering_Train\5678')

    # with open(r'E:\cht_project\Experimental_Result\ER\Multi_Domain_Sentiment_Dataset\DAN\E_B.csv', 'r') as f:
    #     reader = csv.reader(f)
    #
    #     data_list0 = []
    #     data_list1 = []
    #
    #     for i, row in enumerate(reader):
    #         if i < 5:
    #             data_list0.append(float(row[0]))
    #         elif i > 5:
    #             data_list1.append(float(row[0]))
    #
    # data_list1.sort(reverse=True)
    # print(data_list1)
    # print(np.mean(data_list1[:14]))