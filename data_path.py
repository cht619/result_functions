# -*- coding: utf-8 -*-
# @Time : 2021/2/16 11:37
# @Author : CHT
# @Site : 
# @File : data_path.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Function:


# Image-CLEF
# Image_CLEF_root_path = r'/home/dell/Documents/Kmeans_Transer_learning/Dataset/imageCLEF_resnet50'
Image_CLEF_root_path = r'E:\cht_project\domain_adaptation_images\imageCLEF_resnet50'
domain_c = 'c_c.csv'
domain_i = 'i_i.csv'
domain_p = 'p_p.csv'
domain_ci = 'c_i.csv'
domain_cp = 'c_p.csv'
domain_ic = 'i_c.csv'
domain_ip = 'i_p.csv'
domain_pc = 'p_c.csv'
domain_pi = 'p_i.csv'

# Office-Caltech
Office_Caltech_root_path = r'E:\cht_project\domain_adaptation_images\decaf6'
domain_dslr = 'dslr_decaf'
domain_caltech = 'caltech_decaf'
domain_amazon = 'amazon_decaf'
domain_webcam = 'webcam_decaf'

# # Office-31
# Office_31_root_path = r'E:\cht_project\domain_adaptation_images\office31_resnet50'
# domain_amazon = 'amazon_amazon.csv'
# domain_dslr = 'dslr_dslr.csv'
# domain_


# Office-Home
Office_Home_root_path = r'E:\cht_project\domain_adaptation_images\Office-Home_resnet50'
domain_ar = 'Art_Art.csv'
domain_ar_cl = 'Art_Clipart.csv'
domain_ar_pr = 'Art_Product.csv'
domain_ar_rw = 'Art_RealWorld.csv'
domain_cl_ar = 'Clipart_Art.csv'
domain_cl = 'Clipart_Clipart.csv'
domain_cl_pr = 'Clipart_Product.csv'
domain_cl_rw =  'Clipart_RealWorld.csv'
domain_pr_ar =  'Product_Art.csv'
domain_pr_cl = 'Product_Clipart.csv'
domain_pr = 'Product_Product.csv'
domain_pr_rw =  'Product_RealWorld.csv'
domain_rw_ar = 'RealWorld_Art.csv'
domain_rw_cl = 'RealWorld_Clipart.csv'
domain_rw_pr = 'RealWorld_Product.csv'
domain_rw = 'RealWorld_RealWorld.csv'

# VisDA-2017
VisDA_root_path = r'E:\cht_project\domain_adaptation_images\VisDA_resnet50'
domain_train = 'train_train.csv'
domain_vali = 'train_validation.csv'

# Multi Domain Sentiment Dataset
MDS_root_path = r'E:\cht_project\domain_adaptation_images\Multi_Domain_Sentiment_Dataset\Amazon_review'
domain_B = 'books_400.mat'
domain_D = 'dvd_400.mat'
domain_K = 'kitchen_400.mat'
domain_E = 'elec_400.mat'


if __name__ == '__main__':
    import os
    print(os.listdir(Office_Home_root_path))