# -*- coding: utf-8 -*-
# @Time : 2021/2/17 9:24
# @Author : CHT
# @Site : 
# @File : distance_functions.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Function:

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance


cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    # fix_sigma是我们的核函数-径向基函数的参数
    n_samples = int(source.shape[0]) + int(target.shape[0])
    total = torch.cat([source, target], dim=0)
    # 这里计算是十分巧妙的，根据每一个点扩展，然后整一组扩展，相减再求和就能得到结果。
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    #　sigma参数这里定义的非常简单，有点类似于服从均衡分布
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    # 得到不同的sigma之后，直接构成multi kernels
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    # 这里要注意两个batch大小不一样的情况
    batch_size = min(source.shape[0], target.shape[0], 100)
    kernels = guassian_kernel(source[:100], target[:100],
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / (float(batch_size) + 1e-9)


def MMD_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    # 传入是tensor
    # 内存不够，只能分块进行计算

    # batch_size = 100 # 每一次算100个，总共算200个点
    # transition_matrix = FloatTensor(batch_size, source.shape[-1]).fill_(0.0)
    # if source.shape[0] > target.shape[0]:
    #     for i in range(source % 100)

    batch_size = source.shape[0]
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
    return loss.item()


def KL_divergence(src, tgt):
    # 返回时两个分布之间的KL
    # 记住Q必须先求log，目的可能是为了输出稳定？
    # P 就直接softmax
    # target 指导 source
    #　这里source是一个样本 target 是一组样本 list

    # kl = 0
    # log_softmax_source = F.log_softmax(src, dim=-1)
    #
    #
    # for j in range(len(tgt_list)):
    #     softmax_target = F.softmax(tgt_list[j], dim=-1)
    #     kl += F.kl_div(log_softmax_source, softmax_target, reduction='mean').cpu().numpy()

    kl = 0
    kl_list = []
    for s in src:  # 取每一行样本
        log_softmax_source = F.log_softmax(torch.tensor(s).clone().detach(), dim=-1).clone().detach()
        for t in tgt:  # 取每一行样本
            softmax_target = F.softmax(torch.tensor(t).clone().detach(), dim=-1).clone().detach()
            kl += F.kl_div(log_softmax_source.clone().detach(), softmax_target.clone().detach(), reduction='mean').cpu().numpy()
        kl_list.append(kl / len(tgt))

    return np.mean(kl)


def EMD(src, tgt):
    # 返回时两个分布之间的EMD
    # 同样距离越短，越接近

    d = 0
    d_list = []
    for s in src: # 取每一行样本
        for t in tgt: # 取每一行样本
            d += wasserstein_distance(s, t)
        d_list.append(d / len(tgt))
    return np.mean(d_list)


def get_clusters_distance(fea_src, fea_tgt, distance_method):
    # 这里传入的就是整个数据了
    dis = 0
    if distance_method == 'MMD':
        dis = mmd_rbf_accelerate(FloatTensor(fea_src), FloatTensor(fea_tgt))
    elif distance_method  == 'KL':
        # 重新construct一个tensor更安全，稳定
        dis = KL_divergence(torch.tensor(fea_src), torch.tensor(fea_tgt))
    elif distance_method == 'EMD':
        dis = EMD(fea_src, fea_tgt)
    # print('clusters distance:{}'.format(dis))
    return dis



def clustering_distance_MMD(fea_src_list, fea_tgt_list):
    # 要使用分配算法
    distance_matrix = np.ones((len(fea_src_list), len(fea_tgt_list)))
    for i in range(len(fea_src_list)):
        for j in range(len(fea_tgt_list)):
            # 只能使用cpu进行计算
            mmd_distance = mmd_rbf_accelerate(torch.tensor(fea_src_list[i]), torch.tensor(fea_tgt_list[j]))
            # mmd_distance = distance.mmd_rbf_noaccelerate(torch.FloatTensor(fea_src_list[i]), torch.FloatTensor(fea_tgt_list[j]))
            # if mmd_distance < sigma
            distance_matrix[i, j] = mmd_distance

    # row_ind, col_ind = linear_sum_assignment(distance_matrix)
    # clusters_pairs = [[row, col] for (row, col) in zip(row_ind, col_ind)]
    return distance_matrix


def clustering_distance_EMD(fea_src_list, fea_tgt_list):
    distance_matrix = np.ones((len(fea_src_list), len(fea_tgt_list)))
    for i in range(len(fea_src_list)):
        for j in range(len(fea_tgt_list)):
            # 只能使用cpu进行计算，gpu内存不够
            emd_ = EMD(fea_src_list[i], fea_tgt_list[j])
            # if mmd_loss < sigma
            distance_matrix[i, j] = emd_

    # print(distance_matrix)
    # row_ind, col_ind = linear_sum_assignment(distance_matrix)
    # clusters_pairs = [[row, col] for (row, col) in zip(row_ind, col_ind)]
    return distance_matrix


def clustering_distance_KL(fea_src_list, fea_tgt_list):
    distance_matrix = np.ones((len(fea_src_list), len(fea_tgt_list)))
    for i in range(len(fea_src_list)):
        for j in range(len(fea_tgt_list)):
            # 只能使用cpu进行计算，gpu内存不够
            kl = KL_divergence(fea_src_list[i], fea_tgt_list[j])
            # if mmd_loss < sigma
            distance_matrix[i, j] = kl

    # print(distance_matrix)
    # row_ind, col_ind = linear_sum_assignment(distance_matrix)
    # clusters_pairs = [[row, col] for (row, col) in zip(row_ind, col_ind)]
    return distance_matrix


def get_distance_matrix(feas_src_list, feas_tgt_list, distance_method):

    d_matrix = 0

    if distance_method == 'MMD':
        d_matrix = clustering_distance_MMD(feas_src_list, feas_tgt_list)

    elif distance_method == 'KL':
        d_matrix = clustering_distance_KL(feas_src_list, feas_tgt_list)

    elif distance_method == 'EMD':
        d_matrix = clustering_distance_EMD(feas_src_list, feas_tgt_list)

    return d_matrix



if __name__ == '__main__':
    p = torch.tensor([0.4, 0.4, 0.2], dtype=torch.float32)
    softmax_p = torch.softmax(p, -1)
    log_softmax_p = torch.log(softmax_p)
    ls = F.log_softmax(p, -1)

    print(softmax_p)
    print(log_softmax_p)
    print(ls)

    distance = wasserstein_distance([0, 1, 5], [0, 1, 3])
    print(distance)



