# -*- coding: utf-8 -*-
# @Time : 2021/2/20 10:33
# @Author : CHT
# @Site : 
# @File : plot2.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Function:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = [1,2,3,4]
x1 = [2, 4, 6, 8]
y= [6,7,8,9]
z = 5
ax.bar(x, y, zs=1, zdir='y', color='rgb', alpha=0.5)
ax.bar(x1, x, 4, zdir='y', color='orange', alpha=0.5)
plt.show()

print(help(plt.bar))