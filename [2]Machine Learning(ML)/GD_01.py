"""
 author:jjk
 datetime:2019/9/21
 coding:utf-8
 project name:Pycharm_workstation
 Program function: 一元二次函数-梯度下降变化：0.5, 1.5，2.0，2.5-变化率
"""

import numpy as np
import matplotlib as mpl # 画图
import matplotlib.pyplot as plt
import math # 数学公式
from mpl_toolkits.mplot3d import Axes3D
# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 一维原始图像
def f1(x):
    return  0.5 * (x-0.25) ** 2 # 1/2*(x-0.25)^2
# 导函数
def h1(x):
    return 0.5 * 2 * (x-0.25) # 原函数求导

# 使用梯度下降法求解
GD_X = []
GD_Y = []
x = 4  # 起始位置
alpha = 0.5  # 学习率

f_change = f1(x) # 调用原始函数
f_current = f_change

GD_X.append(x)
GD_Y.append(f_current)
iter_num = 0
# 变化量大于1e-10并且迭代次数小于100时执行循环体

while f_change >1e-10 and iter_num<100:
    iter_num += 1
    x = x - alpha * h1(x)
    tmp = f1(x)
    f_change = np.abs(f_current-tmp) # 变化量
    f_current = tmp # 此时的函数值
    GD_X.append(x)
    GD_Y.append(f_current)
print(u'最终结果为:(%.5f,%.5f)' % (x,f_current))
print(u"迭代过程中x的取值，迭代次数:%d" % iter_num)
print(GD_X)

# 构建数据
X = np.arange(-4, 4.5, 0.05)  # 随机生成-4到4.5，步长为0.05的数
Y = np.array(list(map(lambda t: f1(t), X)))  # X对应的函数值

# 画图
plt.figure(facecolor='w')
plt.plot(X,Y,'r-',linewidth=2) # 函数原图像
plt.plot(GD_X,GD_Y,'bo--',linewidth=2) # 梯度迭代图
plt.title(u'函数$y=0.5 * (θ - 0.25)^2$; \n学习率:%.3f; 最终解:(%.3f, %.3f);迭代次数:%d' % (alpha, x, f_current, iter_num))
plt.show()