"""
 author:jjk
 datetime:2019/9/21
 coding:utf-8
 project name:Pycharm_workstation
 Program function:
 
"""
import numpy as np
import  math
import  matplotlib.pyplot as plt
import  matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def f2(x,y):
    return  0.6 * (x + y) ** 2 - x * y # 二元函数原型

# 二元函数求导
def hx2(x,y):
    return 0.6 * 2 * (x + y) - y
def hy2(x,y):
    return 0.6 * 2 * (x + y) - x

# 使用梯度下降法求解
GD_X1 = []
GD_X2 = []
GD_Y = []
x1 = 4 # 起始位置
x2 = 4
alpha = 2.0 # 学习率

f_change = f2(x1,x2)
f_current = f_change

GD_X1.append(x1)
GD_X2.append(x2)
GD_Y.append(f_current)
iter_num = 0 # 统计循环次数

while f_change > 1e-10 and iter_num <100:
    iter_num += 1
    prex1 = x1
    prex2 = x2
    x1 = x1 - alpha * hx2(prex1,prex2)
    x2 = x2 - alpha * hy2(prex1,prex2)

    tmp = f2(x1,x2)
    f_change = np.abs(f_current-tmp)
    f_current = tmp
    GD_X1.append(x1) # 迭代
    GD_X2.append(x2)
    GD_Y.append(f_current)

print(u"最终结果为:(%.5f, %.5f, %.5f)" % (x1, x2, f_current))
print(u"迭代过程中X的取值，迭代次数:%d" % iter_num)
print(GD_X1)

# 构建数据
X1 = np.arange(-4,4.5,0.2)
X2 = np.arange(-4,4.5,0.2)
X1,X2 = np.meshgrid(X1,X2)
Y = np.array(list(map(lambda t: f2(t[0], t[1]), zip(X1.flatten(), X2.flatten()))))
Y.shape = X1.shape

# 画图
fig = plt.figure(facecolor='w')
ax = Axes3D(fig)
ax.plot_surface(X1,X2,Y,rstride=1,cstride=1,cmap=plt.cm.jet)
ax.plot(GD_X1,GD_X2,GD_Y,'bo--')

ax.set_title(u'函数$y=0.6 * (θ1 + θ2)^2 - θ1 * θ2$;\n学习率:%.3f; 最终解:(%.3f, %.3f, %.3f);迭代次数:%d' % (alpha, x1, x2, f_current, iter_num))
plt.show()


