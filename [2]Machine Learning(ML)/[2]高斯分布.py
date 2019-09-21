"""
 author:jjk
 datetime:2019/5/2
 coding:utf-8
 project name:Pycharm_workstation
 Program function:
 
"""
import numpy as np
import matplotlib.pyplot as plt

# 均值
def average(data):
    return np.sum(data)/len(data)
# 标准差
def sigma(data,avg):
    sigma_squ = np.sum(np.power((data-avg),2))/len(data)
    return np.power(sigma_squ,0.5) # 数组元素求n次方

# 高斯概率分布-具体参考一维高斯分布的概率密度公式
def prob(data,avg,sig):
    sqrt_2pi = np.power(2*np.pi,0.5)# 乘pi开根号
    coef = 1/(sqrt_2pi*sig)
    powcoef = -1/(2*np.power(sig,2))# sig表示分子
    mypow = powcoef*(np.power((data-avg),2))# 数据减去均值
    return coef*(np.exp(mypow)) # np.exp(mypow):e的次方那部分


# 样本数据
data = np.array([ 0.79,  0.78,  0.8 ,  0.79,  0.77,  0.81,  0.74,  0.85,  0.8 ,
        0.77,  0.81,  0.85,  0.85,  0.83,  0.83,  0.8 ,  0.83,  0.71,
        0.76,  0.8 ])
# 根据样本求高斯分布的平均数
ave = average(data)
# 根据样本求高斯分布的标准差
sig = sigma(data,ave)
# 获取数据
x = np.arange(0.5,1.0,0.01)
p = prob(x,ave,sig)

# 绘制
plt.plot(x,p)
plt.grid()

plt.xlabel('apple quality factor')
plt.ylabel('prob density')
plt.yticks(np.arange(0,12,1)) # y轴长度以及间隔
plt.title('Gaussian distribution')

plt.show()

