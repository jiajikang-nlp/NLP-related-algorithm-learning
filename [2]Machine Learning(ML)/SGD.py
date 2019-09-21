"""
 author:jjk
 datetime:2019/9/21
 coding:utf-8
 project name:Pycharm_workstation
 Program function:
 
"""

import numpy as np
# 构造训练数据集
x_train = np.array([[2, 0., 3], [3, 1., 3], [0, 2., 3], [4, 3., 2], [1, 4., 4]])
# 构建一个权重作为数据集的真正的权重，theta1主要是用来构建y_train，然后通过模型计算
# 拟合的theta，这样可以比较两者之间的差异，验证模型。
theta1 = np.array([[2,3,4]]).T

# 构建标签数据集,y=t1*x1+t2*x2+t3*x3+b即y=向量x_train乘向量theta+b, 这里b=2
y_train = (x_train.dot(theta1) + np.array([[2],[2],[2],[2],[2]])).ravel()

# 构建一个5行1列的单位矩阵x0，然它和x_train组合，形成[x0, x1, x2, x3]，x0=1的数据形式，
# 这样可以将y=t1*x1+t2*x2+t3*x3+b写为y=b*x0+t1*x1+t2*x2+t3*x3即y=向量x_train乘向
# 量theta其中theta应该为[b, *, * , *]，则要拟合的theta应该是[2,2,3,4]，这个值可以
# 和算出来的theta相比较，看模型的是否达到预期

x0 = np.ones((5,1))
input_data = np.hstack([x0,x_train])
m,n = input_data.shape

# 设置两个终止条件
loop_max = 10000000
epsilon = 1e-6
# 初始化theta（权重）
np.random.seed(0)
theta = np.random.rand(n).T # 随机生成10以内的，n维1列的矩阵
# 初始化步长/学习率
alpha = 0.000001
# 初始化迭代误差（用于计算梯度两次迭代的差）
error = np.zeros(n)

# 初始化偏导数矩阵
diff = np.zeros(n)
# 初始化循环次数
count = 0

while count<loop_max:
    count += 1 # 每运行一次count+1，以此来总共记录运行的次数
    # 计算梯度
    for i in  range(m):
        # 计算每个维度theta的梯度，并运算一个梯度更新它，也就是迭代啦
        diff = input_data[i].dot(theta)-y_train[i]
        theta = theta - alpha * diff*(input_data[i])
    # else中将前一个theta赋值给error,theta - error便表示前后两个梯度的变化，当梯度
    #变化很小（在接收的范围内）时，便停止迭代。
    if np.linalg.norm(theta-error) < epsilon:
        break
    else:
        error = theta
print(theta) # 理论上theta = [2,2,3,4]

