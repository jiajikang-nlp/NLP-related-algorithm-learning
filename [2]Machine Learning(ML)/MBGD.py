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
m = len(x_train)
x0 = np.full((m,1),1)
# 构造一个每个数据第一维特征都是1的矩阵
input_data = np.hstack([x0,x_train])
m,n = input_data.shape
theta1 = np.array([[2,3,4]]).T
# 构建标签数据集，后面的np.random.randn是将数据加一点噪声，仪表模拟数据集
#y_train = (input_data.dot(np.array([1, 2, 3, 4]).T)).T
y_train = x_train.dot(theta1) + np.array([[2],[2],[2],[2],[2]])

# 设置两个终止条件
loop_max = 1000000
epsilon = 1e-5
# 初始theta
np.random.seed(0) # 设置随机种子
theta = np.random.randn(n,1) # 随机去一个1维列向量初始化theta

# 初始化步长/学习率
alpha = 0.00001
# 初始化误差，每个维度的theta都应该有一个误差，所以误差是4维
error = np.zeros((n,1)) # 列向量

# 初始化偏导数
diff = np.zeros((input_data.shape[1],1))
# 初始化循环次数
count = 0
# 设置小批量的样本数
minibatch_size = 2

while count < loop_max:
    count += 1
    sum_m = np.zeros((n,1))
    for i in range(1,m,minibatch_size):
        for j in range(i-1,i+minibatch_size-1,1):
            # 计算每个维度的theta
            diff[j] = (input_data[i].dot(theta)-y_train[i]) * input_data[i,j]
        # 求每个维度的梯度的累加和
        sum_m = sum_m + diff
    # 利用这个累加和更新梯度
    theta = theta - alpha*(1.0/minibatch_size)*sum_m
    # else中将前一个theta赋值给error，theta-error便表示前后两个梯度的变化，当梯度变化小(在接收的范围内)时，便停止迭代

    if np.linalg.norm(theta-error) < epsilon:
        break
    else:
        error = theta
print(theta) # 输出梯度：真实的应该是2234

