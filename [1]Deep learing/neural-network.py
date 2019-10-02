#!/usr/bin/env python
# coding: utf-8

# In[18]:


import tensorflow as tf # 导入tensorflow模型
import os
import _pickle as cPickle
import numpy as np
 
CIFAR_DIR = "./cifar-10-batches-py"
print('路径文件：',os.listdir(CIFAR_DIR))


# In[85]:


# 为何使函数具有泛化能力
import tensorflow as tf # 导入tensorflow模型
import os
import _pickle as cPickle
import numpy as np
 
CIFAR_DIR = "./cifar-10-batches-py"
print('路径文件：',os.listdir(CIFAR_DIR))


def load_data(filename):
    """读取文件"""
    with open(filename,'rb') as f:
        data = cPickle.load(f,encoding='bytes')
        #print(type(data))
        return data[b'data'],data[b'labels'] # 每个图片的像素值和labels值

class CifarData:
    def __init__(self,filenames,need_shuffle): # need_shuffle:使得数据之间没有相互关系，泛化能力特别强
        all_data = []
        all_labels = []
        for filename in filenames:
            data,labels = load_data(filename)
            #all_data.append(data)
            #all_labels.append(labels)
            # 0,1类别
            for item,label in zip(data,labels):# 将两个数据绑在一起
                if label in [0,1]: # 如果在0-1之间的话 50000个样本，但是有10个类别，只挑了2个
                    all_data.append(item)
                    all_labels.append(label)
                    
        self._data = np.vstack(all_data) # 将最好的值合并，再转化为一个np的一个矩阵，data是item的一个矩阵，item就是np的中向量，使用vstack给纵向合并成一个矩阵
        """这里是直接将数据值拿过来直接用了，但是一般来说，针对图像来说，会习惯性的将图片缩放到-1到1之间，因此，这里做一个缩放"""
        self._data = self._data / 127.5-1 # 这样数据集是0-255的，除过来是0-2之间的一个数，然后在减去1就是一个-1到1的一个数
        self._labels = np.hstack(all_labels)
        
        # 测试一下
        print(self._data.shape) # 输出应该是50000张，但是只采用了10000张
        print(self._labels.shape)
        
        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0 # 起始位置
        if self._need_shuffle:
            self._shuffle_data()
            
    def _shuffle_data(self):
        # [0，1，2，3，4,5] -> [5,3,2,4,0,1]
        p = np.random.permutation(self._num_examples) # 0-n个数做一个混排
        self._data = self._data[p] # 第五个放在第一个位置
        self._labels = self._labels[p]
        
    def next_batch(self,batch_size):
        """返回这三个的样本"""
        end_indicator = self._indicator + batch_size # 结束位置
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0 # 重置起始位置
                end_indicator = batch_size
            else:
                raise Exception("have no more examples")
        if end_indicator > self._num_examples:
            raise Exception("batch size is larger than all examples")
        batch_data = self._data[self._indicator:end_indicator]
        batch_labels = self._labels[self._indicator:end_indicator]
        self._indicator = end_indicator
        return batch_data,batch_labels

train_filenames=[os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(1, 6)]
test_filenames = [os.path.join(CIFAR_DIR, 'test_batch')]

train_data = CifarData(train_filenames, True)
test_data = CifarData(test_filenames, False)

batch_data,batch_labels = train_data.next_batch(10) # next_batch大小，自己选
# print(batch_data)
# print(batch_labels) # 训练集，测试集郑只要1和0，确认cifar是可以正常工作的


# In[76]:


tf.reset_default_graph()

# 定义x和y的占位符来作为将要输入的神经网络变量 
x = tf.placeholder(tf.float32,[None,3072]) 
# [None]
y = tf.placeholder(tf.int64,[None]) # None样本数目是不确定的
# (3072,1)
w = tf.get_variable('w',[x.get_shape()[-1],1],
                    initializer=tf.random_normal_initializer(0,1)) # 输出：1， initializer:如何初始化w-正态分布
b = tf.get_variable('b',[1],
                    initializer=tf.constant_initializer(0.0)) # 偏置——维度和第二位是一样的，w的第二位输出是1位，所以b的第二位就是一维，b=0.0初始化为0

# [None,3072] * [3072,1] = [None,1]    x*w的意思
y_ = tf.matmul(x,w) + b # 矩阵乘法+b；y_:每一个样值和w做内积 + b
# [None,1]
p_y_1 = tf.nn.sigmoid(y_)# y_变成一个概率值

# [None,1]
y_reshaped = tf.reshape(y,(-1,1)) # 先对y做一个reshape，
print(type(y_reshaped))
print(y_reshaped)

y_reshaped_float = tf.cast(y_reshaped,tf.float32) # 因为x是32的，在tensorflow中对于类型是特别敏感的，所以要进行以下类型转化
loss = tf.reduce_mean(tf.square(y_reshaped_float - p_y_1)) # 真实值和预测值差的平方，然后再去求均值，得到结果就是loss ; 平方差-计算损失函数，


# bool
predict = p_y_1 > 0.5 # 大于0.5我就预测为True,小于0.5就是False
correct_prediction = tf.equal(tf.cast(predict,tf.int64),y_reshaped)# 预测正确的预测值

# 计算准确率-准确率是一个平均
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) # 先格式转换

# 梯度下降的方法
with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss) # AdamOptimizer是梯度下降的一个变状，梯度下降是在loss上面做
# 以上计算图就完成了


# In[86]:


# 二分类逻辑斯蒂回归
# 初始化
init = tf.global_variables_initializer() # 执行初始化
batch_size = 20
train_steps = 10000 # 步骤，也就是运行1000次
test_steps = 100 # 100*20=2000张图片



with tf.Session() as sess: # 是一个会话的界面，执行这个计算图
    sess.run(init)
    for i in range(train_steps):
        batch_data,batch_labels = train_data.next_batch(batch_size)
        loss_val,acc_val, _= sess.run(
            [loss,accuracy,train_op],
            feed_dict={
                x: batch_data,
                y: batch_labels})  #塞入输入  feed_dict={x:,y:} 图片数据和label数据
        
        if (i+1) % 500 ==0:# 中间过程
            print('[Train Step: %d, loss: %4.5f, acc: %4.5f]' % (i+1,loss_val,acc_val)) # 每500次打印一次，执行了1000次
        """
        虽然我们打印了训练的过程，但是我们为了去评估真是的一个标准，还需要在测试集上做评测，
        所以，此时我们还需要去评测测试集
        """
        if (i+1) % 5000 ==0:
            """每5000次去运行一次测试代码"""
            all_test_acc_val = [] # 对总结test结果做一个平均
            test_data = CifarData(test_filenames,False)
            for j in range(test_steps): # 对test_steps中的每一个值都去做预测
                test_batch_data,test_batch_labels = test_data.next_batch(batch_size)
                test_acc_val = sess.run([accuracy],feed_dict={x: test_batch_data, y: test_batch_labels})
                all_test_acc_val.append(test_acc_val) # 将获取的test_acc_val添加到all_test_acc_val中
            test_acc = np.mean(all_test_acc_val) # 调用一个函数，做平均
            print('[Test ] Step: %d, acc: %4.5f' % (i+1,test_acc))
             
                
"""
1、以上的所有的代码就是：单个神经元的一个逻辑斯蒂回归模型，在二分类上的一个问题的一个解。
2、tensorflow.Dataset可以更好进行数据处理

"""


# In[ ]:





# In[ ]:




