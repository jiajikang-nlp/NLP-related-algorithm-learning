#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf # 导入tensorflow模型
import os
import _pickle as cPickle
import numpy as np
# 卷积神经网络-图像分类

CIFAR_DIR = "./cifar-10-batches-py"
print('路径文件：',os.listdir(CIFAR_DIR))


# In[2]:


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
            data,labels = load_data(filename) # """为了实现多分类的逻辑斯蒂回归，现在已经是10个类的数据"""
            all_data.append(data)
            all_labels.append(labels)
            # 0,1类别
            #for item,label in zip(data,labels):# 将两个数据绑在一起
            #    if label in [0,1]: # 如果在0-1之间的话 50000个样本，但是有10个类别，只挑了2个
            #        all_data.append(item)
            #        all_labels.append(label)
        self._data = np.vstack(all_data) # 将最好的值合并，再转化为一个np的一个矩阵，data是item的一个矩阵，item就是np的中向量，使用vstack给纵向合并成一个矩阵
        """这里是直接将数据值拿过来直接用了，但是一般来说，针对图像来说，会习惯性的将图片缩放到-1到1之间，因此，这里做一个缩放"""
        #self._data = self._data / 127.5-1 # 这样数据集是0-255的，除过来是0-2之间的一个数，然后在减去1就是一个-1到1的一个数
        self._labels = np.hstack(all_labels)
        
        # 测试一下
        print(self._data.shape) # 输出应该是50000张样本的数据集，test是10000个样本的数据集
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

# batch_data,batch_labels = train_data.next_batch(10) # next_batch大小，自己选
# print(batch_data)
# print(batch_labels) # 训练集，测试集郑只要1和0，确认cifar是可以正常工作的





# In[ ]:





# In[3]:


tf.reset_default_graph()

# 定义x和y的占位符来作为将要输入的神经网络变量 
x = tf.placeholder(tf.float32,[None,3072]) 
y = tf.placeholder(tf.int64,[None]) # None样本数目是不确定的

"""
CNN:输入是图片，所以将x展开的一维向量变成，具有一个三通道的一个图片格式
所以，这里要做一个转换

"""
x_image = tf.reshape(x,[-1,3,32,32])
# 图片大小是：32*32
x_image = tf.transpose(x_image,perm=[0,2,3,1])# perm:通道的一个排列顺序

# 这里将会用卷积神经网络CNN来实现
# 卷积层
# 输入到卷积神经中去
# conv1：神经元图，feature_map,输出图像
conv1_1 = tf.layers.conv2d(x_image,32,(3,3),padding='same',activation=tf.nn.relu,name='conv1_1')# 32:输出通道的数目;(3,3):kernel size卷积核的大小;activation:激活函数
# conv1_1后面后添加了一个卷积层conv1_2
conv1_2 = tf.layers.conv2d(conv1_1,32,(3,3),padding='same',activation=tf.nn.relu,name='conv1_2')# 32:输出通道的数目;(3,3):kernel size卷积核的大小;activation:激活函数


# 池化层
# 16*16
pooling1 = tf.layers.max_pooling2d(conv1_2,
                                  (2, 2), # kernel size
                                  (2, 2), # stride size 步长
                                  name = 'pooll')


# 第二个：卷积层
# conv2：神经元图，feature_map,输出图像
conv2_1 = tf.layers.conv2d(pooling1,32,(3,3),padding='same',activation=tf.nn.relu,name='conv2_1')# 32:输出通道的数目;(3,3):kernel size卷积核的大小;activation:激活函数
conv2_2 = tf.layers.conv2d(conv2_1,32,(3,3),padding='same',activation=tf.nn.relu,name='conv2_2')# 32:输出通道的数目;(3,3):kernel size卷积核的大小;activation:激活函数

# 第二个：池化层
# 8*8
pooling2 = tf.layers.max_pooling2d(conv2_2,
                                  (2, 2), # kernel size
                                  (2, 2), # stride size 步长
                                  name = 'pool2')

# 第三个：卷积层
# conv3：神经元图，feature_map,输出图像
conv3_1 = tf.layers.conv2d(pooling2,32,(3,3),padding='same',activation=tf.nn.relu,name='conv3_1')# 32:输出通道的数目;(3,3):kernel size卷积核的大小;activation:激活函数
conv3_2 = tf.layers.conv2d(conv3_1,32,(3,3),padding='same',activation=tf.nn.relu,name='conv3_2')# 32:输出通道的数目;(3,3):kernel size卷积核的大小;activation:激活函数

# 第三个：池化层
# 4*4
pooling3 = tf.layers.max_pooling2d(conv3_2,
                                  (2, 2), # kernel size
                                  (2, 2), # stride size 步长
                                  name = 'pool3')



# 以上是：三个卷积层，三个池化层,其中三个卷积层后面分别又添加了一个卷积层，从而达到VGGNet的思想
# 就会得到一个一维向量：每一个样本是一个向量，但是它还是二维的矩阵，因为还有不同的样本
# [None,4*4*32]
# 因为还是一个4*4*32的一个多维矩阵，所以就先展平
flatten = tf.layers.flatten(pooling3)

# 加完卷积层和池化层之后，我们来加全连接层
# 展平之后，就加入全连接层，
y_ = tf.layers.dense(flatten,10)

# 然后就可以计算它的损失函数
# 交叉熵损失函数比平方差损失函数实现起来更简单一些 
loss = tf.losses.sparse_softmax_cross_entropy(labels=y,logits=y_) # labels=y:图片真实的类别值；logits=y_：图片经过计算得到的内积值
# y_ -> sofmax
# y  -> one_hot
# loss = ylogy_


# indices
predict = tf.argmax(y_,1) 
# 求正确的预测值，predict已经是一个预测值了，所以如下所示，就不需要在变化她的类型了
correct_prediction = tf.equal(predict,y)# 预测正确的预测值
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float64)) # 将correct_prdiction做均值


# 梯度下降的方法
with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss) # AdamOptimizer是梯度下降的一个变状，梯度下降是在loss上面做
# 以上计算图就完成了


# In[4]:


# 二分类逻辑斯蒂回归
# 初始化
init = tf.global_variables_initializer() # 执行初始化
batch_size = 20
train_steps = 1000 # 步骤，也就是运行1000次
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
        
        if (i+1) % 100 ==0:# 中间过程
            print('[Train Step: %d, loss: %4.5f, acc: %4.5f]' % (i+1,loss_val,acc_val)) # 每500次打印一次，执行了1000次
        """
        虽然我们打印了训练的过程，但是我们为了去评估真是的一个标准，还需要在测试集上做评测，
        所以，此时我们还需要去评测测试集
        """
        if (i+1) % 1000 ==0:
            """每5000次去运行一次测试代码"""
            all_test_acc_val = [] # 对总结test结果做一个平均
            test_data = CifarData(test_filenames,False)
            for j in range(test_steps): # 对test_steps中的每一个值都去做预测
                test_batch_data,test_batch_labels = test_data.next_batch(batch_size)
                test_acc_val = sess.run([accuracy],feed_dict={x: test_batch_data, y: test_batch_labels})
                all_test_acc_val.append(test_acc_val) # 将获取的test_acc_val添加到all_test_acc_val中
            test_acc = np.mean(all_test_acc_val) # 调用一个函数，做平均
            print('[Test ] Step: %d, acc: %4.5f' % (i+1,test_acc))


# In[ ]:




