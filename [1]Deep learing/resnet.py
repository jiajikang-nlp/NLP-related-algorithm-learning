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





# In[3]:


tf.reset_default_graph()

"""
因为在实现ResNet网络过程中，要经过很多个残差连接块，所以将残差连接块抽象出来，形成一个函数
resNet的计算图：
"""
def residual_block(x,output_channel):# x:输入，output_chanel：输出的通道数，ResNet每经过一个降采样过程，就会使他输出的channel数翻倍。这样就避免因为降采样带来的信息损失
    """residual_block conection implementation"""
    input_channel = x.get_shape().as_list()[-1]# x的最后一维
    if input_channel * 2 == output_channel:
        increase_dim = True
        strides = (2,2) # 需要降采样过程
    elif input_channel == output_channel:
        increase_dim = False
        strides = (1,1) # 不增加通道的数目，不需要进行降采样，所以将卷积层的步长都设置为1
    else:# 如果是其他的情况，就抛出一个异常
        raise Exception("input channer can't match output channel")
    # 然后就经过两个卷积层
    
    # 第一个分支：经过两个卷积层
    # 第二个分支：恒等变换：如果有降采样的话，就需要做一个max pool
    convl = tf.layers.conv2d(x,
                             output_channel,
                             (3,3),
                             strides = strides,
                             padding = 'same',
                             activation = tf.nn.relu,
                             name = 'convl') # 使用(3,3)的卷积核，stridex:步长就是上面获取的变量，padding：如果步长是(1,1)就是一样大小
    conv2 = tf.layers.conv2d(convl,
                             output_channel,
                             (3,3),
                             strides = (1,1),
                             padding = 'same',
                             activation = tf.nn.relu,
                             name = 'conv2') # stridex:如果是降下来的话，已经是降下来的那个值，就不需要再次做降采样了，
    
    # 处理第二个分支
    if increase_dim:
        #pooled_x是一个四通道的，[None,image_width,image_height,channel] - >[,,,channel*2]  最后一个是channel数目：
        pooled_x = tf.layers.average_pooling2d(x,
                                               (2,2),# 步长和pool都是(2,2)，使得图像变为原来的一半
                                               (2,2),
                                               padding = 'valid')# 因为图像是32*32的，所以都可以除的尽的。
        # 这里的x有一个问题，pooling不会增加通道数目，在上面我的输出通道数有可能通过降采样，增加到2倍。然后经过conv1和conv2会翻倍。
        # 所以说，在这里，图的大小是一样的，但是，最后一个维度：通道数目不能减少，所以这里做一个padding，这个padding在通道上。
        # 就会使得padded_x和conv2是同等大小的。
        padded_x = tf.pad(pooled_x,
                          [[0,0],
                           [0,0],
                           [0,0],# 第1，2，3维度不需要做补充
                           [input_channel//2,input_channel//2]])# output_channel是input_chanel的2倍，所以说差一个input_channel；也就是2倍的input_channel-1倍的input_channel
    else:
        padded_x = x # 等于x的本身
    output_x = conv2 + padded_x # 输出：卷积后的结果 + 恒等变换的结果
    return output_x
    # 以上这个函数就是残差连接的基本函数 
    


# 使用残差连接块——搭建我们需要卷积神经网络
def res_net(x,# x:输入，
            num_residual_blocks, #每一层需要有多少个残差连接块
            # num_subsampling,  #
            num_filter_base,  # 最初的通道数目，也就是基数
            class_num): # 适应多种类别数目不同的数据集
    """residual network implementation"""
    """
    Args:
    - x:
    - num_residual_blocks:[3,4,6,3] 残差连接块的多少
    - num_subsampling：长度是4和num_residual_blocks相等的； 需要做多少次降采样;
    - num_filter_base: 
    - class_num:
    
    """
    num_subsampling = len(num_residual_blocks)
    # 用一个列表保存之前的层次,每次都在这个列表的最后一个，去找到最新的一个层次。
    layers = []
    # x:[None,width,height,chanel]--->[width,height,chanel]
    input_size = x.get_shape().as_list()[1:]
    # 输入层，先经过一个扑捉建立层
    with tf.variable_scope('conv0'):
        conv0 = tf.layers.conv2d(x,
                                 num_filter_base,
                                 (3,3),# 卷积核
                                 strides = (1,1), # 步长
                                 padding = 'same', # 卷积
                                 activation = tf.nn.relu, # 激活函数
                                 name = 'conv0')
        layers.append(conv0)
    # 残差块有多个sted，每个残差块有不同数目的残差块
    # num_subsampling = 4, sample_id = [0,1,2,3]
    # 整个残差网络我们使用一个for循环来实现
    for sample_id in range(num_subsampling):
        for i in range(num_residual_blocks[sample_id]):
            with tf.variable_scope("conv%d_%d" % (sample_id,i)):
                conv = residual_block(
                    layers[-1], # 输出就是layer中的最后一个
                    # 控制输出通道数目
                    num_filter_base * (2 ** sample_id)) # sample_id：从0到 num_subsampling-1
                # 将新的一层添加到layers中
                layers.append(conv)
    
    # 为了保证程序的正确性，我们预测一下最后输出的神经元的大小是什么，和实际的神经元大小是否是一致的
    # 做一个预测
    multiplier = 2 ** (num_subsampling - 1) # 图像大小降低了是：除；通道数目是乘以它
    assert layers[-1].get_shape().as_list()[1:] == [input_size[0] / multiplier,
                                                   input_size[1] / multiplier,
                                                   num_filter_base * multiplier]
    
    
    # 如果以上通过的话，接下来就是average pool
    # 然后再做一个全连接到类别上去。
    
    with tf.variable_scope('fc'):
        # layer[-1].shape : [None,width,height,channel]
        # global_pool使得一个神经元图从二维的图，变成一个像素点
        # global_pool和普通的pool一样的，只是kernal_size = 图的大小：image_width,image_height,因此一张图经过global_pool就变成了一个均值
        global_pool = tf.reduce_mean(layers[-1],[1,2]) # layers[-1]还是一个四通道的——>1,2维度做pool——>全连接
        logits = tf.layers.dense(global_pool,class_num)
        layers.append(logits)
    return layers[-1]



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


y_ = res_net(x_image,[2,3,2],32,10)


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
        if (i+1) % 100 ==0:
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




