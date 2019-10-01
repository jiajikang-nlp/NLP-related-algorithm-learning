#!/usr/bin/env python
# coding: utf-8

# In[38]:


import _pickle as cPickle
import os
import numpy as np

CIFAR_DIR = "./cifar-10-batches-py"
print(os.listdir(CIFAR_DIR))


# In[35]:


# f = open('./cifar-10-batches-py/data_batch_1','rb')
# data = cPickle.load(f,encoding='bytes')
# print(type(data)) # 查看数据类型
    
with open(os.path.join(CIFAR_DIR,"data_batch_1"),'rb') as f:
    data = cPickle.load(f,encoding='bytes')
    print(type(data)) # 查看数据类型
    print(data.keys()) # 查看字典的值
    print(type(data[b'batch_label'])) #各个数据类型
    print(type(data[b'labels']))
    print(type(data[b'data']))
    print(type(data[b'filenames']))
    
    print('data=',data[b'data'].shape) # 每一个文件中有10000条数据，每一个数据是3072个维度，将图片展开，图片：32*32  = 1024 *3 = 3072
    print(data[b'data'][0:2]) # 前两条数据是什么样子的
    print(data[b'labels'][0:2]) # 标签，因为是10类，所以分别代替第7类和第10类
    print(data[b'batch_label']) # 文件的含义
    print(data[b'filenames'][0:2]) # 前两个具体的文件名称
    
# RG GG BB 展开到一维，然后拼接在一起

    


# In[ ]:


image_arr = data[b'data'][100] # 第100张,因为是3通道，所以把3通道拆开，然后再把32*32的变成2为，调用reshape，最后对图片做反解析
image_arr = image_arr.reshape((3,32,32)) # 32，32，3
image_arr = image_arr.transpose((1,2,0)) # 通道的交换
# 显示图片
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
get_ipython().run_line_magic('matplotlib', 'inline')
imshow(image_arr)

