#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
第一步：分词
第二步：词语转换成id：将词语转化成长度为m的一个向量\n",
           1、定义一个矩阵matrix:第一个维度：词的大小，第二个维度：embed_size————》 [|V|,embed_size]
           2、需要一个索引：词语A -> id(5)：这样就可以从matrix中知道这个词语对应的embed_size哪一行的数据
           3、将词表统计出来
第三步：label的也统计出来，让它也变成id，也就是实现一个label的id的转换；label ->id 每一个label用一个数字表示，这样才能使用softmax

"""

import jieba # 分词
import os # 文件路径
import sys

# 输入文件input_files
train_file = 'cnews.train.txt'
val_file = 'cnews.val.txt'
test_file = 'cnews.test.txt'
# 输出文件output_files
seg_train_file = 'cnews.train.seg.txt'
seg_val_file = 'cnews.val.seg.txt'
seg_test_file = 'cnews.test.seg.txt'

# 词语转换的一个映射文件
vocab_file = 'cnews.vocab.txt' # 生成词表文件
category_file = 'cnews.category.txt'


# In[2]:


with open(val_file,'r',encoding='utf-8') as f:
    lines = f.readlines()


# print(lines[0]) # 看看第一个长个啥样子
# 解析一下
#label, content = lines[0].encode("utf-8").decode('utf-8').strip('\r\n').split('\t') # 移除换行符号，以制表符进行切片；将utf-8转化成Unicode
label, content = lines[0].strip('\r\n').split('\t')
word_iter = jieba.cut(content)

print(content)
print('/ '.join(word_iter))




# In[4]:


# 开始预处理任务
def generate_seg__file(input_file,output_seg_file):
    """按行对input_file内容进行先分词，"""
    with open(input_file,'r',encoding='utf-8') as f:
        lines = f.readlines() # 全部加载
    with open(output_seg_file,'w',encoding='utf-8') as f:
        for line in lines:
            #label, content = line.encode('utf-8').decode('utf-8').strip('\r\n').split('\t') # 去除换行符
            label, content = line.strip('\r\n').split('\t')
            word_iter = jieba.cut(content)
            word_content = ''# 分词后的结果
            for word in word_iter:
                word = word.strip(' ')# 去除空格
                if word !='':
                    word_content += word + ' '
            out_line = '%s\t%s\n' % (label,word_content.strip(' '))
            f.write(out_line)
            
# 调用分词函数
generate_seg__file(train_file,seg_train_file)
generate_seg__file(val_file,seg_val_file)
generate_seg__file(test_file,seg_test_file)

        


# In[6]:


# 词表的构建
def generate_vacab_file(input_seg_file,output_vocab_file):# 输入分词后的文件，输出词表
    with open(input_seg_file,'r',encoding='utf-8') as f:
        lines = f.readlines()
    word_dict = {}
    # 词频信息
    for line in lines:
        label, content = line.strip('\r\n').split('\t')
        for word in content.split():# 所有词出现的词频统计
            word_dict.setdefault(word,0)
            word_dict[word] += 1
    # 逆排序
    # [(word,frequency),....] 一个列表，元组
    sorted_word_dict = sorted(word_dict.items(),key = lambda d:d[1],reverse=True)# key：按什么排序，第一个元素，也就是列表中的第二个值
    # 输出到输出文件
    with open(output_vocab_file,'w',encoding='utf-8') as f:
        f.write('<UNK>\t1000000\n')# 代表当我找不到这个词的时候，就返回自定义的这个UNK
        for item in sorted_word_dict:
            f.write('%s\t%d\n' % (item[0],item[1]))# 词语，制表符，词频 换行符

# 生成词表
generate_vacab_file(seg_train_file,vocab_file)    
print('finished~~~')


# In[7]:


# 生成laber信息-类别表的生成
def generate_category_dict(input_file,category_file):
    with open(input_file,'r',encoding='utf-8') as f:
        lines = f.readlines()
    category_dict = {}
    word_dict = {}
    for line in lines:
        label, content = line.strip('\r\n').split('\t') # 剔除，分割
        category_dict.setdefault(label,0)
        category_dict[label] += 1 # laber统计
    category_number = len(category_dict)
    with open(category_file,'w',encoding='utf-8') as f:
        for category in category_dict:
            line = '%s\n' % category
            print('%s\t%d' % (category,category_dict[category]))
            f.write(line) # 写入

generate_category_dict(train_file,category_file)
print('finished！！！')


# In[ ]:




