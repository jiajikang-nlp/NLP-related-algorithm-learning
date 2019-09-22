"""
 author:jjk
 datetime:2019/9/22
 coding:utf-8
 project name:Pycharm_workstation
 Program function: 中文语料处理

 维基百科提供的语料是xml格式的，因此需要将其转化为txt格式。
 由于维基百科中有很多繁体中文网页，所以也需要将这些繁体字转化为简体字。
 另外，再用语料库训练词向量之前需要对中文句子进行分词，这里我就循规蹈矩的用很成熟的jieba分词吧(关于分词可以浏览博主博客：分词1,，分词2)

"""

import jieba
from gensim.corpora import WikiCorpus
#from util.langconv import *
from langconv import *


def my_function():
    space = ' '
    i = 0
    l = []
    zhwiki_name = './yuliao/zhwiki-latest-pages-articles.xml.bz2'.encode('utf-8')# 中文语料
    f = open('./yuliao/reduce_zhiwiki.txt','w',encoding='utf-8') # 创建文件reduce_zhiwiki用来将原始语料xml格式转化为txt格式
    wiki = WikiCorpus(zhwiki_name,lemmatize=False,dictionary={})
    for text in wiki.get_texts():
        for temp_sentence in text:
            # zh-hans:将繁体转换成简体
            # zh-hant:将简体转换成繁体
            temp_sentence = Converter('zh-hans').convert(temp_sentence) # 每行转换成简体
            #temp_sentence.enconde('utf-8')
            seg_list = list(jieba.cut(temp_sentence)) # 转换过来的每行简体-然后结巴分词
            for temp_term in seg_list:
                temp_term.encode('utf-8')
                l.append(temp_term) # 结巴分词完毕之后添加到l中

        f.write(space.join(l) + '\n')
        l = []
        i = i+1
        if (i % 200 ==0):
            print('Saved' + str(i) + 'articles')
    f.close() # 关闭文件指针

if __name__ == '__main__':
    my_function()