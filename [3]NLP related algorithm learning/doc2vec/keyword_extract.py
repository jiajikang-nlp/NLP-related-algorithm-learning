"""
 author:jjk
 datetime:2019/9/23
 coding:utf-8
 project name:Pycharm_workstation
 Program function: 关键字提取
 
"""

import jieba.posseg as pseg
from jieba import analyse

def keyword_extract(data,file_name): # 提取关键字
    tfidf = analyse.extract_tags
    keywords = tfidf(data)
    return keywords

def getKeywords(docpath,savepath):
    with open(docpath,'r',encoding='utf-8') as docf, open(savepath,'w',encoding='utf-8') as outf:
        for data in docf:
            data = data[:len(data)-1]
            keywords = keyword_extract(data,savepath)
            for word in keywords:
                outf.write(word+' ') # 写入
            outf.write('\n') # 换行

if __name__ == '__main__':
    getKeywords('../yuliao/P2.txt','../yuliao/P2_keywords.txt')

