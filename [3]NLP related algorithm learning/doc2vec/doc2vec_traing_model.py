"""
 author:jjk
 datetime:2019/9/24
 coding:utf-8
 project name:Pycharm_workstation
 Program function: doc2vec段向量模型训练
 
"""

import gensim.models as g
from gensim.corpora import WikiCorpus
import logging # 日志信息
from langconv import * # 繁简体转换

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
class TaggedWikiDocument(object):
    def __init__(self,wiki):
        self.wiki = wiki
        self.wiki.metadata = True
    def __iter__(self):
        import jieba
        for content,(page_id,title) in self.wiki.get_texts():
            yield g.doc2vec.LabeledSentence(words=[w for c in content for w in jieba.cut(Converter('zh-hans').convert(c))],tags=[title])


def my_function():
    zhwiki_name = 'zhwiki-latest-pages-articles.xml.bz2'
    wiki = WikiCorpus(zhwiki_name,lemmatize=False,dictionary={})
    documents = TaggedWikiDocument(wiki)
    # dm:训练模型的种类，一般默认为1，指的是使用DM模型，当dm等于其他值，使用DBOW模型训练词向量
    model = g.Doc2Vec(documents,dm=0,dbow_words=1,size=192,window=8,min_count=19,iter=5,workers=8)
    model.save('zhiwiki_news.doc2vec') # 保存

if __name__ == '__main__':
    my_function()















