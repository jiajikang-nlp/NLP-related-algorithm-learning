
# 使用gensim模块训练词向量：
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging # 记录日志
try:
    import cPickle as pickle
except ImportError as e:
    import pickle

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) # 基本信息

def my_function():
    wiki_news = open('./yuliao/reduce_zhiwiki.txt','r',encoding='utf-8') # 打开中文语料文件
    # sg=表示使用CBOW模型训练词向量；sg=1表示利用Skip-gram训练词向量。
    # 参数size表示词向量的维度。
    # windows表示当前词和预测词可能的最大距离，其中windows越大所需要枚举的预测此越多，计算时间就越长。
    # min_count表示最小出现的次数。
    # workers表示训练词向量时使用的线程数。
    sentence = LineSentence(wiki_news) # 为要训练的txt的路径
    model = Word2Vec(sentence,sg=0,size=192,window=5,min_count=5,workers=9)
    print('训练~~~')
    model.save('./yuliao/zhiwiki_news.pkl')# model_path为模型路径。保存模型，通常采用pkl形式保存，以便下次直接加载即可
    model.save('./yuliao/zhiwiki_news.word2vec')
    # model.wv.save_word2vec_format(embedding_path,binary=True) # 二进制保存


# 走你
if __name__ == '__main__':
    my_function()
    print('训练结束')

