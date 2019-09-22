import gensim
def my_function():

    model = gensim.models.Word2Vec.load('./yuliao/zhiwiki_news.word2vec')# 加载训练好的模型
    print(model.similarity('西红柿','番茄'))# 0. 
    print(model.similarity('西红柿','香蕉'))#  
    #print(model.similarity('人工智能','机器学习'))
    #print(model.similarity('滴滴','共享单车'))

    word = '中国'
    if word in model.wv.index2word:
        print(model.most_similar(word))# 如果word包含在模型中，则输出最相似的词

if __name__ == '__main__':
    my_function()# 调用函数