# """
#  author:jjk
#  datetime:2019/5/3
#  coding:utf-8
#  project name:Pycharm_workstation
#  Program function: 正向最大匹配( Maximum Match Method , MM 法)分词
# """


"""
 S1、导入分词词典input.txt，存储为字典形式dic、导入停用词词典stop_words.utf8 ，存储为字典形式stoplis、需要分词的文本文件 fenci.txt，存储为字符串chars
      S2、遍历分词词典，找出最长的词，其长度为此算法中的最大分词长度max_chars 
      S3、创建空列表words存储分词结果
      S4、初始化字符串chars的分词起点n=0
      S5、判断分词点n是否在字符串chars内，即n < len(chars)  如果成立，则进入下一步骤，否则进入S9
      S6、根据分词长度i（初始值为max_chars）截取相应的需分词文本chars的字符串s 
      S7、判断s是否存在于分词词典中，若存在，则分两种情况讨论，一是s是停用词,那么直接删除，分词起点n后移i位，转到步骤5；
          二是s不是停用词，那么直接添加到分词结果words中，分词起点n后移i位，
        转到步骤5；若不存在，则分两种情况讨论，一是s是停用词，那么直接删除，分词起点后移i位，
        转到步骤5；二是s不是停用词，分词长度i>1时，分词长度i减少1，
        转到步骤6 ,若是此时s是单字，则转入步骤8；
      S8、将s添加到分词结果words中，分词起点n后移1位，转到步骤5
      S9、将需分词文本chars的分词结果words输出到文本文件result.txt中
"""
import codecs

#分词字典
f1 = codecs.open('input.txt', 'r', encoding='utf8')
dic = {}
while 1:
    line = f1.readline()
    if len(line) == 0:
        break
    term = line.strip() #去除字典两侧的换行符，避免最大分词长度出错
    dic[term] = 1
f1.close()

#获得需要分词的文本
f2 = codecs.open('fenci.txt', 'r', encoding='utf8')
chars = f2.read().strip()
f2.close()

#停用词典，存储为字典形式
f3 = codecs.open('stop_words.utf8', 'r', encoding='utf8')
stoplist = {}
while 1:
    line = f3.readline()
    if len(line) == 0:
        break
    term = line.strip()
    stoplist[term] = 1
f3.close()

"""
正向匹配最大分词算法
遍历分词词典，获得最大分词长度
"""
max_chars = 0
for key in dic:
    if len(key) > max_chars:
        max_chars = len(key)

#定义一个空列表来存储分词结果
words = []
n = 0
while n < len(chars):
    matched = 0
    #range([start,] stop[, step])，根据start与stop指定的范围以及step设定的步长 step=-1表示去掉最后一位
    for i in range(max_chars, 0, -1): #i等于max_chars到1
        s = chars[n : n + i] #截取文本字符串n到n+1位
        #判断所截取字符串是否在分词词典和停用词词典内
        if s in dic:
            if s in stoplist: #判断是否为停用词
                words.append(s)
                matched = 1
                n = n + i
                break
            else:
                words.append(s)
                matched = 1
                n = n + i
                break
        if s in stoplist:
            words.append(s)
            matched = 1
            n = n + i
            break
    if not matched: #等于 if matched == 0
        words.append(chars[n])
        n = n + 1
#分词结果写入文件
f3 = open('MMresult.txt','w', encoding='utf8') # 输出结果写入到MMresult.txt中
f3.write('/'.join('%s' %id for id in words))
print('/'.join('%s' %id for id in words)) # 打印到控制台
f3.close() # 关闭文件指针
