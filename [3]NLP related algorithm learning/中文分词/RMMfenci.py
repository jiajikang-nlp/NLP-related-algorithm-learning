# """
#  author:jjk
#  datetime:2019/5/3
#  coding:utf-8
#  project name:Pycharm_workstation
#  Program function: 逆向最大匹配(Reverse Maximum Match Method , RMM 法)分词
# """

class IMM(object):
    def __init__(self,dic_path):
        self.dictionary=set()
        self.maximum = 0
        # 读取字典
        with open(dic_path,'r',encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.dictionary.add(line)
            self.maximum = len(self.dictionary)

    def cut(self,text):
        # 用于存放切分出来的词
        result = []
        index = len(text)
        # 记录没有在词典中的词，可以用于发现新词
        no_word = ''
        while index>0:
            word = None
            # 从前往后匹配，以此实现最大匹配
            for first in range(index):
                if text[first:index] in self.dictionary:
                    word = text[first:index]
                    # 如果之前存放字典里面没有出现过的词
                    if no_word != '':
                        result.append(no_word[::-1])
                        no_word = ''
                    result.append(text[first:index])
                    index = first
                    break
            if word == None:
                index = index - 1
                no_word += text[index]
        return  result[::-1]


def main():
    text = '南京市长江大桥'
    tokenizer = IMM('imm_dic.utf8') # 调用类函数
    print(tokenizer.cut(text)) # 输出

if __name__ == '__main__':
    main()