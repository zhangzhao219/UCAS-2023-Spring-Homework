import numpy as np
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split           #划分训练/测试集
from sklearn.feature_extraction.text import CountVectorizer    #抽取特征
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
#读取并清洗数据
#因为几个文档的编码不大一样，所以兼容了三种编码模式，根据经验，这三种是经常会遇到的
def get_txt_data(txt_file):
    mostwords=[]
    try:
        file=open(txt_file,'r',encoding='utf-8')
        for line in file.readlines():
            curline=line.strip().split("\t")
            mostwords.append(curline)
    except:
        try:
            file=open(txt_file,'r',encoding='gb2312')
            for line in file.readlines():
                curline=line.strip().split("\t")
                mostwords.append(curline)
        except:
            try:
                file=open(txt_file,'r',encoding='gbk')
                for line in file.readlines():
                    curline=line.strip().split("\t")
                    mostwords.append(curline)
            except:
                ''   
    return mostwords

neg_doc=get_txt_data(r'D:\nltk_data\waimai_nlp\情感分析语料\waimai_neg.txt')
pos_doc=get_txt_data(r'D:\nltk_data\waimai_nlp\情感分析语料\waimai_pos.txt')

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)