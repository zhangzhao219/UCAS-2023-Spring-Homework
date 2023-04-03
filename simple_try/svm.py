##----------4.使用SVM模型进行训练----##
from sklearn import svm 
#使用TF-IDF提取特征，使用SVM训练，结果为0.83125
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from sklearn.tree import DecisionTreeRegressor

import jieba
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import mean_squared_error

def dataprocess(x):
    return ','.join(list(jieba.cut(x)))

data_waimai_train = pd.read_csv('./data/waimai/Train.csv')
data_waimai_val = pd.read_csv('./data/waimai/Val.csv')
data_waimai_test = pd.read_csv('./data/waimai/Test.csv')

x_train = data_waimai_train["text"]
x_train = x_train.map(dataprocess)
y_train = data_waimai_train["label"]

x_test = data_waimai_test["text"]
x_test = x_test.map(dataprocess)
y_test = data_waimai_test["label"]


vec=TfidfVectorizer(analyzer='word', ngram_range=(1,4), max_features=500)
tfidf_x_train=vec.fit_transform(x_train) #与上面一种TfidfTransformer
tfidf_x_test=vec.fit_transform(x_test)

classfier = svm.SVC(kernel='linear')
classfier.fit(vec.transform(x_train),y_train)
#print(classfier.score(vec.transform(x_test),y_test))

print ('训练集:',classfier.score(vec.transform(x_train), y_train))  # 精度
print ('测试集:',classfier.score(vec.transform(x_test), y_test))
'''训练集: 0.85234375
测试集: 0.83125'''
y_train_hat=classfier.predict(vec.transform(x_train))
print ('训练集准确率：',accuracy_score(y_train_hat,y_train))
print ('训练集召回率：',recall_score(y_train_hat,y_train))
print ('F1:',f1_score(y_train_hat,y_train))
print ('ROC值：',roc_auc_score(y_train_hat,y_train))
'''训练集准确率： 0.85234375
训练集召回率： 0.8585552543453995
F1: 0.8506873123716229
ROC值： 0.852466476919981'''

# 分类报告：precision/recall/fi-score/均值/分类个数
from sklearn.metrics import classification_report
target_names = ['-1','1']
print(classification_report(y_train_hat,y_train, target_names=target_names))
