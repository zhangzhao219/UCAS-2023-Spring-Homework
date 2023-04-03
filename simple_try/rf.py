from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

import jieba
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import mean_squared_error

from sklearn.naive_bayes import MultinomialNB

def dataprocess(x):
    return ','.join(list(jieba.cut(x)))

data_waimai_train = pd.read_csv('../data/waimai/Train.csv')
data_waimai_val = pd.read_csv('../data/waimai/Val.csv')
data_waimai_test = pd.read_csv('../data/waimai/Test.csv')

x_train = data_waimai_train["text"]
x_train = x_train.map(dataprocess)
y_train = data_waimai_train["label"]

x_test = data_waimai_test["text"]
x_test = x_test.map(dataprocess)
y_test = data_waimai_test["label"]


vec=TfidfVectorizer(analyzer='word', ngram_range=(1,4), max_features=500)
tfidf_x_train=vec.fit_transform(x_train) #与上面一种TfidfTransformer
tfidf_x_test=vec.fit_transform(x_test)

forest_reg=RandomForestRegressor()
forest_reg.fit(vec.transform(x_train),y_train)
print('训练集:',forest_reg.score(vec.transform(x_test),y_test))
print('测试集:',forest_reg.score(vec.transform(x_train),y_train))
'''训练集: 0.5592796842264617
测试集: 0.8471523627994759'''

