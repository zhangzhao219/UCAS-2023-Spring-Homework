#引入随机搜索，选择最优模型参数
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

import jieba
import numpy as np
import pandas as pd


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
 
tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)
 
param_grid = [{'vect__ngram_range': [(1, 1)],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              ]
 
lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0))])
 
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)

gs_lr_tfidf.fit(x_train, y_train)
print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(x_test, y_test))
'''
Best parameter set: {'clf__C': 10.0, 'clf__penalty': 'l2', 'vect__ngram_range': (1, 1)}
CV Accuracy: 0.877
Test Accuracy: 0.888'''
