##----------2.决策树-----##
#模型效果不好的时候（拟合不足），考虑换个更强大的模型，决策树
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
tree_reg=DecisionTreeRegressor()

tree_reg.fit(vec.transform(x_train),y_train)

y_test_hat=tree_reg.predict(vec.transform(x_test))
y_train_hat=tree_reg.predict(vec.transform(x_train))
tree_mse1=mean_squared_error(y_test,y_test_hat)
tree_mse2=mean_squared_error(y_train,y_train_hat)
tree_rmse1=np.sqrt(tree_mse1)
tree_rmse2=np.sqrt(tree_mse2)
print ('测试集',tree_rmse1)
print ('训练集',tree_rmse2)
