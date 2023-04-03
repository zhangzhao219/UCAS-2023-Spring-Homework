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

lin_reg=LinearRegression()

lin_reg.fit(vec.transform(x_train),y_train)

y_test_hat=lin_reg.predict(vec.transform(x_test))
y_train_hat=lin_reg.predict(vec.transform(x_train))
lin_mse1=mean_squared_error(y_test,y_test_hat)
lin_mse2=mean_squared_error(y_train,y_train_hat)
lin_rmse1=np.sqrt(lin_mse1)
lin_rmse2=np.sqrt(lin_mse2)
print ('测试集:',lin_rmse1)
print ('训练集:',lin_rmse2)

