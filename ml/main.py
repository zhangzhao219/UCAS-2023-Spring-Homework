import pandas as pd
import jieba
import sys
import re
from sklearn.metrics import balanced_accuracy_score,accuracy_score,f1_score,precision_score,recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

sys.path.append("/mnt/d/Programming_Design/Multimodal-Sentiment-Analysis")

from ml_method.linear import LinearMLMethod
from ml_method.nb import NBMLMethod
from ml_method.decisiontree import DTMLMethod
from ml_method.svm import SVMMLMethod
from ml_method.rf import RFMLMethod
from ml_method.lr import LRMLMethod

count = 0
results = re.compile(r'[http|https]*://[a-zA-Z0-9.?/&=:]*', re.S)
re_br = re.compile (r'<br\s*?/?>', re.S) #处理换行

def dataprocess(sentence):

    global count
    if count % 100 == 0:
        print(count)
    count += 1

    sentence = re.sub(results, '', sentence)
    sentence = re.sub(re_br, '', sentence)

    words_list=[]
    cut_words = list(jieba.cut(sentence))
    stopwords = getStopWords()
    for word in cut_words:
        if not (word in stopwords):
            if word != " ":
                words_list.append(word)
    return ','.join(words_list)

def getStopWords():
    stop = open('../baidu_stopwords.txt', 'r+', encoding='utf-8')
    return stop.read().split('\n')

def read_data(data_folder_name):
    data_train = pd.read_csv('../data/' + data_folder_name + '/Train.csv')
    data_val = pd.read_csv('../data/' + data_folder_name + '/Val.csv')
    data_test = pd.read_csv('../data/' + data_folder_name + '/Test.csv')

    print("train_data")
    x_train = data_train["text"]
    x_train = x_train.map(dataprocess)
    y_train = data_train["label"]
    
    print("val_data")
    x_val = data_val["text"]
    x_val = x_val.map(dataprocess)
    y_val = data_val["label"]

    print("test_data")   
    x_test = data_test["text"]
    x_test = x_test.map(dataprocess)
    y_test = data_test["label"]

    return x_train, y_train, x_val, y_val, x_test, y_test

def read_processed_data(data_folder_name):
    data_train = pd.read_csv('../data/' + data_folder_name + '/Train.csv')
    data_val = pd.read_csv('../data/' + data_folder_name + '/Val.csv')
    data_test = pd.read_csv('../data/' + data_folder_name + '/Test.csv')

    x_train = data_train["text"]
    y_train = data_train["label"]

    x_val = data_val["text"]
    y_val = data_val["label"]
 
    x_test = data_test["text"]
    y_test = data_test["label"]

    return x_train, y_train, x_val, y_val, x_test, y_test


# target_names = ['0','1']
FOLDER_NAME = "climate"

# # 处理数据的代码开始
# x_train, y_train, x_val, y_val, x_test, y_test = read_data(FOLDER_NAME)

# train_data = pd.concat([x_train,y_train],axis=1).dropna()
# val_data = pd.concat([x_val,y_val],axis=1).dropna()
# test_data = pd.concat([x_test,y_test],axis=1).dropna()

# train_data.to_csv('../data/ml_data/'+FOLDER_NAME+'/Train.csv',index=False, encoding='utf-8')
# val_data.to_csv('../data/ml_data/'+FOLDER_NAME+'/Val.csv',index=False, encoding='utf-8')
# test_data.to_csv('../data/ml_data/'+FOLDER_NAME+'/Test.csv',index=False, encoding='utf-8')

# FOLDER_NAME = "climate"

# x_train, y_train, x_val, y_val, x_test, y_test = read_data(FOLDER_NAME)

# train_data = pd.concat([x_train,y_train],axis=1).dropna()
# val_data = pd.concat([x_val,y_val],axis=1).dropna()
# test_data = pd.concat([x_test,y_test],axis=1).dropna()

# train_data.to_csv('../data/ml_data/'+FOLDER_NAME+'/Train.csv',index=False, encoding='utf-8')
# val_data.to_csv('../data/ml_data/'+FOLDER_NAME+'/Val.csv',index=False, encoding='utf-8')
# test_data.to_csv('../data/ml_data/'+FOLDER_NAME+'/Test.csv',index=False, encoding='utf-8')

# FOLDER_NAME = "IMDB"

# x_train, y_train, x_val, y_val, x_test, y_test = read_data(FOLDER_NAME)

# train_data = pd.concat([x_train,y_train],axis=1).dropna()
# val_data = pd.concat([x_val,y_val],axis=1).dropna()
# test_data = pd.concat([x_test,y_test],axis=1).dropna()

# train_data.to_csv('../data/ml_data/'+FOLDER_NAME+'/Train.csv',index=False, encoding='utf-8')
# val_data.to_csv('../data/ml_data/'+FOLDER_NAME+'/Val.csv',index=False, encoding='utf-8')
# test_data.to_csv('../data/ml_data/'+FOLDER_NAME+'/Test.csv',index=False, encoding='utf-8')

# FOLDER_NAME = "waimai"

# x_train, y_train, x_val, y_val, x_test, y_test = read_data(FOLDER_NAME)

# train_data = pd.concat([x_train,y_train],axis=1).dropna()
# val_data = pd.concat([x_val,y_val],axis=1).dropna()
# test_data = pd.concat([x_test,y_test],axis=1).dropna()

# train_data.to_csv('../data/ml_data/'+FOLDER_NAME+'/Train.csv',index=False, encoding='utf-8')
# val_data.to_csv('../data/ml_data/'+FOLDER_NAME+'/Val.csv',index=False, encoding='utf-8')
# test_data.to_csv('../data/ml_data/'+FOLDER_NAME+'/Test.csv',index=False, encoding='utf-8')

# # 处理数据的代码结束



def linear_score(predict):
    predict[predict < -0.5] = int(-1)
    predict[(predict > -0.5) & (predict < 0.5)] = int(0)
    predict[(predict > 0.5) & (predict < 1.5)] = int(1)
    predict[predict > 1.5] = int(2)
    return list(map(int,predict))

def linear_score2(predict):
    predict[predict < 0.5] = int(0)
    predict[predict >= 0.5] = int(1)
    return list(map(int,predict))

# x_train, y_train, x_val, y_val, x_test, y_test = read_processed_data(FOLDER_NAME)

# vec=TfidfVectorizer(analyzer='word', ngram_range=(1,4), max_features=500)
# tfidf_x_train=vec.fit_transform(x_train)

# predict_y_train,predict_y_val,predict_y_test, = RFMLMethod(vec, x_train, y_train, x_val, y_val, x_test, y_test)
# # LinearMLMethod
# # LRMLMethod
# # NBMLMethod
# # DTMLMethod
# # RFMLMethod
# # SVMMLMethod



# predict_y_train = linear_score(predict_y_train)
# predict_y_val = linear_score(predict_y_val)
# predict_y_test = linear_score(predict_y_test)

# # print("训练集：")
# # print("wa：",balanced_accuracy_score(y_pred=predict_y_train,y_true=y_train))
# # print("acc：",accuracy_score(y_pred=predict_y_train,y_true=y_train))
# # print("F1-score：",f1_score(y_pred=predict_y_train,y_true=y_train,average='macro'))

# # print("验证集：")
# # print("wa：",balanced_accuracy_score(y_pred=predict_y_val,y_true=y_val))
# # print("acc：",accuracy_score(y_pred=predict_y_val,y_true=y_val))
# # print("F1-score：",f1_score(y_pred=predict_y_val,y_true=y_val,average='macro'))

# print("测试集：")
# # print("wa：",balanced_accuracy_score(y_pred=predict_y_test,y_true=y_test))
# print("acc：",accuracy_score(y_pred=predict_y_test,y_true=y_test))
# print("pre：",precision_score(y_pred=predict_y_test,y_true=y_test,average='macro'))
# print("rec：",recall_score(y_pred=predict_y_test,y_true=y_test,average='macro'))
# print("F1-score：",f1_score(y_pred=predict_y_test,y_true=y_test,average='macro'))


FOLDER_NAME = "IMDB"

x_train, y_train, x_val, y_val, x_test, y_test = read_processed_data(FOLDER_NAME)

vec=TfidfVectorizer(analyzer='word', ngram_range=(1,4), max_features=500)
tfidf_x_train=vec.fit_transform(x_train)

predict_y_train,predict_y_val,predict_y_test, = LinearMLMethod(vec, x_train, y_train, x_val, y_val, x_test, y_test)
predict_y_train = linear_score2(predict_y_train)
predict_y_val = linear_score2(predict_y_val)
predict_y_test = linear_score2(predict_y_test)
print("LinearMLMethod")
print("acc：",accuracy_score(y_pred=predict_y_test,y_true=y_test))
print("pre：",precision_score(y_pred=predict_y_test,y_true=y_test,average='macro'))
print("rec：",recall_score(y_pred=predict_y_test,y_true=y_test,average='macro'))
print("F1-score：",f1_score(y_pred=predict_y_test,y_true=y_test,average='macro'))


predict_y_train,predict_y_val,predict_y_test, = LRMLMethod(vec, x_train, y_train, x_val, y_val, x_test, y_test)
predict_y_train = linear_score2(predict_y_train)
predict_y_val = linear_score2(predict_y_val)
predict_y_test = linear_score2(predict_y_test)
print("LRMLMethod")
print("acc：",accuracy_score(y_pred=predict_y_test,y_true=y_test))
print("pre：",precision_score(y_pred=predict_y_test,y_true=y_test,average='macro'))
print("rec：",recall_score(y_pred=predict_y_test,y_true=y_test,average='macro'))
print("F1-score：",f1_score(y_pred=predict_y_test,y_true=y_test,average='macro'))


predict_y_train,predict_y_val,predict_y_test, = NBMLMethod(vec, x_train, y_train, x_val, y_val, x_test, y_test)
predict_y_train = linear_score2(predict_y_train)
predict_y_val = linear_score2(predict_y_val)
predict_y_test = linear_score2(predict_y_test)
print("NBMLMethod")
print("acc：",accuracy_score(y_pred=predict_y_test,y_true=y_test))
print("pre：",precision_score(y_pred=predict_y_test,y_true=y_test,average='macro'))
print("rec：",recall_score(y_pred=predict_y_test,y_true=y_test,average='macro'))
print("F1-score：",f1_score(y_pred=predict_y_test,y_true=y_test,average='macro'))

predict_y_train,predict_y_val,predict_y_test, = DTMLMethod(vec, x_train, y_train, x_val, y_val, x_test, y_test)
predict_y_train = linear_score2(predict_y_train)
predict_y_val = linear_score2(predict_y_val)
predict_y_test = linear_score2(predict_y_test)
print("DTMLMethod")
print("acc：",accuracy_score(y_pred=predict_y_test,y_true=y_test))
print("pre：",precision_score(y_pred=predict_y_test,y_true=y_test,average='macro'))
print("rec：",recall_score(y_pred=predict_y_test,y_true=y_test,average='macro'))
print("F1-score：",f1_score(y_pred=predict_y_test,y_true=y_test,average='macro'))

predict_y_train,predict_y_val,predict_y_test, = RFMLMethod(vec, x_train, y_train, x_val, y_val, x_test, y_test)
predict_y_train = linear_score2(predict_y_train)
predict_y_val = linear_score2(predict_y_val)
predict_y_test = linear_score2(predict_y_test)
print("RFMLMethod")
print("acc：",accuracy_score(y_pred=predict_y_test,y_true=y_test))
print("pre：",precision_score(y_pred=predict_y_test,y_true=y_test,average='macro'))
print("rec：",recall_score(y_pred=predict_y_test,y_true=y_test,average='macro'))
print("F1-score：",f1_score(y_pred=predict_y_test,y_true=y_test,average='macro'))

predict_y_train,predict_y_val,predict_y_test, = SVMMLMethod(vec, x_train, y_train, x_val, y_val, x_test, y_test)
predict_y_train = linear_score2(predict_y_train)
predict_y_val = linear_score2(predict_y_val)
predict_y_test = linear_score2(predict_y_test)
print("SVMMLMethod")
print("acc：",accuracy_score(y_pred=predict_y_test,y_true=y_test))
print("pre：",precision_score(y_pred=predict_y_test,y_true=y_test,average='macro'))
print("rec：",recall_score(y_pred=predict_y_test,y_true=y_test,average='macro'))
print("F1-score：",f1_score(y_pred=predict_y_test,y_true=y_test,average='macro'))


FOLDER_NAME = "waimai"

x_train, y_train, x_val, y_val, x_test, y_test = read_processed_data(FOLDER_NAME)

vec=TfidfVectorizer(analyzer='word', ngram_range=(1,4), max_features=500)
tfidf_x_train=vec.fit_transform(x_train)

predict_y_train,predict_y_val,predict_y_test, = LinearMLMethod(vec, x_train, y_train, x_val, y_val, x_test, y_test)
predict_y_train = linear_score2(predict_y_train)
predict_y_val = linear_score2(predict_y_val)
predict_y_test = linear_score2(predict_y_test)
print("LinearMLMethod")
print("acc：",accuracy_score(y_pred=predict_y_test,y_true=y_test))
print("pre：",precision_score(y_pred=predict_y_test,y_true=y_test,average='macro'))
print("rec：",recall_score(y_pred=predict_y_test,y_true=y_test,average='macro'))
print("F1-score：",f1_score(y_pred=predict_y_test,y_true=y_test,average='macro'))


predict_y_train,predict_y_val,predict_y_test, = LRMLMethod(vec, x_train, y_train, x_val, y_val, x_test, y_test)
predict_y_train = linear_score2(predict_y_train)
predict_y_val = linear_score2(predict_y_val)
predict_y_test = linear_score2(predict_y_test)
print("LRMLMethod")
print("acc：",accuracy_score(y_pred=predict_y_test,y_true=y_test))
print("pre：",precision_score(y_pred=predict_y_test,y_true=y_test,average='macro'))
print("rec：",recall_score(y_pred=predict_y_test,y_true=y_test,average='macro'))
print("F1-score：",f1_score(y_pred=predict_y_test,y_true=y_test,average='macro'))


predict_y_train,predict_y_val,predict_y_test, = NBMLMethod(vec, x_train, y_train, x_val, y_val, x_test, y_test)
predict_y_train = linear_score2(predict_y_train)
predict_y_val = linear_score2(predict_y_val)
predict_y_test = linear_score2(predict_y_test)
print("NBMLMethod")
print("acc：",accuracy_score(y_pred=predict_y_test,y_true=y_test))
print("pre：",precision_score(y_pred=predict_y_test,y_true=y_test,average='macro'))
print("rec：",recall_score(y_pred=predict_y_test,y_true=y_test,average='macro'))
print("F1-score：",f1_score(y_pred=predict_y_test,y_true=y_test,average='macro'))

predict_y_train,predict_y_val,predict_y_test, = DTMLMethod(vec, x_train, y_train, x_val, y_val, x_test, y_test)
predict_y_train = linear_score2(predict_y_train)
predict_y_val = linear_score2(predict_y_val)
predict_y_test = linear_score2(predict_y_test)
print("DTMLMethod")
print("acc：",accuracy_score(y_pred=predict_y_test,y_true=y_test))
print("pre：",precision_score(y_pred=predict_y_test,y_true=y_test,average='macro'))
print("rec：",recall_score(y_pred=predict_y_test,y_true=y_test,average='macro'))
print("F1-score：",f1_score(y_pred=predict_y_test,y_true=y_test,average='macro'))

predict_y_train,predict_y_val,predict_y_test, = RFMLMethod(vec, x_train, y_train, x_val, y_val, x_test, y_test)
predict_y_train = linear_score2(predict_y_train)
predict_y_val = linear_score2(predict_y_val)
predict_y_test = linear_score2(predict_y_test)
print("RFMLMethod")
print("acc：",accuracy_score(y_pred=predict_y_test,y_true=y_test))
print("pre：",precision_score(y_pred=predict_y_test,y_true=y_test,average='macro'))
print("rec：",recall_score(y_pred=predict_y_test,y_true=y_test,average='macro'))
print("F1-score：",f1_score(y_pred=predict_y_test,y_true=y_test,average='macro'))

predict_y_train,predict_y_val,predict_y_test, = SVMMLMethod(vec, x_train, y_train, x_val, y_val, x_test, y_test)
predict_y_train = linear_score2(predict_y_train)
predict_y_val = linear_score2(predict_y_val)
predict_y_test = linear_score2(predict_y_test)
print("SVMMLMethod")
print("acc：",accuracy_score(y_pred=predict_y_test,y_true=y_test))
print("pre：",precision_score(y_pred=predict_y_test,y_true=y_test,average='macro'))
print("rec：",recall_score(y_pred=predict_y_test,y_true=y_test,average='macro'))
print("F1-score：",f1_score(y_pred=predict_y_test,y_true=y_test,average='macro'))