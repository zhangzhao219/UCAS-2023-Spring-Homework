import pandas as pd
from sklearn.metrics import precision_score, recall_score,f1_score
from sklearn.metrics import classification_report

count = 0

def judge(score):
    if score >= 0:
        return 1
    else:
        return 0

#基于波森情感词典计算情感值
def getscore(text):

    global count
    if count % 100 == 0:
        print(count)
    count += 1

    df = pd.read_table("BosonNLP_sentiment_score.txt", sep=" ", names=['key', 'score'])
    key = df['key'].values.tolist()
    score = df['score'].values.tolist()
    segs = text.split(",")
    # 计算得分
    score_list = [score[key.index(x)] for x in segs if(x in key)]
    
    return judge(sum(score_list))

def run(name):
    data = pd.read_csv('../data/ml_data/' + FOLDER_NAME + '/' + name + '.csv')
    data["predict"] = data["text"].apply(getscore)
    print(name)
    print(classification_report(data["predict"],data["label"], target_names=target_names))
    print("精确率：",precision_score(y_pred=data["predict"],y_true=data["label"]))
    print("召回率：",recall_score(y_pred=data["predict"],y_true=data["label"]))
    print("F1-score：",f1_score(y_pred=data["predict"],y_true=data["label"]))

if __name__=='__main__':

    FOLDER_NAME = "waimai"
    target_names = ['0','1']

    run("Test")

