import pandas as pd
from sklearn.model_selection import train_test_split 
from collections import Counter


print("开始进行数据集划分")
print("开始划分waimai数据集")

# 读取数据
data_waimai = pd.read_csv('./data/waimai/waimai_10k.csv')
data_waimai_x = data_waimai['review']
data_waimai_y = data_waimai['label']

# 按照7 2 1的比例进行划分
train_x,val_x,train_y,val_y = train_test_split(data_waimai_x, data_waimai_y, test_size = 0.3,random_state = 2023, stratify=data_waimai_y)
val_x,test_x,val_y,test_y = train_test_split(val_x, val_y, test_size = 0.3,random_state = 2023, stratify=val_y)

# 拼接划分好的数据
pd.concat([train_x,train_y],axis=1).to_csv('./data/waimai/Train.csv', header=["text","label"],index=False, encoding='utf-8')
pd.concat([val_x,val_y], axis=1).to_csv('./data/waimai/Val.csv', header=["text","label"],index=False, encoding='utf-8')
pd.concat([test_x,test_y], axis=1).to_csv('./data/waimai/Test.csv', header=["text","label"],index=False, encoding='utf-8')


print("waimai数据集训练集样本标签分布：",Counter(train_y))
print("waimai数据集训练集样本标签分布：",Counter(val_y))
print("waimai数据集训练集样本标签分布：",Counter(test_y))

print("开始划分climate数据集")

# 读取数据
data_climate = pd.read_csv('./data/climate/twitter_sentiment_data.csv')
data_climate_x = data_climate['message']
data_climate_y = data_climate['sentiment']

# 按照7 2 1的比例进行划分
train_x,val_x,train_y,val_y = train_test_split(data_climate_x, data_climate_y, test_size = 0.3,random_state = 2023, stratify=data_climate_y)
val_x,test_x,val_y,test_y = train_test_split(val_x, val_y, test_size = 0.3,random_state = 2023, stratify=val_y)

# 拼接划分好的数据
pd.concat([train_x,train_y],axis=1).to_csv('./data/climate/Train.csv', header=["text","label"],index=False, encoding='utf-8')
pd.concat([val_x,val_y], axis=1).to_csv('./data/climate/Val.csv', header=["text","label"],index=False, encoding='utf-8')
pd.concat([test_x,test_y], axis=1).to_csv('./data/climate/Test.csv', header=["text","label"],index=False, encoding='utf-8')

print("climate数据集训练集样本标签分布：",Counter(train_y))
print("climate数据集训练集样本标签分布：",Counter(val_y))
print("climate数据集训练集样本标签分布：",Counter(test_y))