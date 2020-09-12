# -*-coding:utf-8 -*-
# https://zhuanlan.zhihu.com/p/52560971
# conda install -c anaconda scikit-learn/ conda install scikit-learn /pip install -U scikit-learn
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import matplotlib
# %matplotlib inline
def GetNewDataByPandas():
    wine = pd.read_csv("data/wine.csv")
    wine['alcohol**2'] = pow(wine["alcohol"], 2)
    wine['volatileAcidity*alcohol'] = wine["alcohol"] * wine['volatile acidity']
    y = np.array(wine.quality)
    X = np.array(wine.drop("quality", axis=1))
    # X = np.array(wine)

    columns = np.array(wine.columns)

    return X, y, columns
'''
加载数据
'''
# Read wine quality data from file
X, y, wineNames = GetNewDataByPandas()
# X, y, wineNames = GetDataByPandas()
# split data to [0.8,0.2,01]
x_train, x_predict, y_train, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)

# take fixed holdout set 30% of data rows
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=100)
'''
展示数据
'''
wineNames
# array(['fixed acidity', 'volatile acidity', 'citric acid',
#        'residual sugar', 'chlorides', 'free sulfur dioxide',
#        'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
#        'quality', 'alcohol**2', 'volatileAcidity*alcohol'], dtype=object)
print(len(x_train),len(y_train))
print(len(x_test))
print(len(x_predict))
'''
加载到DMatrix
'''
dtrain = xgb.DMatrix(data=x_train,label=y_train)
dtest = xgb.DMatrix(data=x_test,label=y_test)
'''
设定参数
'''
# Booster参数
param = {'max_depth': 7, 'eta': 1,  'objective': 'reg:squarederror'}
param['nthread'] = 4
param['seed'] = 100
param['eval_metric'] = 'auc'
# 还可以指定多个ecal指标
param['eval_metric'] = ['auc', 'ams@0']
# 此处我们进行回归运算，只设置rmse
param['eval_metric'] = ['rmse']
param['eval_metric']
# 指定设置为监视性能的验证
evallist = [(dtest, 'eval'), (dtrain, 'train')]
'''
训练
'''
num_round = 10
bst_without_evallist = xgb.train(param, dtrain, num_round)
num_round = 10
bst_with_evallist = xgb.train(param, dtrain, num_round, evallist)
'''
保存模型
'''
models_path = "xgboost/"
bst_with_evallist.save_model(models_path+'bst_with_evallist_0001.model')
'''
加载模型
'''
bst_with_evallist = xgb.Booster({'nthread': 4})  # init model
bst_with_evallist.load_model(models_path+'bst_with_evallist_0001.model')  # load data
