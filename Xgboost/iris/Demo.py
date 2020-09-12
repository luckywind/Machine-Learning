# -*-coding:utf-8 -*-
import xgboost as xgb
from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris.data
y = iris.target

from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

'''
设置参数
参数很关键
参考https://www.kaggle.com/tanitter/grid-search-xgboost-with-scikit-learn
 Generally try with eta 0.1, 0.2, 0.3, max_depth in range of 2 to 10 and num_round around few hundred.
'''
param = {
    'max_depth': 3,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 3}  # the number of classes that exist in this datset
num_round = 20  # the number of training iterations

bst = xgb.train(param, dtrain, num_round)
bst.dump_model('dump.raw.txt')
'''
输出每个类别的概率
'''
preds = bst.predict(dtest)
best_preds = np.asarray([np.argmax(line) for line in preds])

'''
准确率
'''
from sklearn.metrics import precision_score
print("准确率")
print(precision_score(y_test, best_preds, average='macro'))

'''
模型存储与加载
'''
models_path = "model/"
bst.save_model(models_path+'bst_iris.model')
bst = xgb.Booster({'nthread': 4})  # init model
bst.load_model(models_path+'bst_iris.model')
