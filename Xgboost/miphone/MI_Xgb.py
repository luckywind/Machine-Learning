# -*-coding:utf-8 -*-
import xgboost as xgb
from sklearn import datasets
import numpy as np
from sqlalchemy import create_engine
import pymysql
import pandas as pd


def loadDataFromDB(sql):
    db_connection_str = 'mysql+pymysql://root:12345678@localhost/miphone'
    db_connection = create_engine(db_connection_str)
    df = pd.read_sql(sql, con=db_connection)
    return np.array(df.iloc[:,:-1]),df.iloc[:,-1]


sql_data='''select caton.caton_num,caton_duration,serious_caton_num,serious_caton_duration,
app_cate.category ,
use_count,use_duration,days,his.package,
u.brand, u.total_use_days,u.user_age,u.user_sex,u.age,user_degree,
brand_price.price ,
case when concat('',tag * 1) = tag then 1
     else 0
		 end tag 
from  (select * from user limit 10) u  left join app_history his on u.uid=his.uid
join app_cate on his.package =app_cate.package_name
join phone_caton caton on his.package=caton.pkn
join brand_price  on u.brand=brand_price.brand
'''
X,y=loadDataFromDB(sql_data)
print("loaded data")
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
    'num_class': 2}  # the number of classes that exist in this datset
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
model_file = 'bst_mi.model'
bst.save_model(models_path + model_file)
bst = xgb.Booster({'nthread': 4})  # init model
bst.load_model(models_path + model_file)
