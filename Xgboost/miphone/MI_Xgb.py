# -*-coding:utf-8 -*-
import xgboost as xgb
from sklearn import datasets
import numpy as np
from sklearn.metrics import f1_score
from sqlalchemy import create_engine
import pymysql
import pandas as pd

models_path = "model/"
model_file = 'bst_mi.model'
def loadXyFromDB(sql):
    db_connection_str = 'mysql+pymysql://root:12345678@localhost/miphone'
    db_connection = create_engine(db_connection_str)
    df = pd.read_sql(sql, con=db_connection)
    # return np.array(df.iloc[:,:-1]),df.iloc[:,-1]
    return df.iloc[:,:-1],df.iloc[:,-1]

def loadDfFromDB(sql):
    db_connection_str = 'mysql+pymysql://root:12345678@localhost/miphone'
    db_connection = create_engine(db_connection_str)
    df = pd.read_sql(sql, con=db_connection)
    return df





def gen_model(sql):
    global bst, models_path, model_file
    X, y = loadXyFromDB(sql)
    print("loaded data")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
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
    print("开始训练")
    bst = xgb.train(param, dtrain, num_round)
    print("训练完成")
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
    print("f1值")
    print(f1_score(y_test, best_preds, average='macro'))
    '''
    模型存储与加载
    '''
    bst.save_model(models_path + model_file)


def gen_result(sql_tdata,res_file):
    global bst
    bst = xgb.Booster({'nthread': 4})  # init model
    bst.load_model(models_path + model_file)
    # 加载测试数据集
    pdata, uid = loadXyFromDB(sql_tdata)
    bst_predict = bst.predict(xgb.DMatrix(pdata))
    # print(bst_predict)
    best_preds = np.asarray([np.argmax(line) for line in bst_predict])
    pdata_label = pd.DataFrame({"uid": uid, "label": best_preds})
    pdata_label.to_csv("/Users/chengxingfu/code/AI/mi/res/0913/"+res_file,index=False)
    print(pdata)
    print('---')
    print(best_preds)
    print('---')
    pdata['label']=best_preds

    print(pdata[pdata.label.eq(1)])


if __name__ == '__main__':
    sql_data='''
    select  CAST(total_use_days AS DECIMAL(4)) tdays,
     CAST(user_age AS DECIMAL(4)) uage,
          CAST(user_sex AS DECIMAL(4)) usex,
           CAST(age AS DECIMAL(4))realage,
    case when concat('',tag * 1) = tag then 1
    else 0
    end tag
    from user;'''
    gen_model(sql_data)
    ndata_sql = '''select  CAST(total_use_days AS DECIMAL(4)) tdays,
     CAST(user_age AS DECIMAL(4)) uage,
      CAST(user_sex AS DECIMAL(4)) usex,
       CAST(age AS DECIMAL(4)) realage,uid
    from user_test ;'''
    gen_result(ndata_sql,"xgb_tdays_age_sex_rage.csv")


