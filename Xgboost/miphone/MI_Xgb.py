# -*-coding:utf-8 -*-
import xgboost as xgb
from sklearn import datasets
import numpy as np
from sklearn.metrics import f1_score
from sqlalchemy import create_engine
import pymysql
import pandas as pd

models_path = "model/"
# model_file = 'bst_mi.model'
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





def gen_model(sql,model_file):
    # global bst, models_path, model_file
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
    score = precision_score(y_test, best_preds, average='macro')
    print(score)
    print("f1值")
    f_score = f1_score(y_test, best_preds, average='macro')
    print(f_score)
    append_file("model_matrix.txt","model: %s 准确度:%s f1:%s"
                %(model_file,score,f_score))
    '''
    模型存储与加载
    '''
    bst.save_model(models_path + model_file)


def gen_result(sql_tdata,model_file,res_file):
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


def append_file(filename,str):
    # 打开文件
    fo = open(filename, "w")
    fo.write( str )
    # 关闭文件
    fo.close()

def test_a_model(sql_data,ndata_sql,mfile,res_file):
    gen_model(sql_data, model_file=mfile)
    gen_result(ndata_sql, model_file=mfile, res_file=res_file)



if __name__ == '__main__':
    test_a_model('''
    select CAST(total_use_days AS DECIMAL(4)) tdays,
     CAST(user_age AS DECIMAL(4)) uage,
          CAST(user_sex AS DECIMAL(4)) usex,
           CAST(age AS DECIMAL(4))realage,
            CASE WHEN a.1 = 'True' THEN 1 ELSE 0 END ,
             CASE WHEN a.2 = 'True' THEN 1 ELSE 0 END ,
              CASE WHEN a.3 = 'True' THEN 1 ELSE 0 END ,
               CASE WHEN a.4 = 'True' THEN 1 ELSE 0 END ,
                CASE WHEN a.5 = 'True' THEN 1 ELSE 0 END ,
                 CASE WHEN a.6 = 'True' THEN 1 ELSE 0 END ,
                  CASE WHEN a.7 = 'True' THEN 1 ELSE 0 END ,
                   CASE WHEN a.8 = 'True' THEN 1 ELSE 0 END ,
                    CASE WHEN a.9 = 'True' THEN 1 ELSE 0 END ,
                     CASE WHEN a.0 = 'True' THEN 1 ELSE 0 END ,
                      CASE WHEN a.11 = 'True' THEN 1 ELSE 0 END ,
                       CASE WHEN a.12 = 'True' THEN 1 ELSE 0 END ,
                        CASE WHEN a.13 = 'True' THEN 1 ELSE 0 END ,
                         CASE WHEN a.14 = 'True' THEN 1 ELSE 0 END ,
                          CASE WHEN a.15 = 'True' THEN 1 ELSE 0 END ,
                           CASE WHEN a.16 = 'True' THEN 1 ELSE 0 END ,
                            CASE WHEN a.17 = 'True' THEN 1 ELSE 0 END ,
                             CASE WHEN a.18 = 'True' THEN 1 ELSE 0 END ,
                              CASE WHEN a.19 = 'True' THEN 1 ELSE 0 END ,
                               CASE WHEN a.20 = 'True' THEN 1 ELSE 0 END ,
                                CASE WHEN a.21 = 'True' THEN 1 ELSE 0 END ,
                                 CASE WHEN a.22 = 'True' THEN 1 ELSE 0 END ,
                                  CASE WHEN a.23 = 'True' THEN 1 ELSE 0 END ,
                                   CASE WHEN a.24 = 'True' THEN 1 ELSE 0 END ,
                                    CASE WHEN a.25 = 'True' THEN 1 ELSE 0 END ,
                                     CASE WHEN a.26 = 'True' THEN 1 ELSE 0 END ,
                                      CASE WHEN a.27 = 'True' THEN 1 ELSE 0 END ,
                                       CASE WHEN a.28 = 'True' THEN 1 ELSE 0 END ,
                                        CASE WHEN a.29 = 'True' THEN 1 ELSE 0 END ,
                                         CASE WHEN a.30 = 'True' THEN 1 ELSE 0 END ,
    case when concat('',tag * 1) = tag then 1
    else 0
    end tag
from user u
join user_active_history30 a on u.uid=a.uid''',
                 '''select CAST(total_use_days AS DECIMAL(4)) tdays,
     CAST(user_age AS DECIMAL(4)) uage,
          CAST(user_sex AS DECIMAL(4)) usex,
           CAST(age AS DECIMAL(4))realage,
                      CASE WHEN a.1 = 'True' THEN 1 ELSE 0 END ,
             CASE WHEN a.2 = 'True' THEN 1 ELSE 0 END ,
              CASE WHEN a.3 = 'True' THEN 1 ELSE 0 END ,
               CASE WHEN a.4 = 'True' THEN 1 ELSE 0 END ,
                CASE WHEN a.5 = 'True' THEN 1 ELSE 0 END ,
                 CASE WHEN a.6 = 'True' THEN 1 ELSE 0 END ,
                  CASE WHEN a.7 = 'True' THEN 1 ELSE 0 END ,
                   CASE WHEN a.8 = 'True' THEN 1 ELSE 0 END ,
                    CASE WHEN a.9 = 'True' THEN 1 ELSE 0 END ,
                     CASE WHEN a.0 = 'True' THEN 1 ELSE 0 END ,
                      CASE WHEN a.11 = 'True' THEN 1 ELSE 0 END ,
                       CASE WHEN a.12 = 'True' THEN 1 ELSE 0 END ,
                        CASE WHEN a.13 = 'True' THEN 1 ELSE 0 END ,
                         CASE WHEN a.14 = 'True' THEN 1 ELSE 0 END ,
                          CASE WHEN a.15 = 'True' THEN 1 ELSE 0 END ,
                           CASE WHEN a.16 = 'True' THEN 1 ELSE 0 END ,
                            CASE WHEN a.17 = 'True' THEN 1 ELSE 0 END ,
                             CASE WHEN a.18 = 'True' THEN 1 ELSE 0 END ,
                              CASE WHEN a.19 = 'True' THEN 1 ELSE 0 END ,
                               CASE WHEN a.20 = 'True' THEN 1 ELSE 0 END ,
                                CASE WHEN a.21 = 'True' THEN 1 ELSE 0 END ,
                                 CASE WHEN a.22 = 'True' THEN 1 ELSE 0 END ,
                                  CASE WHEN a.23 = 'True' THEN 1 ELSE 0 END ,
                                   CASE WHEN a.24 = 'True' THEN 1 ELSE 0 END ,
                                    CASE WHEN a.25 = 'True' THEN 1 ELSE 0 END ,
                                     CASE WHEN a.26 = 'True' THEN 1 ELSE 0 END ,
                                      CASE WHEN a.27 = 'True' THEN 1 ELSE 0 END ,
                                       CASE WHEN a.28 = 'True' THEN 1 ELSE 0 END ,
                                        CASE WHEN a.29 = 'True' THEN 1 ELSE 0 END ,
                                         CASE WHEN a.30 = 'True' THEN 1 ELSE 0 END ,

    u.uid
from user_test u
join user_active_history30_test a on u.uid=a.uid''',
                 mfile= 'model_u_30a',
                 res_file= "xgb_model_u_30a.csv"
                 )


'''只使用user的几个属性'''
    # test_a_model('''
    # select  CAST(total_use_days AS DECIMAL(4)) tdays,
    #  CAST(user_age AS DECIMAL(4)) uage,
    #       CAST(user_sex AS DECIMAL(4)) usex,
    #        CAST(age AS DECIMAL(4))realage,
    # case when concat('',tag * 1) = tag then 1
    # else 0
    # end tag
    # from user;''',
    #              '''select  CAST(total_use_days AS DECIMAL(4)) tdays,
    #              CAST(user_age AS DECIMAL(4)) uage,
    #               CAST(user_sex AS DECIMAL(4)) usex,
    #                CAST(age AS DECIMAL(4)) realage,uid
    #             from user_test ;''',
    #             mfile= 'model_u_30a',
    #             res_file= "xgb_model_u_30a.csv"
    #              )


