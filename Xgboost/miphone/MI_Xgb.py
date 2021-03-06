# -*-coding:utf-8 -*-
import time

import xgboost as xgb
from sklearn import datasets
import numpy as np
from sklearn.metrics import f1_score
from sqlalchemy import create_engine
import pymysql
import pandas as pd
import random
from sklearn import  metrics
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

models_path = "model/"
# model_file = 'bst_mi.model'
def loadXyFromDB(sql):
    df=loadDfFromDB(sql)
    # return np.array(df.iloc[:,:-1]),df.iloc[:,-1]
    return df,df.iloc[:,:-1],df.iloc[:,-1]

def loadDfFromDB(sql):
    db_connection_str = 'mysql+pymysql://root:12345678@localhost/miphone'
    db_connection = create_engine(db_connection_str)
    df = pd.read_sql(sql, con=db_connection)
    return df


'''证样本占比'''
def undersampling(train, desired_apriori):

    # Get the indices per target value
    idx_0 = train[train['tag'] == 0].index
    idx_1 = train[train['tag'] == 1].index
    # Get original number of records per target value
    nb_0 = len(train.loc[idx_0])
    nb_1 = len(train.loc[idx_1])
    # Calculate the undersampling rate and resulting number of records with target=0
    undersampling_rate = ((1-desired_apriori)*nb_1)/(nb_0*desired_apriori)
    undersampled_nb_0 = int(undersampling_rate*nb_0)
    print('Rate to undersample records with target=0: {}'.format(undersampling_rate))
    print('Number of records with target=0 after undersampling: {}'.format(undersampled_nb_0))
    # Randomly select records with target=0 to get at the desired a priori
    undersampled_idx = random.sample(list(idx_0), undersampled_nb_0)
    # Construct list with remaining indices
    idx_list = list(undersampled_idx) + list(idx_1)
    # Return undersample data frame
    train = train.loc[idx_list].reset_index(drop=True)

    return train


def get_precision_f1(y_test,best_preds):
    from sklearn import  metrics
    precision = metrics.precision_score(y_test, best_preds)
    f1_score = metrics.f1_score(y_test, best_preds)
    return precision,f1_score
def gen_model(df,model_file, desired_apriori,naround,save_model=True):
    print(df)
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    y.columns=['tag']
    X_src=X.copy()
    y_src=y.copy()
    print('采样前的正负样本数')
    print(df.groupby('tag').size())
    # 把df预留部分出来，用于测算f1（不预留时，全训练集算出的f1会偏高）
    df_train, df_test = train_test_split(df, test_size=0.01)
    X_y_sample=undersampling(df_train, desired_apriori)
    # 把使用天数过少的用户直接置为不换机，这部分数据也不加入模型训练,最终测试发现这个规则不好用
    # X_y_sample=X_y_sample[X_y_sample['tdays']<=350]
    print('调整后的正负样本数')
    print(X_y_sample.groupby('tag').size())
    X,y=(X_y_sample.iloc[:,:-1],X_y_sample.iloc[:,-1])

    print("loaded data")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

    # print("---")
    # print(X_train)
    # print('--y_train')
    # print(y_train)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    '''
设置参数
参数很关键
参考https://www.kaggle.com/tanitter/grid-search-xgboost-with-scikit-learn
 Generally try with eta 0.1, 0.2, 0.3, max_depth in range of 2 to 10 and num_round around few hundred.
'''
    param = {
        'max_depth': 4,  # the maximum depth of each tree
        'eta': 0.1,  # the training step for each iteration
        'objective': 'binary:logistic',
        'subsample':0.6,
        'colsample_bytree':0.8
        # ,  # error evaluation for multiclass training
        # 'num_class': 2
    }  # the number of classes that exist in this datset
    # num_round = 400  # the number of training iterations
    # for num_round in range(50,300,10):
    num_round=naround
    print("迭代次数%s"%num_round)
    print("开始训练")
    bst = xgb.train(param, dtrain, num_round)
    print("训练完成")
    bst.dump_model('dump.raw.txt')
    '''
    输出每个类别的概率
    '''
    preds = bst.predict(dtest)
    # print('preds')
    # print(preds)
    # best_preds = np.asarray([np.argmax(line) for line in preds]) 用于多分类场景
    best_preds = [ 1 if prob>0.5 else 0 for prob in preds]
    '''
    算分
    '''
    # (1)计算测试集f1值
    score,f_score = get_precision_f1(y_test, best_preds)
    score_line = "model: %s 迭代次数:%s 测试集精确率:%s f1:%s" % (model_file, num_round, score, f_score)
    print("采样测试集，预测换机数：%s, 真实换机数：%s"%(str(np.sum(best_preds)),str(np.sum(y_test))))
    print(score_line)
    append_file("model_matrix.txt", score_line)

    # (2)计算训练集f1值
    train_preds_prob = bst.predict(dtrain)
    train_preds= [ 1 if prob>0.5 else 0 for prob in train_preds_prob]
    print("采样训练集，预测换机数：%s, 真实换机数：%s"%(str(np.sum(train_preds)),str(np.sum(y_train))))
    train_score,train_f1_score = get_precision_f1(y_train, train_preds)
    train_score_line = "model: %s 迭代次数:%s 训练集精确率:%s f1:%s" % (model_file, num_round, train_score, train_f1_score)
    print(train_score_line)
    append_file("model_matrix.txt", train_score_line)

    ''' (3)在整个训练集上计算f1'''
    d_matrix = xgb.DMatrix(X_src)
    train_src_preds_prob = bst.predict(d_matrix)
    train_src_preds= [ 1 if prob>0.5 else 0 for prob in train_src_preds_prob]
    print("全训练集，预测换机数：%s, 真实换机数：%s"%(str(np.sum(train_src_preds)),str(np.sum(y_src))))
    train_src_score,train_src_f1_score=get_precision_f1(train_src_preds,y_src)
    train_src_score_line = "model: %s 迭代次数:%s 训练全集精确率:%s f1:%s" % (model_file, num_round, train_src_score, train_src_f1_score)
    print(train_src_score_line)
    append_file("model_matrix.txt", train_src_score_line)

    # (4)计算在预留数据集df_test的f1
    df_test_matrix = xgb.DMatrix(df_test.iloc[:, :-1])
    df_test_y=df_test.iloc[:,-1]
    df_test_preds_prob = bst.predict(df_test_matrix)
    df_test_preds= [ 1 if prob>0.5 else 0 for prob in df_test_preds_prob]
    # 使用天数过少的不参与训练的数据，直接置为不换机
    # for i in list(df_test[df_test['tdays']<=350].index):
    #     df_test_preds[i]=0
    print("预留数据集，预测换机数：%s, 真实换机数：%s"%(str(np.sum(df_test_preds)),str(np.sum(df_test_y))))
    df_test_score,df_test_f1_score=get_precision_f1(df_test_preds,df_test_y)
    df_test_score_line = "model: %s 迭代次数:%s 预留集合精确率:%s f1:%s" % (model_file, num_round, df_test_score, df_test_f1_score)
    print(df_test_score_line)
    append_file("model_matrix.txt", df_test_score_line)

    '''
    模型存储与加载
    '''
    if(save_model):
        model_file_abs = models_path + model_file + "_" + str(num_round)+"_"+str(desired_apriori)#+str(time.time())
        print("保存模型:%s" %(model_file_abs))
        bst.save_model(model_file_abs)
        append_file("model_matrix.txt", model_file_abs)


def gen_result(sql_tdata,str_feature_index_list,model_file,res_file):
    global bst
    bst = xgb.Booster({'nthread': 4})  # init model
    bst.load_model(models_path + model_file)
    # 加载测试数据集
    print("model  loaded...")
    # pdf,pdata, uid = loadXyFromDB(sql_tdata)
    pdf = loadDfFromDB(sql_tdata)
    pdf=one_hot_part(pdf,str_feature_index_list)
    pdata=pdf.iloc[:,:-1]
    uid=pdf.iloc[:,-1]
    print("测试数据集维度:")
    print(pdata.shape)
    bst_predict = bst.predict(xgb.DMatrix(pdata))
    # print(bst_predict)
    # best_preds = np.asarray([np.argmax(line) for line in bst_predict])
    best_preds = [ 1 if prob>0.5 else 0 for prob in bst_predict]
    uid_label = pd.DataFrame({"uid": uid, "label": best_preds})
    pdf['label']=best_preds  #预测数据集加一列预测值

    print("uid_label")
    print(uid_label.shape)
    # 换机概率大的直接置为换机
    huanji_df=loadDfFromDB('select uid,1 label from user_test where total_use_days between 2250 and 2700')
    tmp_res=pd.concat([huanji_df,uid_label])
    tmp_res.drop_duplicates(subset=['uid'], inplace=True, keep='first')

    # 缺少的数据直接预测为0不换机(这里包括没有进入训练集的数据)
    uiddf=loadDfFromDB('select uid,0 label from user_test ')
    print("uiddf")
    print(uiddf.shape)
    res=pd.concat([uid_label,uiddf])
    res.drop_duplicates(subset=['uid'], inplace=True, keep='first')

    print("res")
    print(res.shape)
    res.to_csv("/Users/chengxingfu/code/AI/mi/res/0918/"+res_file,index=False)
    pdata['label']=best_preds
    print("测试集预测换机数")
    print(pdata[pdata.label.eq(1)].shape)


def append_file(filename,str):
    # 打开文件
    fo = open(filename, "a")
    fo.write( str+"\n" )
    # 关闭文件
    fo.close()

'''独热处理字符型特征矩阵,至少要有两个列'''
def one_hot(X):
    encoded_x = None
    for i in range(0, X.shape[1]):
        label_encoder = LabelEncoder()
        feature = label_encoder.fit_transform(X.iloc[:,0])
        feature = feature.reshape(X.shape[0], 1) #转成列
        onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
        feature = onehot_encoder.fit_transform(feature)
        if encoded_x is None:
            encoded_x = feature
        else:
            encoded_x = np.concatenate((encoded_x, feature), axis=1) #把多个string属性产生的稀疏矩阵并列起来
    print("X shape: : ", encoded_x.shape)
    return encoded_x

'''把df中下标在str_feature_index_list的特征进行one_hot编码'''
def one_hot_part(df, str_feature_index_list):
    one_hot_x = one_hot(df.iloc[:, str_feature_index_list])
    one_hot_df=pd.DataFrame(data=one_hot_x)
    # df=np.concatenate((one_hot_df,df),axis=1)
    df.drop(df.columns[str_feature_index_list],axis=1,inplace=True)
    df=pd.concat([one_hot_df,df],axis=1)
    # df=pd.DataFrame(data=df)
    return df


def test_a_model(str_feature_index_list,sql_data, pdata_sql, mfile, res_file):
    df = loadDfFromDB(sql_data)
    # 加一步：把string型特征one-hoting处理
    df=one_hot_part(df,str_feature_index_list)
    for naround in range(200,280,4):
        for desired_apriori in np.arange(0.2,0.5,0.1):
        # naround=240
        #     desired_apriori=0.3 #因为每次都是0.3最优，确定之
            train_param = "迭代次数：%s,正样本率:%s" % (str(naround), str(desired_apriori))
            append_file("model_matrix.txt", train_param)
            print(train_param)
            gen_model(df, model_file=mfile,desired_apriori=desired_apriori,naround=naround,save_model=False)

    # naround=290
    # desired_apriori=0.3#0.5981061167977055 f1:0.4749759478510141
    # 295,0.2 训练全集精确率:0.5143401077980517 f1:0.5515623135762862
    # naround=260
    # desired_apriori=0.4
    # gen_model(df, model_file=mfile,desired_apriori=desired_apriori,naround=naround,save_model=True)
    gen_result(pdata_sql,str_feature_index_list, model_file=mfile + "_" + str(naround) + "_" + str(desired_apriori), res_file=res_file + "_" + str(naround) + "_" + str(desired_apriori) + str(".csv"))



if __name__ == '__main__':

    test_a_model([0,1,2,3,4],'''
    select 
    case 
  when u.brand like '小米MAX%' then '小米MAX'
  when u.brand like '小米Mix%' then '小米Mix'
  when u.brand like '小米Note%' then '小米Note'
  when u.brand like '小米手机%' then '小米手机'
  when u.brand like '红米Note%' then '红米Note'
  when u.brand like '红米K%' then '红米Kill'
  when u.brand like '红米手机%' or u.brand in ('红米Pro','红米S2') then '红米手机'
  else u.brand end as brand_type,
  modelname,
  version,
  user_degree,
  resident_city_type,
CAST(total_use_days AS DECIMAL(4)) tdays,
 CAST(user_age AS DECIMAL(4)) uage,
  CAST(user_sex AS DECIMAL(4)) usex,
  CAST(age AS DECIMAL(4))realage,
CAST(SUBSTRING_INDEX(price, '-', 1) AS UNSIGNED) min_price,
CAST(SUBSTRING_INDEX(price, '-', -1) AS UNSIGNED) max_price,
       ucnt1,  ucnt2, ucnt3, ucnt4, ucnt5, ucnt6, ucnt7, ucnt8, ucnt9, ucnt10,
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
from  user u
join user_active_history30 a on u.uid=a.uid
join brand_price b on u.brand=b.brand
join user_act_h30 ua on u.uid=ua.uid;
''',
                 '''select 
                     case 
  when u.brand like '小米MAX%' then '小米MAX'
  when u.brand like '小米Mix%' then '小米Mix'
  when u.brand like '小米Note%' then '小米Note'
  when u.brand like '小米手机%' then '小米手机'
  when u.brand like '红米Note%' then '红米Note'
  when u.brand like '红米K%' then '红米Kill'
  when u.brand like '红米手机%' or u.brand in ('红米Pro','红米S2') then '红米手机'
  else u.brand end as brand_type,
    modelname,
  version,  user_degree,
  resident_city_type,
CAST(total_use_days AS DECIMAL(4)) tdays,
 CAST(user_age AS DECIMAL(4)) uage,
  CAST(user_sex AS DECIMAL(4)) usex,
  CAST(age AS DECIMAL(4))realage,
    CAST(SUBSTRING_INDEX(price, '-', 1) AS UNSIGNED) min_price,
     CAST(SUBSTRING_INDEX(price, '-', -1) AS UNSIGNED) max_price,
    ucnt1,  ucnt2, ucnt3, ucnt4, ucnt5, ucnt6, ucnt7, ucnt8, ucnt9, ucnt10,
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
 join user_active_history30_test a on u.uid=a.uid
 join brand_price b on u.brand=b.brand
 join user_act_h30_test ua on u.uid=ua.uid;
''',
                 mfile= 'model_u_qcy_price_min_max_cnt10_a30_str',
                 res_file= "xgb_model_u_qcy_price_min_max_cnt10_a30_str"
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


