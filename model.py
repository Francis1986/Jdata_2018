'''
Created on 2018年6月11日

@author: Francis
'''
import pandas as pd
import numpy as np
import lightgbm as lgb
import time
from datetime import datetime, timedelta


def predict_lgb_S1(X_train,X_test,Y_train,pre_col):
    #S1使用LightGBM回归订单数的方式来进行预测，比分类的效果更好
    param = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2'},
    'num_leaves': 40,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
     }
    num_boost_round = 500
    lgb_train = lgb.Dataset(X_train,Y_train)
    print ("GBDT开始训练购买标签数据，请稍后... ...")
    gbm=lgb.train(param,lgb_train,num_boost_round=num_boost_round)
    print ("GBDT购买标签数据训练完成，开始购买标签预测，请稍后... ...")
    Y_pred = gbm.predict(X_test,num_iteration=gbm.best_iteration)
    fea_imp_S1 = pd.Series(gbm.feature_importance(),pre_col).sort_values(ascending=False)
    return Y_pred,fea_imp_S1


def predict_lgb_S2(X_train,X_test,Y_train,pre_col):
    #S2使用LightGBM回归的方式进行预测
    param = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2'},
    'num_leaves': 40,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
     }
    num_boost_round = 500
    lgb_train = lgb.Dataset(X_train,Y_train)
    print ("GBDT开始训练日期，请稍后... ...")
    gbm=lgb.train(param,lgb_train,num_boost_round=num_boost_round)
    print ("GBDT日期训练完成，开始预测日期，请稍后... ...")
    Y_pred = gbm.predict(X_test,num_iteration=gbm.best_iteration)
    print ("GBDT日期预测完成")
    Y_pred = np.round(Y_pred)
    fea_imp_S2 = pd.Series(gbm.feature_importance(), pre_col).sort_values(ascending=False)
    return Y_pred,fea_imp_S2


def score(Y_pred_S1,Y_pred_S2,Y_test_S1,Y_test_S2):
    score=pd.DataFrame()
    score['pre_proba']=pd.Series(Y_pred_S1)
    score['true_buy']=pd.Series(Y_test_S1)
    score['pre_date']=pd.Series(Y_pred_S2)
    score['true_date']=pd.Series(Y_test_S2)    
    score.ix[score['true_buy']!=0,"true_buy"]=1 #原来的标签是订单数（0,3），实际标签是买没买，转换一下
    score.ix[score['pre_date']<=0,"pre_date"]=1 #day小于0或者等于0的就记为1号
    score.sort_values(by = 'pre_proba',ascending=False,inplace=True)
    score=score.iloc[0:50000,:]
 
    score['id'] = np.arange(1,50001,1)
    #计算S1
    score['weight']= 1/(1+np.log(score['id']))    
    score['weight_2'] = score['weight']*score['true_buy']
    S1 = score['weight_2'].sum()/score['weight'].sum()
    print ("S1得分为：",S1)
    #计算S2
    score=score[score['true_date']!=0]
    score['fu']=10/(10+(score['pre_date']-score['true_date'])**2)
    len(Y_test_S2[Y_test_S2!=0])
    Fu = score['fu'].sum()
    Ur = len(Y_test_S2[Y_test_S2!=0])#答案用户集合
    S2 = Fu/Ur
    print ("S2得分为：",S2)
    return S1,S2


def feat_arange(wash_on,file_name):
    time_1 = time.clock()
    print ('开始读取特征，请稍后...')
    fea = pd.read_csv('./feature/'+file_name+'.csv')
    time_2 = time.clock()
    print ('完成特征读取！用时',time_2-time_1,'s')    
    fea_test = fea[fea['label_bool']==-1]
    if wash_on:
        fea_train = fea.ix[(fea['label_bool']!=-1)&(fea['label_wash']!=0),:]
    else:
        fea_train = fea.ix[fea['label_bool']!=-1,:]
    fea = []#释放内存
    
    drop_column = ['user_id','sex','age','label_wash','label_date','label_o_num','label_bool','label_sku_num']
    X = fea_train.drop(drop_column,axis=1).values
    pre_col = fea_train.drop(drop_column,axis=1).columns
    Y = fea_train[['label_o_num','label_date']].values
    fea_train = []#释放内存
    X_predict = fea_test.drop(drop_column,axis=1).values
    fea_test = []#释放内存
    print("X:",X.shape)
    print('Y:',Y.shape)
    print("X_predict:",X_predict.shape)
    return X,Y,X_predict,pre_col



def train(X,Y,pre_col):
    #使用四月用户的购买情况作为测试集
    X_test = X[0:99446]
    Y_test_S1 = Y[0:99446,0]
    Y_test_S2 = Y[0:99446,1]
    #使用其他的作为训练集
    X_train = X[99446::]
    Y_train_S1 = Y[99446::,0]
    Y_train_S2 = Y[99446::,1]
    print ('开始训练！')
    time_0 = time.clock()
    Y_pred_S1,fea_imp_S1 = predict_lgb_S1(X_train,X_test,Y_train_S1,pre_col)
    time_1 = time.clock()
    print("购买标签训练和预测用时：",time_1-time_0,"s")
    Y_pred_S2,fea_imp_S2 = predict_lgb_S2(X_train,X_test,Y_train_S2,pre_col)
    time_2 = time.clock()
    print("日期训练和预测用时：",time_2-time_1,"s")
    S1,S2 = score(Y_pred_S1,Y_pred_S2,Y_test_S1,Y_test_S2)
    S = 0.4*S1+0.6*S2
    print ("总得分为：",S)
    print("总用时",time_2-time_0,"s")
    return S1,S2,fea_imp_S1,fea_imp_S2

def submit(X,Y,X_predict,pre_col):
    Y_train_S1 = Y[:,0]
    Y_train_S2 = Y[:,1]
    Y_pred_S1,fea_imp_s1 = predict_lgb_S1(X,X_predict,Y_train_S1,pre_col)
    Y_pred_S2,fea_imp_s2 = predict_lgb_S2(X,X_predict,Y_train_S2,pre_col) 
    submit = pd.DataFrame({'user_id':np.arange(1,99447,1)})
    submit['proba']=pd.Series(Y_pred_S1)
    submit['pred_date']=np.round(pd.Series(Y_pred_S2))
    submit.ix[submit['pred_date']<1,'pred_date']=1
    submit['pred_date'] = submit['pred_date'].map(lambda day: datetime(2017,9,1)+timedelta(days=np.round(day-1)))
    submit.sort_values(by = 'proba',ascending=False,inplace=True)
    submit=submit.iloc[0:50000,:].drop(['proba'], axis=1)
    return submit,fea_imp_s1,fea_imp_s2
