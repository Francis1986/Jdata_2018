'''
Created on 2018��6��15��

@author: Francis
'''
from model import feat_arange
from main_fea import main_fea
from model import train,submit
import time

tran = 0
sub = 1
fea_exist = 0

###########################################################################################################################################
####特征生成
if fea_exist == 0:
    end_date_set = ['2017-09-01','2017-08-01','2017-07-01','2017-05-01','2017-04-01','2017-03-01','2017-02-01']
    time_gap={'day_num_set':[7,14],'month_num_set':[1,3,5]}
    dummy_on = 0
    user_fea= main_fea(end_date_set,time_gap,dummy_on)
    print ('正在保存特征，请稍后... ...')
    time_2 = time.clock()
    file_name = 'feature'
    user_fea.to_csv('./feature/'+file_name+'.csv',index=None)
    user_fea=[]#释放内存
    time_3 = time.clock()
    print ('保存完成！保存用时',time_3-time_2,'s')
############################################################################################################################################
##训练
if tran:
    wash_on=0
    X,Y,X_predict,pre_col = feat_arange(wash_on,file_name='feature')
    S1,S2,fea_imp_s1,fea_imp_s2 = train(X,Y,pre_col)
    fea_imp_s1.to_csv('./fea_imp_train/fea_imp_s1_train.csv')
    fea_imp_s2.to_csv('./fea_imp_train/fea_imp_s2_train.csv')
    print ('fmp保存完成！')
##################################################################################################################################################
###提交
if sub:
    wash_on=1
    X,Y,X_predict,pre_col = feat_arange(wash_on,file_name='feature')
    submit,fea_imp_s1,fea_imp_s2 = submit(X,Y,X_predict,pre_col)
    fea_imp_s1.to_csv('./fea_imp_sub/fea_imp_s1_sub.csv')
    fea_imp_s2.to_csv('./fea_imp_sub/fea_imp_s2_sub.csv')
    submit.to_csv('./submit/submit_3505.csv',index=None)
    print ('提交结果保存完成！')
#################################################################################################33