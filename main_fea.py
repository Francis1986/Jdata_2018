'''
Created on 2018年6月6日

@author: Francis
'''
from data_load import data_load
from feat_gen import feat_extract
import time
import pandas as pd

def main_fea(end_date_set,time_gap,dummy_on):
    #数据读取
    time_0 = time.clock() 
    print ('开始读取数据')
    order,action,user_info = data_load(dummy_on)
    time_1 = time.clock()
    print ('数据读取完成，开始特征抽取,用时',time_1-time_0,'s')
    print ('开始提取特征')
    user_fea_concat = []
    for end_date in end_date_set:
        print(end_date)
        user_fea = feat_extract(end_date,time_gap,order,action,user_info)
        user_fea_concat.append(user_fea)  
    user_fea = pd.concat(user_fea_concat,axis=0)  
    time_2 = time.clock()
    print ('特征和标签提取完成！用时',time_2-time_1,'s')
    return user_fea
