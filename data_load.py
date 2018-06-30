'''
Created on 2018年6月6日

@author: Francis
'''
import pandas as pd
def data_load(dummy_on):    
    jdata_user_basic_info = pd.read_csv('./data/jdata_user_basic_info.csv')
    jdata_sku_basic_info = pd.read_csv('./data/jdata_sku_basic_info.csv')
    jdata_user_order = pd.read_csv('./data/jdata_user_order.csv')
    jdata_user_action = pd.read_csv('./data/jdata_user_action.csv')
    jdata_user_comment_score = pd.read_csv('./data/jdata_user_comment_score.csv')

    # 转换成pandas日期格式
    jdata_user_order['o_date']=pd.to_datetime(jdata_user_order['o_date'])
    jdata_user_action['a_date']=pd.to_datetime(jdata_user_action['a_date'])
    jdata_user_comment_score['c_date']=pd.to_datetime(jdata_user_comment_score['comment_create_tm'])
    jdata_user_comment_score = jdata_user_comment_score.drop('comment_create_tm',axis=1)

    #jdata_user_order/jdata_user_action/jdata_user_comment_score将日期精细化
    jdata_user_order['o_date_y'] = jdata_user_order['o_date'].dt.year
    jdata_user_order['o_date_m'] = jdata_user_order['o_date'].dt.month
    jdata_user_order['o_date_d'] = jdata_user_order['o_date'].dt.day

   
    jdata_user_action['a_date_y'] = jdata_user_action['a_date'].dt.year
    jdata_user_action['a_date_m'] = jdata_user_action['a_date'].dt.month
    jdata_user_action['a_date_d'] = jdata_user_action['a_date'].dt.day
    
    jdata_user_comment_score['c_date_y'] = jdata_user_comment_score['c_date'].dt.year
    jdata_user_comment_score['c_date_m'] = jdata_user_comment_score['c_date'].dt.month
    jdata_user_comment_score['c_date_d'] = jdata_user_comment_score['c_date'].dt.day
    
    
    comments_1 = jdata_user_comment_score.ix[jdata_user_comment_score['score_level']==1,:].groupby(['o_id']).agg({'score_level':'count'})
    comments_2 = jdata_user_comment_score.ix[jdata_user_comment_score['score_level']==2,:].groupby(['o_id']).agg({'score_level':'count'})
    comments_3 = jdata_user_comment_score.ix[jdata_user_comment_score['score_level']==3,:].groupby(['o_id']).agg({'score_level':'count'})
    comments_1.columns = ['score_level_1_count']
    comments_2.columns = ['score_level_2_count']
    comments_3.columns = ['score_level_3_count']
    comments_trans = pd.concat([comments_1,comments_2,comments_3],axis=0)
    comments_trans=comments_trans.fillna(0)
    comments_trans['o_id']=comments_trans.index

    #jdata_user_order加入cate和sku_id信息 ,评论信息
    jdata_user_order = jdata_user_order.merge(jdata_sku_basic_info,on = "sku_id",how='left')
    jdata_user_order = jdata_user_order.merge(comments_trans,on='o_id',how='left')
    jdata_user_order = jdata_user_order.fillna(0)
    jdata_user_action = jdata_user_action.merge(jdata_sku_basic_info,on = "sku_id",how='left')
    jdata_user_order['cost'] = jdata_user_order['price']*jdata_user_order['o_sku_num']
    
    if dummy_on:
        jdata_user_basic_info[['sex','age','user_lv_cd']]=jdata_user_basic_info[['sex','age','user_lv_cd']].astype('str')
        jdata_user_basic_info = pd.get_dummies(jdata_user_basic_info,prefix=['sex','age','user_lv_cd'])
    return jdata_user_order,jdata_user_action,jdata_user_basic_info