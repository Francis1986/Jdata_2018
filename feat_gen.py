'''
Created on 2018年6月6日

@author: Francis
'''
import pandas as pd
import numpy as np
from datetime import timedelta
from math import log1p

def feat_extract(end_date,time_gap,order,action,user_info):
    order['end_date'] = end_date
    action['end_date'] = end_date

    order['end_date'] = pd.to_datetime(order['end_date'])
    action['end_date'] = pd.to_datetime(action['end_date'])
    #实际购买时间和end_date相差的月数
    order['date_diff'] = order['end_date']-order['o_date']
    action['date_diff'] = action['end_date']-action['a_date']
    #实际购买时间和end_date相差的月数
    order['month_diff'] = (order['end_date'].dt.year - order['o_date_y'])*12+(order['end_date'].dt.month - order['o_date_m'])
    action['month_diff'] = (action['end_date'].dt.year - action['a_date_y'])*12+(action['end_date'].dt.month - action['a_date_m'])
    #标记清洗标签，相差3个月及以内的不清洗
    order['wash_label'] = 0
    order.ix[(order['month_diff']<=3)&(order['month_diff']>0),'wash_label'] = 1  
    #形成标签集合，标签分四类，购买时间标签，购买次数标签，是否购买标签以及清洗标签
    if end_date == "2017-09-01":
        label = pd.DataFrame({'user_id':np.arange(1,len(user_info)+1,1)})
        label.index = label.user_id
        label['label_o_num'] = -1
        label['label_sku_num'] = -1
        label['label_date'] = -1
        label['label_bool']= -1
        label['label_wash'] = -1
        label=label.drop(['user_id'],axis=1)    
    else:
        label = pd.DataFrame({'user_id':np.arange(1,len(user_info)+1,1)})
        label.index = label['user_id']
        order_m = order.ix[order['month_diff']==0,:]
        # s
        label['label_o_num'] = order_m.ix[order['cate'].isin([101,30]),:].pivot_table(index='user_id',values='o_id',aggfunc={'o_id':'nunique'})
        label['label_o_num'] = label['label_o_num'].map(log1p)
        # s
        label['label_sku_num'] = order_m.ix[order['cate'].isin([101,30]),:].pivot_table(index='user_id',values='o_sku_num',aggfunc={'o_sku_num':'sum'})
        label['label_sku_num'] = label['label_sku_num'].map(log1p)

        label['label_date'] = order_m.ix[order['cate'].isin([101,30]),:].pivot_table(index='user_id',values='o_date_d',aggfunc={'o_date_d':'min'})
        label['label_bool']=label['label_o_num']
        label.fillna(0,inplace=True)
        label.ix[label['label_bool']!=0,'label_bool']=1
        label['label_wash'] = order.ix[order['cate'].isin([101,30]),:].pivot_table(index='user_id',columns='wash_label',values='o_id',aggfunc={'o_id':'count'})[1]
        label.fillna(0,inplace=True)
        label=label.drop(['user_id'],axis=1)
        label.ix[label['label_o_num']>1.386,"label_o_num"]=1.386 #大于3的订单数极少，如果按实际的话会引入干扰
        label.ix[label['label_sku_num']>6,"label_sku_num"]=6
    user_fea_concat = []
    #生成2个字典用来存储，满足时间标签值的数据
    order_tmp_dict = {}
    action_tmp_dict = {}
    for gap_name in time_gap.keys():
        if (gap_name =='day_num_set')&(time_gap[gap_name]!=[]):
            for day_num in time_gap[gap_name]:
                #满足近期的时间标签
                order['time_label'] = 0
                order.ix[(order['date_diff']<=timedelta(days=day_num))&(order['date_diff']>timedelta(days=0)),'time_label'] = 1        
                action['time_label'] = 0
                action.ix[(action['date_diff']<=timedelta(days=day_num))&(action['date_diff']>timedelta(days=0)),'time_label'] = 1
                #过滤成满足时间标签的格式
                order_tmp = order.ix[order['time_label']==1,:]
                action_tmp = action.ix[action['time_label']==1,:] 
                order_tmp_dict[str(day_num)+'day'] = order_tmp
                action_tmp_dict[str(day_num)+'day'] = action_tmp
        if (gap_name =='month_num_set')&(time_gap[gap_name]!=[]):
            for month_num in time_gap[gap_name]:
                #满足近期的时间标签
                order['time_label'] = 0
                order.ix[(order['month_diff']<=month_num)&(order['month_diff']>0),'time_label'] = 1    
                action['time_label'] = 0
                action.ix[(action['month_diff']<=month_num)&(action['month_diff']>0),'time_label'] = 1  
                order_tmp = order.ix[order['time_label']==1,:]
                action_tmp = action.ix[action['time_label']==1,:] 
                order_tmp_dict[str(month_num)+'months'] = order_tmp
                action_tmp_dict[str(month_num)+'months'] = action_tmp
    for gap in order_tmp_dict.keys():
        print (gap)     
        #过滤成满足时间标签的格式
        order_tmp = order_tmp_dict[gap]
        action_tmp = action_tmp_dict[gap]        
        #初始化user_fea
        user_fea = pd.DataFrame(user_info['user_id'])
        user_fea.columns = ['user_id']
        user_fea.index = user_fea['user_id']
      
        #实际购买
        #商品价格 s
        #近期最商品总价，这个特征感觉不是很合理，感觉用花费的总数比较合理即price*o_sku_num
        user_fea['10130_price_sum'] = order_tmp.ix[order_tmp['cate'].isin([101,30]),:].pivot_table(index='user_id',values='price',aggfunc={'price':'sum'})
        user_fea['10130_price_sum'] = user_fea['10130_price_sum'].map(log1p)#ln(1+x)平滑
        #近期平均商品价格 s
        user_fea['10130_price_mean'] = order_tmp.ix[order_tmp['cate'].isin([101,30]),:].pivot_table(index='user_id',values='price',aggfunc={'price':'mean'})
        user_fea['10130_price_mean'] = user_fea['10130_price_mean'].map(log1p)#ln(1+x)平滑
        #近期最小商品价格 s
        user_fea['10130_price_min'] = order_tmp.ix[order_tmp['cate'].isin([101,30]),:].pivot_table(index='user_id',values='price',aggfunc={'price':'min'})
        user_fea['10130_price_min'] = user_fea['10130_price_min'].map(log1p)#ln(1+x)平滑
        #近期最大商品价格 s
        user_fea['10130_price_max'] = order_tmp.ix[order_tmp['cate'].isin([101,30]),:].pivot_table(index='user_id',values='price',aggfunc={'price':'max'})
        user_fea['10130_price_max'] = user_fea['10130_price_max'].map(log1p)#ln(1+x)平滑
        #近期订单金额中位数 s
        user_fea['10130_price_median'] = order_tmp.ix[order_tmp['cate'].isin([101,30]),:].pivot_table(index='user_id',values='price',aggfunc={'price':'median'})    
        user_fea['10130_price_median'] = user_fea['10130_price_median'].map(log1p)#ln(1+x)平滑
        #近期所有商品总消费 s
        user_fea['cost_sum'] = order_tmp.pivot_table(index='user_id',values='cost',aggfunc={'cost':'sum'})
        user_fea['cost_sum'] = user_fea['cost_sum'].map(log1p)   #ln(1+x)平滑
        #近期（101,30）商品消费 s
        user_fea['10130_cost_sum'] = order_tmp.ix[order_tmp['cate'].isin([101,30]),:].pivot_table(index='user_id',values='cost',aggfunc={'cost':'sum'}) 
        user_fea['10130_cost_sum'] = user_fea['10130_cost_sum'].map(log1p)#ln(1+x)平滑
       
        #近期所有商品para_1最大值、最小值、均值、中位数
        user_fea['para_1_max'] = order_tmp.pivot_table(index='user_id',values='para_1',aggfunc={'para_1':'max'})  
        user_fea['para_1_min'] = order_tmp.pivot_table(index='user_id',values='para_1',aggfunc={'para_1':'min'})        
        user_fea['para_1_mean'] = order_tmp.pivot_table(index='user_id',values='para_1',aggfunc={'para_1':'mean'})  
        user_fea['para_1_mdedian'] = order_tmp.pivot_table(index='user_id',values='para_1',aggfunc={'para_1':'median'})
        
        #近期101,30商品para_1最大值、最小值、均值、中位数
        user_fea['10130_para_1_max'] = order_tmp.ix[order_tmp['cate'].isin([101,30]),:].pivot_table(index='user_id',values='para_1',aggfunc={'para_1':'max'})
        user_fea['10130_para_1_min'] = order_tmp.ix[order_tmp['cate'].isin([101,30]),:].pivot_table(index='user_id',values='para_1',aggfunc={'para_1':'min'})        
        user_fea['10130_para_1_mean'] = order_tmp.ix[order_tmp['cate'].isin([101,30]),:].pivot_table(index='user_id',values='para_1',aggfunc={'para_1':'mean'})
        user_fea['10130_para_1_median'] = order_tmp.ix[order_tmp['cate'].isin([101,30]),:].pivot_table(index='user_id',values='para_1',aggfunc={'para_1':'median'})
         
        
        #近期101,30商品para_2最大值、最小值、均值、中位数
        user_fea['10130_para_2_max'] = order_tmp.ix[(order_tmp['cate'].isin([101,30]))&(order_tmp['para_2']!=-1),:].pivot_table(index='user_id',values='para_2',aggfunc={'para_2':'mean'})
        user_fea['10130_para_2_min'] = order_tmp.ix[(order_tmp['cate'].isin([101,30]))&(order_tmp['para_2']!=-1),:].pivot_table(index='user_id',values='para_2',aggfunc={'para_2':'min'})        
        user_fea['10130_para_2_mean'] = order_tmp.ix[(order_tmp['cate'].isin([101,30]))&(order_tmp['para_2']!=-1),:].pivot_table(index='user_id',values='para_2',aggfunc={'para_2':'mean'})
        user_fea['10130_para_2_median'] = order_tmp.ix[(order_tmp['cate'].isin([101,30]))&(order_tmp['para_2']!=-1),:].pivot_table(index='user_id',values='para_2',aggfunc={'para_2':'median'})
    
        
        #近期101,30商品para_3最大值、最小值、均值、中位数
        user_fea['10130_para_3_max'] = order_tmp.ix[(order_tmp['cate'].isin([101,30]))&(order_tmp['para_3']!=-1),:].pivot_table(index='user_id',values='para_3',aggfunc={'para_3':'max'})
        user_fea['10130_para_3_min'] = order_tmp.ix[(order_tmp['cate'].isin([101,30]))&(order_tmp['para_3']!=-1),:].pivot_table(index='user_id',values='para_3',aggfunc={'para_3':'min'})        
        user_fea['10130_para_3_mean'] = order_tmp.ix[(order_tmp['cate'].isin([101,30]))&(order_tmp['para_3']!=-1),:].pivot_table(index='user_id',values='para_3',aggfunc={'para_3':'mean'})
        user_fea['10130_para_3_median'] = order_tmp.ix[(order_tmp['cate'].isin([101,30]))&(order_tmp['para_3']!=-1),:].pivot_table(index='user_id',values='para_3',aggfunc={'para_3':'median'})
        
        
        #满足近期的订单数
        user_fea['order_num'] = order_tmp.pivot_table(index='user_id',values='o_id',aggfunc={'o_id':'nunique'})
        user_fea['order_num'] = user_fea['order_num'].map(log1p) #ln(1+x)平滑
        user_fea['10130_order_num'] = order_tmp.ix[order_tmp['cate'].isin([101,30]),:].pivot_table(index='user_id',values='o_id',aggfunc={'o_id':'nunique'})
        user_fea['10130_order_num'] = user_fea['10130_order_num'].map(log1p) #ln(1+x)平滑
               
        #近期购买商品的次数
        user_fea['sku_count'] = order_tmp.pivot_table(index='user_id',values='sku_id',aggfunc={'sku_id':'count'})
        user_fea['10130_sku_count'] = order_tmp.ix[order_tmp['cate'].isin([101,30]),:].pivot_table(index='user_id',values='sku_id',aggfunc={'sku_id':'count'})       
        
        #近期的商品购买数
        user_fea['sku_num'] = order_tmp.pivot_table(index='user_id',values='o_sku_num',aggfunc={'o_sku_num':'sum'})
        user_fea['10130_sku_num'] = order_tmp.ix[order_tmp['cate'].isin([101,30]),:].pivot_table(index='user_id',values='o_sku_num',aggfunc={'o_sku_num':'sum'})
    
        #近期购买的10130商品数的占比
        user_fea['10130_sku_occupation'] = user_fea['10130_sku_num']/user_fea['sku_num']
        user_fea.ix[user_fea['10130_sku_occupation'].isnull(),'10130_sku_occupation']=-1 #分子分母都是零时置为-1  
        user_fea['sku_num'] = user_fea['sku_num'].map(log1p) #ln(1+x)平滑
        user_fea['10130_sku_num'] = user_fea['10130_sku_num'].map(log1p)#ln(1+x)平滑
      
       
        #近期购买的sku商品种类数(去重)
        user_fea['sku_unique'] = order_tmp.pivot_table(index='user_id',values='sku_id',aggfunc={'sku_id':'nunique'})
        user_fea['10130_sku_unique'] = order_tmp.ix[order_tmp['cate'].isin([101,30]),:].pivot_table(index='user_id',values='sku_id',aggfunc={'sku_id':'nunique'})

        
        #近期购买的101,30 sku商品次数/SKU商品种类，这个指标可以反映用户对商品的购买集中度
        user_fea['10130_sku_focus']= user_fea['10130_sku_count']/user_fea['10130_sku_unique']        
        user_fea.ix[user_fea['10130_sku_focus']==float('inf'),'10130_sku_focus']=0#正无穷置为0
        
        user_fea['sku_count'] = user_fea['sku_count'].map(log1p)#ln(1+x)平滑
        user_fea['10130_sku_count'] = user_fea['10130_sku_count'].map(log1p)#ln(1+x)平滑
      
        
        
        #近期购买同一商品(所有商品，(101,30))的最大次数，反映用户对某一款sku的忠诚度 s
        tmp = order_tmp.ix[order_tmp['cate'].isin([101,30]),:].groupby(['user_id','sku_id'],as_index=False).agg({'o_id':'count'})
        user_fea['10130_samesku_maxcount']= tmp.pivot_table(index='user_id',values='o_id',aggfunc={'o_id':'max'})
        user_fea['10130_samesku_maxcount'] = user_fea['10130_samesku_maxcount'].map(log1p) #ln(1+x)平滑      

        #用户101.30历经的下单地点数量
        user_fea['10130_o_area_num']= order_tmp.ix[order_tmp['cate'].isin([101,30]),:].pivot_table(index='user_id',values='o_area',aggfunc={'o_area':'nunique'})
        
        #用户101.30历经的下单最多的o_area
        o_area_mode= order_tmp.ix[order_tmp['cate'].isin([101,30]),:].groupby(['user_id','o_area']).agg({'o_sku_num':'sum'}).reset_index('user_id')
        o_area_mode = o_area_mode.sort_values(by=['user_id','o_sku_num']).drop_duplicates(['user_id'],keep='last')
        o_area_mode['o_area']=o_area_mode.index
        o_area_mode.index = o_area_mode['user_id']
        user_fea['10130_o_area_mode'] = o_area_mode['o_area']
        
        #近期首次下单的day
        user_fea['o_firstday'] =order_tmp.pivot_table(index='user_id',values='o_date_d',aggfunc={'o_date_d':'min'})
        user_fea['10130_o_firstday'] =order_tmp.ix[order_tmp['cate'].isin([101,30]),:].pivot_table(index='user_id',values='o_date_d',aggfunc={'o_date_d':'min'})

        
        #近期最后一次下单的day
        user_fea['o_lastday'] = order_tmp.pivot_table(index='user_id',values='o_date_d',aggfunc={'o_date_d':'max'})
        user_fea['10130_o_lastday'] = order_tmp.ix[order_tmp['cate'].isin([101,30]),:].pivot_table(index='user_id',values='o_date_d',aggfunc={'o_date_d':'max'})

        #近期用户下单day平均值
        user_fea['o_meanday'] = order_tmp.pivot_table(index='user_id',values='o_date_d',aggfunc={'o_date_d':'mean'})
        user_fea['10130_o_meanday'] = order_tmp.ix[order_tmp['cate'].isin([101,30]),:].pivot_table(index='user_id',values='o_date_d',aggfunc={'o_date_d':'mean'})
       
        #如果时间窗口是3个月或者5个月，计算月首购买day的最大，最小，平均，中位数
        if (gap=='3months')|(gap=='4months')|(gap=='5months')|(gap=='6months')|(gap=='90day')|(gap=='180day'):
            order_tmp_mf=order_tmp.sort_values(by=["user_id",'o_date'])
            order_month_unique = pd.DataFrame(order_tmp_mf.drop_duplicates(['user_id','o_date_m'],keep='first'))
            user_fea['mon_firstday_min'] = order_month_unique.pivot_table(index='user_id',values='o_date_d',aggfunc={'o_date_d':'min'})
            user_fea['mon_firstday_max'] = order_month_unique.pivot_table(index='user_id',values='o_date_d',aggfunc={'o_date_d':'max'})
            user_fea['mon_firstday_mean'] = order_month_unique.pivot_table(index='user_id',values='o_date_d',aggfunc={'o_date_d':'mean'})
            user_fea['mon_firstday_median'] = order_month_unique.pivot_table(index='user_id',values='o_date_d',aggfunc={'o_date_d':'median'})
            
            order_tmp_mf=order_tmp.ix[order_tmp['cate'].isin([101,30]),:].sort_values(by=["user_id",'o_date'])
            order_month_unique = pd.DataFrame(order_tmp_mf.drop_duplicates(['user_id','o_date_m'],keep='first'))
            user_fea['10130_mon_firstday_min'] = order_month_unique.pivot_table(index='user_id',values='o_date_d',aggfunc={'o_date_d':'min'})
            user_fea['10130_mon_firstday_max'] = order_month_unique.pivot_table(index='user_id',values='o_date_d',aggfunc={'o_date_d':'max'})
            user_fea['10130_mon_firstday_mean'] = order_month_unique.pivot_table(index='user_id',values='o_date_d',aggfunc={'o_date_d':'mean'})
            user_fea['10130_mon_firstday_median'] = order_month_unique.pivot_table(index='user_id',values='o_date_d',aggfunc={'o_date_d':'median'})
      
        #近期有购买行为的天数 
        user_fea['order_day_num'] = order_tmp.pivot_table(index='user_id',values='o_date',aggfunc={'o_date':'nunique'})
        user_fea['10130_order_day_num'] = order_tmp.ix[order_tmp['cate'].isin([101,30]),:].pivot_table(index='user_id',values='o_date',aggfunc={'o_date':'nunique'})

      
                
        #近期有购买行为的月数
        user_fea['order_mon_num'] = order_tmp.pivot_table(index='user_id',values='o_date_m',aggfunc={'o_date_m':'nunique'})
        user_fea['10130_order_mon_num'] = order_tmp.ix[order_tmp['cate'].isin([101,30]),:].pivot_table(index='user_id',values='o_date_m',aggfunc={'o_date_m':'nunique'})
        

        #收藏和浏览行为       
        #近期的商品（收藏或浏览）的数量
        user_fea['act_sku_sum'] = action_tmp.pivot_table(index='user_id',values='a_num',aggfunc={'a_num':'sum'})
        user_fea['act_sku_sum'] = user_fea['act_sku_sum'].map(log1p)#ln(1+x)平滑  
        user_fea['10130_act_sku_sum'] = action_tmp.ix[action_tmp['cate'].isin([101,30]),:].pivot_table(index='user_id',values='a_num',aggfunc={'a_num':'sum'})
        user_fea['10130_act_sku_sum'] = user_fea['10130_act_sku_sum'].map(log1p)#ln(1+x)平滑  

        #近期的商品收藏的数量
        user_fea['col_sku_sum'] = action_tmp.ix[action_tmp['a_type']==2,:].pivot_table(index='user_id',values='a_num',aggfunc={'a_num':'sum'})
        user_fea['col_sku_sum'] = user_fea['col_sku_sum'].map(log1p)#ln(1+x)平滑  

        user_fea['10130_col_sku_sum'] = action_tmp.ix[(action_tmp['cate'].isin([101,30]))&(action_tmp['a_type']==2),:].pivot_table(index='user_id',values='a_num',aggfunc={'a_num':'sum'})
        user_fea['10130_col_sku_sum'] = user_fea['10130_col_sku_sum'].map(log1p)#ln(1+x)平滑  
      
        #近期有（浏览或收藏）行为的天数
        user_fea['act_day_num'] = action_tmp.pivot_table(index='user_id',values='a_date',aggfunc={'a_date':'nunique'})
        user_fea['10130_act_day_num'] = action_tmp.ix[action_tmp['cate'].isin([101,30]),:].pivot_table(index='user_id',values='a_date',aggfunc={'a_date':'nunique'})

        #近期有收藏行为的天数
        user_fea['col_day_num'] = action_tmp.ix[action_tmp['a_type']==2,:].pivot_table(index='user_id',values='a_date',aggfunc={'a_date':'nunique'})
        user_fea['10130_col_day_num'] = action_tmp.ix[(action_tmp['cate'].isin([101,30]))&(action_tmp['a_type']==2),:].pivot_table(index='user_id',values='a_date',aggfunc={'a_date':'nunique'})



        #近期有（浏览或收藏）行为的商品数（种类）
        user_fea['act_unique'] = action_tmp.pivot_table(index='user_id',values='sku_id',aggfunc={'sku_id':'nunique'})
        user_fea['10130_act_unique'] = action_tmp.ix[action_tmp['cate'].isin([101,30]),:].pivot_table(index='user_id',values='sku_id',aggfunc={'sku_id':'nunique'})
 
        #近期有收藏行为的商品数（种类）
        user_fea['col_unique'] = action_tmp.ix[action_tmp['a_type']==2,:].pivot_table(index='user_id',values='sku_id',aggfunc={'sku_id':'nunique'})
        user_fea['10130_col_unique'] = action_tmp.ix[(action_tmp['cate'].isin([101,30]))&(action_tmp['a_type']==2),:].pivot_table(index='user_id',values='sku_id',aggfunc={'sku_id':'nunique'})

              
        #调整
        user_fea.fillna(0,inplace=True)        
        #转化率(新加入)
        user_fea['trans_rate'] = user_fea['sku_unique']/user_fea['act_unique']
        user_fea['10130_trans_rate'] = user_fea['10130_sku_unique']/user_fea['10130_act_unique']
        
        user_fea.ix[user_fea['trans_rate']==float('inf'),'trans_rate']=-1#遇到负无穷转化成-1
        user_fea.ix[user_fea['10130_trans_rate']==float('inf'),'10130_trans_rate']=-1#遇到负无穷转化成-1
        
       
        
        #用户近期所有商品，好评，中评，差评的数量 
        order_tmp_com = order_tmp.drop_duplicates(['o_id'],keep='first')
        #好评数量
        user_fea['com_score1_count']=order_tmp_com.pivot_table(index='user_id',values = 'score_level_1_count',aggfunc={'score_level_1_count':'sum'})
        #中评数量
        user_fea['com_score2_count']=order_tmp_com.pivot_table(index='user_id',values = 'score_level_2_count',aggfunc={'score_level_2_count':'sum'})
        #差评数量
        user_fea['com_score3_count']=order_tmp_com.pivot_table(index='user_id',values = 'score_level_3_count',aggfunc={'score_level_3_count':'sum'})
     
        #用户近期（101,30）好评，中评，差评的数量
        order_tmp_com = order_tmp.ix[order_tmp['cate'].isin([101,30]),:].drop_duplicates(['o_id'],keep='first')
        #好评数量
        user_fea['10130_com_score1_count']=order_tmp_com.pivot_table(index='user_id',values = 'score_level_1_count',aggfunc={'score_level_1_count':'sum'})
        #中评数量
        user_fea['10130_com_score2_count']=order_tmp_com.pivot_table(index='user_id',values = 'score_level_2_count',aggfunc={'score_level_2_count':'sum'})
        #差评数量
        user_fea['10130_com_score3_count']=order_tmp_com.pivot_table(index='user_id',values = 'score_level_3_count',aggfunc={'score_level_3_count':'sum'})
      
        #最后调整
        user_fea = user_fea.drop(['user_id'],axis=1) #把user_id去掉，因为index就是user_id,不然pd.concat会重复
        user_fea = user_fea.add_prefix(gap+'_') #列名加上月度标签前缀
        user_fea_concat.append(user_fea)
    user_fea = pd.concat(user_fea_concat,axis=1) 
    #把用户信息扔进去 
    user_info.index = user_info['user_id']
    user_fea=user_info.merge(user_fea,right_index=True,left_index=True,how='left')
    
    #将标签集合并
    user_fea = pd.concat([user_fea,label],axis=1)
    user_fea.fillna(0,inplace=True)
    return user_fea

