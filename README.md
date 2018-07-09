## **概述**
- 赛事：[京东Jdata2018‘如期而至’用户购买时间预测](https://jdata.jd.com/html/detail.html?id=2)
- 队伍名称：STAR_BIGDATA
- 项目贡献者：
  [长离未离](https://github.com/Francis1986)
  [Euphoric0x0](https://github.com/liht1996)
- 排名：  
  A榜：33/739  
  B榜：17/137
- 赛提方案说明：
  [方案说明](https://github.com/Francis1986/Jdata_2018/blob/master/%E6%96%B9%E6%A1%88%E8%AF%B4%E6%98%8E.md)

## **代码说明文档**
#### 编程环境
- 语言：python 3.6.5
- 硬件配置：(python的lightgbm库会因cpu的不同，训练结果出现微小误差，不过我们没遇到，为以防万一还是说一下)
1. 型号：thinkpad x270 cpu：Inter(R) Core(TM) i7-6500 2.5GHz 内存：8GB 操作系统：Win7
2. 型号：MacBook Pro (Retina, 13-inch, Early 2015) cpu：i5-5257U System：mac OS 10.13.5  
注：程序在以上两个系统均可正常运行，且结果完全一致。

#### 第三方库的版本
- pandas：0.22.0 **（这个注意一下，如果版本升级到0.23.0的话无法复现，结果会有微小差异）**
- numpy：1.14.3
- lightgbm：2.1.2
- datetime：python3.6.5原生库
- math：python3.6.5原生库
- time：python3.6.5原生库

#### 代码使用说明
- 文件夹说明：  
data:原始数据  
feature:生成的特征矩阵  
fea_imp_sub:提交时特征重要性排行  fea_imp_train:训练时特征重要性排行      
submit:提交文件
- 相关参数：main_final.py中tran = 0 sub = 1 fea_exist = 0时为生成提交文件
- 程序入口：main_final.py (直接运行该文件即可)