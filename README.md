# predict_travel

import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.model_selection import cross_val_score
from sklearn import ensemble
from sklearn.preprocessing import LabelEncoder
import re
%matplotlib inline


action_train=pd.read_csv(r'E:\Program_Files\datacastal\practice\huangbaoche\huangbaoche_match\data\trainingset\action_train.csv')
orderFuture_train=pd.read_csv(r'E:\Program_Files\datacastal\practice\huangbaoche\huangbaoche_match\data\trainingset\orderFuture_train.csv')
orderHistory_train=pd.read_csv(r'E:\Program_Files\datacastal\practice\huangbaoche\huangbaoche_match\data\trainingset\orderHistory_train.csv')
userComment_train=pd.read_csv(r'E:\Program_Files\datacastal\practice\huangbaoche\huangbaoche_match\data\trainingset\userComment_train.csv')
userProfile_train=pd.read_csv(r'E:\Program_Files\datacastal\practice\huangbaoche\huangbaoche_match\data\trainingset\userProfile_train.csv')
action_train2=pd.read_csv(r'E:\Program_Files\datacastal\practice\huangbaoche\huangbaoche_match\data\trainingset\action_train2.csv')


action_test=pd.read_csv(r'E:\Program_Files\datacastal\practice\huangbaoche\huangbaoche_match\data\test\action_test.csv')
orderFuture_test=pd.read_csv(r'E:\Program_Files\datacastal\practice\huangbaoche\huangbaoche_match\data\test\orderFuture_test.csv')
orderHistory_test=pd.read_csv(r'E:\Program_Files\datacastal\practice\huangbaoche\huangbaoche_match\data\test\orderHistory_test.csv')
userComment_test=pd.read_csv(r'E:\Program_Files\datacastal\practice\huangbaoche\huangbaoche_match\data\test\userComment_test.csv')
userProfile_test=pd.read_csv(r'E:\Program_Files\datacastal\practice\huangbaoche\huangbaoche_match\data\test\userProfile_test.csv')
action_test2=pd.read_csv(r'E:\Program_Files\datacastal\practice\huangbaoche\huangbaoche_match\data\test\action_test2.csv')


##########对训练集构建特征工程#############
####构建新特征变量orderType1，曾经购买过精品服务的设为1，其余设为0
orderHistory_train1=orderHistory_train.drop_duplicates(['userid'])
train_ordertype_number=[]
for i in range(len(orderHistory_train1.index)):
    count_number=list(orderHistory_train['userid']).count(orderHistory_train.loc[orderHistory_train1.index[i],'userid'])
    count_1_number=list(orderHistory_train.loc[orderHistory_train1.index[i]:orderHistory_train1.index[i]+count_number-1,'orderType']).count(1)
    train_ordertype_number.append(count_1_number)
orderHistory_train1['orderType1']=train_ordertype_number

####构建新特征变量，每个用户使用app的次数operation_number和总时长operation_time和1到9的点击次数
action_train1=action_train.drop_duplicates('userid')
number=[]
time_longth=[]
for i in range(len(action_train1.index)-1):
    count_number=action_train1.index[i+1]-action_train1.index[i]
    number.append(count_number)
    time=action_train.loc[action_train1.index[i+1]-1,'actionTime']-action_train.loc[action_train1.index[i],'actionTime']
    time_longth.append(time)
number.append(len(action_train.index)-action_train1.index[-1])
time_longth.append(action_train.loc[action_train.index[-1],'actionTime']-action_train.loc[action_train1.index[-1],'actionTime'])
action_train1['operation_number']=number
action_train1['operation_time']=time_longth
action_train1.drop(['actionType','actionTime'],axis=1,inplace=True)

action_train['actionType']=action_train['actionType'].astype('object')
action_train2=pd.get_dummies(action_train)
action_train3=action_train.drop_duplicates(['userid'])
train_actionType_1_number=[]
train_actionType_2_number=[]
train_actionType_3_number=[]
train_actionType_4_number=[]
train_actionType_5_number=[]
train_actionType_6_number=[]
train_actionType_7_number=[]
train_actionType_8_number=[]
train_actionType_9_number=[]
for i in range(len(action_train3.index)):
    each_actionType_1=action_train2[action_train2['userid']==action_train2.loc[action_train3.index[i],'userid']]['actionType_1'].sum()
    each_actionType_2=action_train2[action_train2['userid']==action_train2.loc[action_train3.index[i],'userid']]['actionType_2'].sum()
    each_actionType_3=action_train2[action_train2['userid']==action_train2.loc[action_train3.index[i],'userid']]['actionType_3'].sum()
    each_actionType_4=action_train2[action_train2['userid']==action_train2.loc[action_train3.index[i],'userid']]['actionType_4'].sum()
    each_actionType_5=action_train2[action_train2['userid']==action_train2.loc[action_train3.index[i],'userid']]['actionType_5'].sum()
    each_actionType_6=action_train2[action_train2['userid']==action_train2.loc[action_train3.index[i],'userid']]['actionType_6'].sum()
    each_actionType_7=action_train2[action_train2['userid']==action_train2.loc[action_train3.index[i],'userid']]['actionType_7'].sum()
    each_actionType_8=action_train2[action_train2['userid']==action_train2.loc[action_train3.index[i],'userid']]['actionType_8'].sum()
    each_actionType_9=action_train2[action_train2['userid']==action_train2.loc[action_train3.index[i],'userid']]['actionType_9'].sum()
    each_sum=each_actionType_1+each_actionType_2+each_actionType_3+each_actionType_4+each_actionType_5+each_actionType_6+each_actionType_7+each_actionType_8+each_actionType_9
    train_actionType_1_number.append(each_actionType_1/each_sum)
    train_actionType_2_number.append(each_actionType_2/each_sum)
    train_actionType_3_number.append(each_actionType_3/each_sum)
    train_actionType_4_number.append(each_actionType_4/each_sum)
    train_actionType_5_number.append(each_actionType_5/each_sum)
    train_actionType_6_number.append(each_actionType_6/each_sum)
    train_actionType_7_number.append(each_actionType_7/each_sum)
    train_actionType_8_number.append(each_actionType_8/each_sum)
    train_actionType_9_number.append(each_actionType_9/each_sum)
action_train3['orderType_1']=train_actionType_1_number
action_train3['orderType_2']=train_actionType_2_number
action_train3['orderType_3']=train_actionType_3_number
action_train3['orderType_4']=train_actionType_4_number
action_train3['orderType_5']=train_actionType_5_number
action_train3['orderType_6']=train_actionType_6_number
action_train3['orderType_7']=train_actionType_7_number
action_train3['orderType_8']=train_actionType_8_number
action_train3['orderType_9']=train_actionType_9_number
action_train3.drop(['actionType','actionTime'],axis=1,inplace=True)

train1=pd.merge(userProfile_train,userComment_train,how='outer',on=['userid'])
train2=pd.merge(train1,orderHistory_train1,how='outer',on=['userid'])
train2.drop(['orderid_x','tags','commentsKeyWords','orderid_y','orderTime','city','country','continent','orderType'],axis=1,inplace=True)
train3=pd.merge(train2,orderFuture_train,how='outer',on=['userid'])
train3=pd.merge(train3,action_train1,how='outer',on=['userid'])
train3=pd.merge(train3,action_train3,how='outer',on=['userid'])
train3.info()

#对测试集构建特征工程
orderHistory_test1=orderHistory_test.drop_duplicates(['userid'])
test_ordertype_number=[]
for i in range(len(orderHistory_test1.index)):
    count_number=list(orderHistory_test['userid']).count(orderHistory_test.loc[orderHistory_test1.index[i],'userid'])
    count_1_number=list(orderHistory_test.loc[orderHistory_test1.index[i]:orderHistory_test1.index[i]+count_number-1,'orderType']).count(1)
    test_ordertype_number.append(count_1_number)
orderHistory_test1['orderType1']=test_ordertype_number

action_test1=action_test.drop_duplicates('userid')
number=[]
time_longth=[]
for i in range(len(action_test1.index)-1):
    count_number=action_test1.index[i+1]-action_test1.index[i]
    number.append(count_number)
    time=action_test.loc[action_test1.index[i+1]-1,'actionTime']-action_test.loc[action_test1.index[i],'actionTime']
    time_longth.append(time)
number.append(len(action_test.index)-action_test1.index[-1])
time_longth.append(action_test.loc[action_test.index[-1],'actionTime']-action_test.loc[action_test1.index[-1],'actionTime'])
action_test1['operation_number']=number
action_test1['operation_time']=time_longth
action_test1.drop(['actionType','actionTime'],axis=1,inplace=True)

action_test['actionType']=action_test['actionType'].astype('object')
action_test2=pd.get_dummies(action_test)
action_test3=action_test.drop_duplicates(['userid'])
test_actionType_1_number=[]
test_actionType_2_number=[]
test_actionType_3_number=[]
test_actionType_4_number=[]
test_actionType_5_number=[]
test_actionType_6_number=[]
test_actionType_7_number=[]
test_actionType_8_number=[]
test_actionType_9_number=[]
for i in range(len(action_test3.index)):
    each_actionType_1=action_test2[action_test2['userid']==action_test2.loc[action_test3.index[i],'userid']]['actionType_1'].sum()
    each_actionType_2=action_test2[action_test2['userid']==action_test2.loc[action_test3.index[i],'userid']]['actionType_2'].sum()
    each_actionType_3=action_test2[action_test2['userid']==action_test2.loc[action_test3.index[i],'userid']]['actionType_3'].sum()
    each_actionType_4=action_test2[action_test2['userid']==action_test2.loc[action_test3.index[i],'userid']]['actionType_4'].sum()
    each_actionType_5=action_test2[action_test2['userid']==action_test2.loc[action_test3.index[i],'userid']]['actionType_5'].sum()
    each_actionType_6=action_test2[action_test2['userid']==action_test2.loc[action_test3.index[i],'userid']]['actionType_6'].sum()
    each_actionType_7=action_test2[action_test2['userid']==action_test2.loc[action_test3.index[i],'userid']]['actionType_7'].sum()
    each_actionType_8=action_test2[action_test2['userid']==action_test2.loc[action_test3.index[i],'userid']]['actionType_8'].sum()
    each_actionType_9=action_test2[action_test2['userid']==action_test2.loc[action_test3.index[i],'userid']]['actionType_9'].sum()
    each_sum=each_actionType_1+each_actionType_2+each_actionType_3+each_actionType_4+each_actionType_5+each_actionType_6+each_actionType_7+each_actionType_8+each_actionType_9
    test_actionType_1_number.append(each_actionType_1/each_sum)
    test_actionType_2_number.append(each_actionType_2/each_sum)
    test_actionType_3_number.append(each_actionType_3/each_sum)
    test_actionType_4_number.append(each_actionType_4/each_sum)
    test_actionType_5_number.append(each_actionType_5/each_sum)
    test_actionType_6_number.append(each_actionType_6/each_sum)
    test_actionType_7_number.append(each_actionType_7/each_sum)
    test_actionType_8_number.append(each_actionType_8/each_sum)
    test_actionType_9_number.append(each_actionType_9/each_sum)
action_test3['orderType_1']=test_actionType_1_number
action_test3['orderType_2']=test_actionType_2_number
action_test3['orderType_3']=test_actionType_3_number
action_test3['orderType_4']=test_actionType_4_number
action_test3['orderType_5']=test_actionType_5_number
action_test3['orderType_6']=test_actionType_6_number
action_test3['orderType_7']=test_actionType_7_number
action_test3['orderType_8']=test_actionType_8_number
action_test3['orderType_9']=test_actionType_9_number
action_test3.drop(['actionType','actionTime'],axis=1,inplace=True)

test1=pd.merge(userProfile_test,userComment_test,how='outer',on=['userid'])
test2=pd.merge(test1,orderHistory_test1,how='outer',on=['userid'])
test2.drop(['orderid_x','tags','commentsKeyWords','orderid_y','orderTime','city','country','continent','orderType'],axis=1,inplace=True)
test3=pd.merge(test2,orderFuture_test,how='outer',on=['userid'])
test3=pd.merge(test3,action_test1,how='outer',on=['userid'])
test3=pd.merge(test3,action_test3,how='outer',on=['userid'])
test3.info()

#########数据清洗#########
train3['rating'].fillna(train3['rating'].mode()[0],inplace=True)
train3['orderType1'].fillna(train3['orderType1'].mode()[0],inplace=True)
train3.drop(['userid','gender','province','age'],axis=1,inplace=True)
test3['rating'].fillna(test3['rating'].mode()[0],inplace=True)
test3['orderType1'].fillna(test3['orderType1'].mode()[0],inplace=True)
test3.drop(['userid','gender','province','age'],axis=1,inplace=True)

###RandomForest######
rf=ensemble.RandomForestClassifier(50)
#features=['rating','orderType1','operation_number']
features=list(train3.columns)
features.remove('orderType')
X=train3[features]
print(X)
y=train3['orderType']
model=rf.fit(X,y)
scores=cross_val_score(rf,X,y,cv=5,scoring='accuracy')
print(np.mean(scores))

X=test3[features]
y=model.predict(X)
np.savetxt('E:\Program_Files\datacastal\practice\huangbaoche\huangbaoche_match\submission3.csv',np.c_[orderFuture_test['userid'],y],delimiter=',',header='userid,orderType',comments='',fmt='%d')


########XGBoost########
import xgboost as xgb
import time
from sklearn.model_selection import train_test_split
#feature engineering

train=train3
train_xy,val=train_test_split(train,test_size=0.2,random_state=1)
y=train_xy['orderType']
X=train_xy.drop(['orderType'],axis=1)
val_y=val['orderType']
val_x=val.drop(['orderType'],axis=1)
xgb_train=xgb.DMatrix(X,label=y)
xgb_val=xgb.DMatrix(val_x,label=val_y)
xgb_test=xgb.DMatrix(test3)

#设置模型参数，建立XGB模型
start_time=time.time()

params={
'booster':'gbtree',
'objective': 'multi:softmax', #多分类的问题
'num_class':2, # 类别数，与 multisoftmax 并用
'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
'max_depth':12, # 构建树的深度，越大越容易过拟合
'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
'subsample':0.7, # 随机采样训练样本
'colsample_bytree':0.7, # 生成树时进行的列采样
'min_child_weight':3,  
'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
'eta': 0.007, # 如同学习率
'seed':1000,
'nthread':2,# cpu 线程数
#'eval_metric': 'auc'
}

plst=list(params.items())
number_rounds=1000
watch_list=[(xgb_train,'train'),(xgb_val,'val')]
model=xgb.train(plst,xgb_train,number_rounds,watch_list,early_stopping_rounds=100)
model.save_model(r'E:\Program_Files\datacastal\practice\huangbaoche\huangbaoche_match\xgb.model')
          
#预测
print("best best_ntree_limit",model.best_ntree_limit )

print ("跑到这里了model.predict")
preds = model.predict(xgb_test,ntree_limit=model.best_ntree_limit)

np.savetxt('E:\Program_Files\datacastal\practice\huangbaoche\huangbaoche_match\submission4.csv',np.c_[orderFuture_test['userid'],preds],delimiter=',',header='userid,orderType',comments='',fmt='%d')

#输出运行时长
cost_time = time.time()-start_time
print ("xgboost success!",'\n',"cost time:",cost_time,"(s)")
