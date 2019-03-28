#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:zbt
# datetime:2019/3/27 21:26
# software: PyCharm
from sklearn.datasets import fetch_california_housing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
def zbt_floatrange(start ,stop ,steps):
    return [start+float(i)*(stop-start)/(float(steps)-1) for i in range(steps)]
housing = fetch_california_housing()
#data 特征 target 结果
data=housing.data
target=housing.target

x_train,x_test,y_train,y_test=train_test_split(data,target,test_size = 0.2,random_state = 1)
lasso=linear_model.Lasso(alpha = 0.1)
#print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
#用lr拟合训练集
predict_model=lasso.fit(x_train,y_train)

#自动调参GridSearchCV
model=linear_model.Lasso()
#需要最优化的参数的取值，值为字典或者列表
para_grid = {'alpha':zbt_floatrange(0.0005,0.01,100)}
grid_search = GridSearchCV(model,para_grid,scoring = 'r2',n_jobs = -1,cv=5)
grid_result=grid_search.fit(x_train,y_train)
means = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']
alpha_x=[]
getscore_y=[]
for mean,param in zip(means,params):
    getscore_y.append(mean)
    print("%f  with:   %r" % (mean,param))

alpha_x=zbt_floatrange(0.0005,0.01,100)
getscore_y=np.array(getscore_y)
plt.plot(alpha_x,getscore_y,'r')
plt.xlabel("Alpha")
plt.ylabel("estimate score")
plt.show()
print("------------------------------")
coef=predict_model.coef_ #直线系数
intercept=predict_model.intercept_ #截距
print("coef={},intercept={}".format(coef,intercept))
print("y={}x+{}".format(coef,intercept)) #通过模型得到的直线方程
#使用 MSE 评测模型
#获取预测值
#R2越接近1 拟合越好 MAE MSE 越接近0越好
print("--------------------------")
print("下面使用LASSO评估实验结果")
y_predict=predict_model.predict(x_test)
mse=mean_squared_error(y_test,y_predict)
print("MSE={}".format(mse))
#使用R^2评估
r2=r2_score(y_test,y_predict)
print("R^2={}".format(r2))
#使用MAE评估
mae=mean_absolute_error(y_test,y_predict)
print("MAE={}".format(mae))

