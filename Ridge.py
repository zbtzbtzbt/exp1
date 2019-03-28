#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:zbt
# datetime:2019/3/27 21:11
# software: PyCharm
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
housing = fetch_california_housing()
#data 特征 target 结果
data=housing.data
target=housing.target

x_train,x_test,y_train,y_test=train_test_split(data,target,test_size = 0.2,random_state = 1)
rg=Ridge(alpha = 0.2)
#print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
#用lr拟合训练集
predict_model=rg.fit(x_train,y_train)
score=rg.score(x_test,y_test)
coef=predict_model.coef_ #直线系数
intercept=predict_model.intercept_ #截距
print("coef={},intercept={}".format(coef,intercept))
print("y={}x+{}".format(coef,intercept)) #通过模型得到的直线方程
#使用 MSE 评测模型
#获取预测值
#R2越接近1 拟合越好 MAE MSE 越接近0越好
print("--------------------------")
print("下面Ridge回归评估实验结果")
y_predict=predict_model.predict(x_test)
mse=mean_squared_error(y_test,y_predict)
print("MSE={}".format(mse))
#使用R^2评估
r2=r2_score(y_test,y_predict)
print("R^2={}".format(r2))
#使用MAE评估
mae=mean_absolute_error(y_test,y_predict)
print("MAE={}".format(mae))
print("Ridge_score={}".format(score))