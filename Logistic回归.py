#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:zbt
# datetime:2019/3/27 21:38
# software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.model_selection import train_test_split
#得到特征和对应类别的值 存在x y中
#t0=time.clock()
dataset = fetch_20newsgroups_vectorized('all')
x=dataset.data
y=dataset.target
#划分得到训练集和测试集
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.2,random_state = 1)
#shape[0] 表示 行 ,shape[1]表示列 .shape 返回的(n,m)表示矩阵的行和列
solvers = ['sag','saga','newton-cg']
X=[]
Y=[]
print("-----------------------------------------------")
print("选择了三个solver检测分类准确度，耗时2min左右,请耐心等待")
for i in solvers:
    #t1=time.clock()
    lr = LogisticRegression(solver = i , multi_class = 'multinomial')
    lr.fit(x_train,y_train)
    y_pre=lr.predict(x_test)
    accuracy = np.sum(y_pre == y_test) / y_test.shape[0]
    #t2=time.clock()
    print("多分类模拟准确度{:.4f}".format(accuracy))
    X.append(i)
    Y.append(accuracy)


plt.title("LogisticRegression classification accuracy")
plt.plot(X,Y,'r')

plt.show()
