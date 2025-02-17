# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 19:47:04 2020

@author: SEUer
"""
#直接去除归一化后x_test和y_test超过某一范围的组

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
x_test=np.load('E:\\Spyder\\x_test_4.npy')
y_test=np.load('E:\\Spyder\\y_test_4.npy')
y_test_predict=np.load('E:\\Spyder\\y_test_predict_4.npy')

i_delete=[]
for i in range(10000):
    for j in range(1024):
        if x_test[i,j] < -0.3 or x_test[i,j] > 1.3:
            i_delete.append(i)
        if y_test[i,j] < -0.3 or y_test[i,j] > 1.3:
            i_delete.append(i)
i_delete=np.unique(i_delete)


y_test=np.delete(y_test,i_delete,axis=0)

y_test_predict=np.delete(y_test_predict,i_delete,axis=0)

n=len(y_test)

mse=np.array([])
for i in range(n):
    x=y_test[i,:]
    y=y_test_predict[i,:]
    mse1=mean_squared_error(x,y)
    mse=np.append(mse,mse1)
b=np.sum(mse)/n

x=range(1,101,1)

#x_=x_test[201,1:101]
#plt.plot(x,x_,label='x')
y=y_test[1198,1:101]
y_pre=y_test_predict[1198,1:101]

plt.plot(x,y,label='true')
plt.plot(x,y_pre,label='predict')
plt.legend()
plt.show()