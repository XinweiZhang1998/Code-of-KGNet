# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 16:21:54 2020

@author: SEUer
"""
#量化-双门限量化方法

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

y_test=np.load('E:\\Spyder\\NN_predict\\Data_CVNet\\y_validation_9.npy')
y_test_predict=np.load('E:\\Spyder\\NN_predict\\Data_CVNet\\y_predict_validation_9.npy')

#y_test=np.load('E:\\Spyder\\NN_predict\\Data_paper_new\\y_validation_70.npy')
#y_test_predict=np.load('E:\\Spyder\\NN_predict\\Data_paper_new\\y_validation_predict_70.npy')

validation_number=len(y_test)
nmse_all=np.zeros(validation_number)
for i in range(validation_number):
    e1=np.square(np.linalg.norm(y_test[i,:]-y_test_predict[i,:], ord=2, axis=None, keepdims=False))
    e2=np.square(np.linalg.norm(y_test[i,:], ord=2, axis=None, keepdims=False))
    nmse_all[i]=(e1/e2)
nmse_average=np.mean(nmse_all)

length=len(y_test)
b=np.zeros((length,128))
for i in range(length):
    b[i,:] = sorted(y_test_predict[i,:],reverse=True)
#test数据为2k组时
"""
y_test=y_test[8000:10000]
y_test_predict=y_test_predict[8000:10000]
"""
#删除NN2预测duty组
"""
i_delete=np.load('E:\\Spyder\\Filter\\Filter_data23\\y2_test_predict_2.npy')
#i_delete=np.load('E:\\Spyder\\Filter\\Filter_data2\\y2_test_predict_2.npy')

m=0
i_delete1=[]
for i in range(10000):
    if i_delete[i]>0.5:
        m=m+1
    else:
        i_delete1.append(i)
        
i_delete1=np.unique(i_delete1)

#y2_test=np.delete(y2_test,i_delete1_fin,axis=0)
#x2_test=np.delete(x2_test,i_delete1_fin,axis=0)
"""

x_quantify=np.zeros((length,128))
y_quantify=np.zeros((length,128))

bad_point=np.zeros(length)

quan_factor=0.3  #量化因子
for i in range(length):
    
    q_up=np.mean(y_test[i,:])+quan_factor*np.std(y_test[i,:])
    q_down=np.mean(y_test[i,:])-quan_factor*np.std(y_test[i,:])
   
    q_up_1=np.mean(y_test_predict[i,:])+quan_factor*np.std(y_test_predict[i,:])
    q_down_1=np.mean(y_test_predict[i,:])-quan_factor*np.std(y_test_predict[i,:])
    
    for j in range(128):
        if y_test[i,j]<q_down:
            x_quantify[i,j]=0
        elif y_test[i,j]>q_up:
            x_quantify[i,j]=1
        else:
            x_quantify[i,j]=-1
            
        if y_test_predict[i,j]<q_down_1:
            y_quantify[i,j]=0
        elif y_test_predict[i,j]>q_up_1:
            y_quantify[i,j]=1
        else:
            y_quantify[i,j]=-1
            
        if (x_quantify[i,j]==-1 or y_quantify[i,j]==-1):
            x_quantify[i,j]=y_quantify[i,j]=-1
            bad_point[i]=bad_point[i]+1

point_0=np.zeros(length)
point_1=np.zeros(length)
key_bits=np.zeros(length)
for i in range(length):
    for j in range(128):
        if x_quantify[i,j]==0:
            point_0[i] =point_0[i] + 1
        elif x_quantify[i,j]==1:
            point_1[i] =point_1[i] + 1
        key_bits[i]=point_0[i]+point_1[i]


        
        
        
ber=np.zeros(length)    
for k in range(length):
    for m in range(128):
        if x_quantify[k,m] != y_quantify[k,m]:
            ber[k]=ber[k]+1
#ber1=np.delete(ber,i_delete1,axis=0)
#n=10000-len(i_delete1)
ber1=ber
n=length

Key_Ber_average=np.sum(ber1)/np.sum(key_bits)

Key_bits_average = np.sum(key_bits)/(n*64)
"""
Key_bits_average = np.sum(key_bits)/(n*128
m=0
y_select=np.zeros(10000)
for k in range(10000):
    if ber[k]<51:
        y_select[k]=1
        m=m+1
"""
        