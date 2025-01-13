# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 10:56:44 2020

@author: SEUer
"""

#H2部分+一个固定的H3-->H2全部


import tensorflow as tf
import numpy as np
from scipy.io import loadmat as load
from sklearn import preprocessing
from matplotlib import pyplot as plt
import os
from keras import backend as K
import math
#构造数据集

x_data_train = load('E:\\matlab\\FDD_key_Generation\\data_paper\\H1_1.mat')
y_data_train = load('E:\\matlab\\FDD_key_Generation\\data_paper\\H2_1.mat')
x_train_1=x_data_train['H1']   #此时读取的就是numpy对象
y_train_1=y_data_train['H2']
"""
x_data_train1 = load('E:\\matlab\\FDD_key_Generation\\data_paper\\NoiseData\\H1_40db.mat')
y_data_train1 = load('E:\\matlab\\FDD_key_Generation\\data_paper\\NoiseData\\H2_40db.mat')
x_train_1=x_data_train1['H1_40db']   #此时读取的就是numpy对象
y_train_1=y_data_train1['H2_40db']
"""

x_data_train2 = load('E:\\matlab\\FDD_key_Generation\\data_paper\\NoiseData\\H1_30db.mat')
y_data_train2 = load('E:\\matlab\\FDD_key_Generation\\data_paper\\NoiseData\\H2_30db.mat')
x_train_2=x_data_train2['H1_30db']   #此时读取的就是numpy对象
y_train_2=y_data_train2['H2_30db']
x_data_train3 = load('E:\\matlab\\FDD_key_Generation\\data_paper\\NoiseData\\H1_20db.mat')
y_data_train3 = load('E:\\matlab\\FDD_key_Generation\\data_paper\\NoiseData\\H2_20db.mat')
x_train_3=x_data_train3['H1_20db']   #此时读取的就是numpy对象
y_train_3=y_data_train3['H2_20db']
x_data_train4 = load('E:\\matlab\\FDD_key_Generation\\data_paper\\NoiseData\\H1_10db.mat')
y_data_train4 = load('E:\\matlab\\FDD_key_Generation\\data_paper\\NoiseData\\H2_10db.mat')
x_train_4=x_data_train4['H1_10db']   #此时读取的就是numpy对象
y_train_4=y_data_train4['H2_10db']
"""
x_data_train5 = load('E:\\matlab\\FDD_key_Generation\\data_paper\\NoiseData\\H1_0db.mat')
y_data_train5 = load('E:\\matlab\\FDD_key_Generation\\data_paper\\NoiseData\\H2_0db.mat')
x_train_5=x_data_train5['H1_0db']   #此时读取的就是numpy对象
y_train_5=y_data_train5['H2_0db']
"""
"""
x_data_train = load('E:\\matlab\\FDD_key_Generation\\data_paper\\NoiseData\\H1_30db.mat')
y_data_train = load('E:\\matlab\\FDD_key_Generation\\data_paper\\NoiseData\\H2_20db.mat')
x_validation=x_data_train['H1_20db']   #此时读取的就是numpy对象
y_validation=y_data_train['H2_20db']
"""

np.random.seed(11)
np.random.shuffle(x_train_1)
np.random.seed(11)
np.random.shuffle(y_train_1)
"""

np.random.seed(11)
np.random.shuffle(x_train_2)
np.random.seed(11)
np.random.shuffle(y_train_2)

np.random.seed(11)
np.random.shuffle(x_train_3)
np.random.seed(11)
np.random.shuffle(y_train_3)

np.random.seed(11)
np.random.shuffle(x_train_4)
np.random.seed(11)
np.random.shuffle(y_train_4)

np.random.seed(11)
np.random.shuffle(x_train_5)
np.random.seed(11)
np.random.shuffle(y_train_5)
"""

"""
np.random.seed(11)
np.random.shuffle(x_validation)
np.random.seed(11)
np.random.shuffle(y_validation)
"""
"""
x=np.vstack((x_train_1[:16000],x_train_2[:16000],x_train_3[:16000],x_train_4[:16000],x_train_5[:16000]))
y=np.vstack((y_train_1[:16000],y_train_2[:16000],y_train_3[:16000],y_train_4[:16000],y_train_5[:16000]))

x=np.vstack((x_train_1[:40000],x_train_5[:40000]))
y=np.vstack((y_train_1[:40000],y_train_5[:40000]))
"""

x=x_train_1
y=y_train_1
np.random.seed(11)
np.random.shuffle(x)
np.random.seed(11)
np.random.shuffle(y)

x_train=x[:80000]
y_train=y[:80000]
x_validation=x[80000:]
y_validation=y[80000:]

train_number=len(x_train)
validation_number=len(x_validation)
#x_validation=x_validation[:20000]
#y_validation=y_validation[:20000]


#归一化
min_max_scaler_x = preprocessing.MinMaxScaler()
min_max_scaler_y = preprocessing.MinMaxScaler()

x_train = min_max_scaler_x.fit_transform(x_train)
y_train = min_max_scaler_y.fit_transform(y_train)
x_validation = min_max_scaler_x.transform(x_validation)
y_validation = min_max_scaler_y.transform(y_validation)


#逐层搭建网络结构

model=tf.keras.models.Sequential(
    [
        #tf.keras.layers.Flatten(kernel_regularizer=tf.keras.regularizers.l2()),
        tf.keras.layers.Reshape((16,8,1)),
        tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(256,activation='relu'),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(128,activation='sigmoid')
        #tf.keras.layers.Dense(128)
    ]
)

#配置训练方法
def NMSE(y_true,y_pred):
    mse1=K.sum(K.square(y_true-y_pred))
    mse2=K.sum(K.square(y_true))
    NMSE=mse1/mse2
    return NMSE

model.compile(
    optimizer=tf.optimizers.Adam(
        lr=0.001,
        beta_1=0.9, 
        beta_2=0.999, 
        epsilon=1e-08, 
        decay=0.0
        #decay=,
        #momentum=,
    ),
    loss='mse',
    #metrics=[BER]
    metrics=[NMSE]
)

#保存模型
"""
checkpoint_save_path="./checkpoint/mnist.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('------load the model---------')
    model.load_weights(checkpoint_save_path)
cp_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                               save_weights_only=True,
                                               monitor='val_loss',
                                               save_best_only=True)
"""
#执行训练过程
#history=model.fit(x_train,y_train, batch_size=1024, epochs=1000, validation_data=(x_validation,y_validation), 
#                validation_freq=10,callbacks=[cp_callback])
history=model.fit(x_train,y_train, batch_size=128, epochs=500, validation_data=(x_validation,y_validation), 
                validation_freq=1)

#打印网络结构
model.summary()

#输出训练参数
"""
print(model.trainable_variables)
file = open('./weights.txt','w')
for v in model.trainable_variables:
    file.write(str(v.name)+'\n')
    file.write(str(v.shape)+'\n')
    file.write(str(v.numpy())+'\n')
file.close()
"""

#训练集和测试集loss画图
loss=history.history['loss']
val_loss=history.history['val_loss']
nmse=history.history['NMSE']
val_nmse=history.history['val_NMSE']
plt.plot(nmse[1:300],label='Training')
plt.plot(val_nmse[1:300],label='Validation')


plt.plot(loss[1:300],label='Training')
plt.plot(val_loss[1:300],label='Validation')
plt.xlabel('epochs');
plt.ylabel('loss');
plt.legend()
plt.show()

#x_test1=x_test.reshape((10000,32,32,1))

"""
x_data_train = load('E:\\matlab\\FDD_key_Generation\\data_paper\\NoiseData\\H1_0db.mat')
y_data_train = load('E:\\matlab\\FDD_key_Generation\\data_paper\\NoiseData\\H2_0db.mat')
x_validation=x_data_train['H1_0db']   #此时读取的就是numpy对象
y_validation=y_data_train['H2_0db']

np.random.seed(11)
np.random.shuffle(x_validation)
np.random.seed(11)
np.random.shuffle(y_validation)

x_validation=x_validation[80000:]
y_validation=y_validation[80000:]

validation_number=len(x_validation)
x_validation = min_max_scaler_x.transform(x_validation)
y_validation = min_max_scaler_y.transform(y_validation)
"""

y_validation_predict=model.predict(x_validation)

nmse_all=np.zeros(validation_number)

for i in range(validation_number):
    e1=np.square(np.linalg.norm(y_validation[i,:]-y_validation_predict[i,:], ord=2, axis=None, keepdims=False))
    e2=np.square(np.linalg.norm(y_validation[i,:], ord=2, axis=None, keepdims=False))
    nmse_all[i]=(e1/e2)
nmse_average=np.mean(nmse_all)
score=model.evaluate(x_validation,y_validation)


#y_test_predict=model.predict(x_validation)
plt.plot(y_validation[0,:],label='Bob')
plt.plot(y_validation_predict[0,:],label='Alice')
plt.plot(y_validation[80600,:],label='Eve')
plt.xlabel('Subcarriers');
plt.ylabel('Value');
plt.legend()
plt.show()

#求eve接收到的CSI和Bob接受到的CSI的NMSE
"""
e1=np.square(np.linalg.norm(y_validation[1,:]-y_validation[80600,:], ord=2, axis=None, keepdims=False))
e2=np.square(np.linalg.norm(y_validation[1,:], ord=2, axis=None, keepdims=False))
nmse_eve=(e1/e2)
"""
loss=np.array(loss)
val_loss=np.array(val_loss)
nmse=np.array(nmse)
val_nmse=np.array(val_nmse)

"""
y_validation_predict=model.predict(x_validation)

nmse_all=np.zeros(validation_number)

for i in range(validation_number):
    e1=np.square(np.linalg.norm(y_validation[i,:]-y_validation_predict[i,:], ord=2, axis=None, keepdims=False))
    e2=np.square(np.linalg.norm(y_validation[i,:], ord=2, axis=None, keepdims=False))
    nmse_all[i]=(e1/e2)
nmse_average=np.mean(nmse_all)
score=model.evaluate(x_validation,y_validation)
"""

#y_test_predict=model.predict(x_validation)
plt.plot(y_validation[1,:],label='True')
plt.plot(y_validation_predict[1,:],label='Predict')
plt.xlabel('Subcarriers');
plt.ylabel('Value');
plt.legend()
plt.show()


loss=np.array(loss)
val_loss=np.array(val_loss)
nmse=np.array(nmse)
val_nmse=np.array(val_nmse)

