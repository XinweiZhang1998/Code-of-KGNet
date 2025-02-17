# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:22:17 2020

@author: SEUer
"""
#H的预测


import tensorflow as tf
import numpy as np
from scipy.io import loadmat as load
from sklearn import preprocessing
from matplotlib import pyplot as plt
import os
from keras import backend as K
import math
import datetime
#构造数据集

x_data_train = load('E:\\matlab\\FDD_key_Generation\\data_paper\\H1_1.mat')
y_data_train = load('E:\\matlab\\FDD_key_Generation\\data_paper\\H2_1.mat')
x_train_1=x_data_train['H1']   #此时读取的就是numpy对象
y_train_1=y_data_train['H2']
"""
x_data_train = load('E:\\matlab\\FDD_key_Generation\\data_paper\\NoiseData\\H1_40db.mat')
y_data_train = load('E:\\matlab\\FDD_key_Generation\\data_paper\\NoiseData\\H2_40db.mat')
x_train_1=x_data_train['H1_40db']   #此时读取的就是numpy对象
y_train_1=y_data_train['H2_40db']
"""
x_data_train = load('E:\\matlab\\FDD_key_Generation\\data_paper\\NoiseData\\H1_0db.mat')
y_data_train = load('E:\\matlab\\FDD_key_Generation\\data_paper\\NoiseData\\H2_0db.mat')
x_train_2=x_data_train['H1_0db']   #此时读取的就是numpy对象
y_train_2=y_data_train['H2_0db']


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
np.random.seed(11)
np.random.shuffle(x_train_2)
np.random.seed(11)
np.random.shuffle(y_train_2)

"""
np.random.seed(11)
np.random.shuffle(x_validation)
np.random.seed(11)
np.random.shuffle(y_validation)
"""
#x_train=np.vstack((x_train_1[:60000],x_train_2[60000:80000]))
#y_train=np.vstack((y_train_1[:60000],y_train_2[60000:80000]))
x_train=x_train_1[:80000]
y_train=y_train_1[:80000]

x_validation=x_train_1[80000:]
y_validation=y_train_1[80000:]
train_number=len(x_train)
validation_number=len(x_validation)
#x_validation=x_validation[:20000]
#y_validation=y_validation[:20000]

"""
x_data_validation = load('E:\\matlab\\FDD_key_Generation\\data_new\\H1_db30_20M_1W.mat')
y_data_validation = load('E:\\matlab\\FDD_key_Generation\\data_new\\H2_db30_20M_1W.mat')
x_validation=x_data_validation['H1_db30_20M_1W']   #此时读取的就是numpy对象
y_validation=y_data_validation['H2_db30_20M_1W']
#x_validation=x_validation[0:10000]
#y_validation=y_validation[0:10000]

x_data_test = load('E:\\matlab\\FDD_key_Generation\\data_paper\\H1_2.mat')
y_data_test = load('E:\\matlab\\FDD_key_Generation\\data_paper\\H2_2.mat')
x_test=x_data_test['H1']   #此时读取的就是numpy对象
y_test=y_data_test['H2']
#x_test=x_test[0:10000]
#y_test=y_test[0:10000]
"""

"""
x_train=x[:-10000]
y_train=y[:-10000]
x_validation=x[-10000:]
y_validation=y[-10000:]
x_test=x_test[:]
y_test=y_test[:]

"""

"""
np.random.seed(11)
np.random.shuffle(x_train)
np.random.seed(11)
np.random.shuffle(y_train)

np.random.seed(11)
np.random.shuffle(x_validation)
np.random.seed(11)
np.random.shuffle(y_validation)

"""
"""
np.random.seed(11)
np.random.shuffle(x_test)
np.random.seed(11)
np.random.shuffle(y_test)
"""

"""
scaler1=preprocessing.QuantileTransformer(random_state=0)
x_train=scaler1.fit_transform(x_train)
x_validation=scaler1.transform(x_validation)
x_test=scaler1.transform(x_test)
#x_test=scaler1.transform(x_test)
scaler2=preprocessing.QuantileTransformer(random_state=0)
y_train=scaler2.fit_transform(y_train)
y_validation=scaler2.transform(y_validation)
y_test=scaler2.transform(y_test)
#y_test=scaler2.transform(y_test)


#均值标准化
scaler=preprocessing.StandardScaler().fit(x_train)
x_train=scaler.transform(x_train)
# x_test=scaler.transform(x_test)
x_validation=scaler.transform(x_validation)

scaler1=preprocessing.StandardScaler().fit(y_train)
y_train=scaler1.transform(y_train)
# y_test=scaler1.transform(y_test)
y_validation=scaler1.transform(y_validation)

"""

#归一化
min_max_scaler_x = preprocessing.MinMaxScaler()
min_max_scaler_y = preprocessing.MinMaxScaler()

x_train = min_max_scaler_x.fit_transform(x_train)
y_train = min_max_scaler_y.fit_transform(y_train)
x_validation = min_max_scaler_x.transform(x_validation)
y_validation = min_max_scaler_y.transform(y_validation)
#x_test = min_max_scaler_x.transform(x_test)
#y_test = min_max_scaler_y.transform(y_test)

"""
x_train=x_train.reshape((90000,32,32,1))
y_train=y_train
x_validation=x_validation.reshape((10000,32,32,1))
y_validation=y_validation
x_test=x_test.reshape((10000,32,32,1))
"""
#dataset=tf.data.Dataset.from_tensor_slices((x_train_minmax,y_train_minmax))
#print(dataset)
"""
i=0
for element in dataset:
    i+=1
    print(element)
print(i)
"""

#逐层搭建网络结构
"""
model=tf.keras.models.Sequential(
    [
        #tf.keras.layers.Flatten(kernel_regularizer=tf.keras.regularizers.l2()),
        tf.keras.layers.Conv2D(filters=12,kernel_size=(5,5),activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2),
        tf.keras.layers.Conv2D(filters=16,kernel_size=(5,5),activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(256,activation='relu'),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(1024,activation='sigmoid',kernel_initializer='random_uniform')

    ]
)
"""

#配置网络
"""
x_train=x_train.reshape((90000,8,8,2))
x_validation=x_validation.reshape((10000,8,8,2))
x_test=x_test.reshape((10000,8,8,2))
model=tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(filters=12,kernel_size=(2,2),activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2),
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(256,activation='relu'),
        tf.keras.layers.Dense(128,activation='sigmoid')
     ]
    )
"""

model=tf.keras.models.Sequential(
    [
        #tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(1024,activation='relu'),
#       tf.keras.layers.Dense(1024,activation='relu'),
        tf.keras.layers.Dense(1024,activation='relu'),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(128,activation='sigmoid')
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
    loss="mse",
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
log_dir="logs\\fit_1\\"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#执行训练过程
#history=model.fit(x_train,y_train, batch_size=1024, epochs=1000, validation_data=(x_validation,y_validation), 
#                validation_freq=10,callbacks=[cp_callback])
history=model.fit(x_train,y_train, 
                  batch_size=128,
                  epochs=500, 
                  validation_data=(x_validation,y_validation), 
                  validation_freq=1,
                  callbacks=[tensorboard_callback])

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

"""
SNR=str(0)
x_data_train = load('E:\\matlab\\FDD_key_Generation\\data_paper\\NoiseData\\H1_'+SNR+'db.mat')
y_data_train = load('E:\\matlab\\FDD_key_Generation\\data_paper\\NoiseData\\H2_'+SNR+'db.mat')
x_validation=x_data_train['H1_'+SNR+'db']   #此时读取的就是numpy对象
y_validation=y_data_train['H2_'+SNR+'db']

np.random.seed(11)
np.random.shuffle(x_validation)
np.random.seed(11)
np.random.shuffle(y_validation)
x_validation=x_validation[80000:]
y_validation=y_validation[80000:]

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
plt.plot(y_validation[4,:],label='True')
plt.plot(y_validation_predict[4,:],label='Predict')
plt.xlabel('Subcarriers');
plt.ylabel('Value');
plt.legend()
plt.show()

K.sum(K.square((1,2,3)))

loss=np.array(loss)
val_loss=np.array(val_loss)
nmse=np.array(nmse)
val_nmse=np.array(val_nmse)