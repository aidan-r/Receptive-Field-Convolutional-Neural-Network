# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 13:55:20 2022

@author: aidan
"""

#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import skimage
import util
import modeling_data_processing
import tensorflow as tf
import keras

#%%load data

path = 'C:/Users/aidan/OneDrive/Desktop/grad courses/neuro/Receptive-Field-Convolutional-Neural-Network/data/2022_04_29_Noise_45deg_120s.mat'
model_data = scipy.io.loadmat(path)

stim = model_data['simple_model_output'][0][0][3]
spikes = model_data['simple_model_output'][0][0][7][0,:]
spikes = spikes[:120000]

#%%
X = stim
X = np.reshape(X,(X.shape[0]*X.shape[1],X.shape[2]))

X = X.transpose()

print(X.shape)

y = modeling_data_processing.return_firing_rate(spikes)
#%%

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33)

X_train = X_train.reshape(X_train.shape[0],150,150)
X_test = X_test.reshape(X_test.shape[0],150,150)

#%%
nkerns = 1

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(filters=nkerns,kernel_size = (150,150),input_shape=(150,150,1),activation='relu',name='conv'))
model.add(tf.keras.layers.GlobalMaxPooling2D())
# model.add(keras.layers.Dense())
model.add(keras.layers.Dense(16,activation=keras.activations.relu))
model.add(keras.layers.Dense(16,activation = keras.activations.relu))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(8,activation=keras.activations.relu))
model.add(keras.layers.Dense(1,activation=keras.activations.relu,use_bias=True))

model.compile(tf.keras.optimizers.Adam(learning_rate=0.001),loss = tf.keras.losses.Poisson())

model.summary()

#tf.keras.losses.Poisson()
#%%
stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=25, restore_best_weights=True)


hist = model.fit(X_train,y_train,epochs=1000,validation_data=[X_test,y_test],callbacks=[stopping])

#Accuracy Metrics
plt.figure()
plt.scatter(y_train,model.predict(X_train),color='red',alpha=0.5)
plt.xlabel('Real Firing Rates')
plt.ylabel('Predicted Firing Rates')
# plt.scatter(y_test,model.predict(X_test),color='blue',alpha=0.5)





for i in range(nkerns):
    
    plt.figure()
    plt.title('Learned Kernel')
    predicted_sta = np.array(model.weights[0][:,:,:,i])
    predicted_sta = predicted_sta.squeeze()
    print(predicted_sta.shape)
    plt.imshow(predicted_sta,cmap='binary')
    plt.show()