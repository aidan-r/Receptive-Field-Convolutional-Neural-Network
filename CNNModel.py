# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 11:41:27 2022

@author: aidan
"""

#%%import modules
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import util
import sklearn


#%%Load Data

#Change these to the respective paths
neuron_path = 'C:/Users/aidan/Desktop/grad/Neuro/Receptive-Field-Convolutional-Neural-Network/crcns-pvc2/2D_noise_natural/Stimulus_Files/'
fname = 'C:/Users/aidan/Desktop/grad/Neuro/Receptive-Field-Convolutional-Neural-Network/data/all_spike_files_2D_noise_natural.mat'

#Load the Data, X is the input image, y is the output firing rate
(spikes,X) = util.load_neuron(40,fname,neuron_path)

y = util.return_firing_rate(spikes,X,plot=True)





# y = y / np.linalg.norm(y)

X = np.reshape(X,(X.shape[0]*X.shape[1],X.shape[2]))


#Could change to kfold after demoing

from sklearn.model_selection import train_test_split


X = X.transpose()
print(X.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33)


#%%Build the model
X_train = X_train.reshape(X_train.shape[0],12,12)
X_test = X_test.reshape(X_test.shape[0],12,12)


model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(filters=1,kernel_size = (12,12),input_shape=(12,12,1),activation='relu',name='conv'))
# model.add(tf.keras.layers.GlobalMaxPooling2D())
# model.add(keras.layers.Dense())
model.add(keras.layers.Dense(8,activation=keras.activations.relu))
model.add(keras.layers.Dense(8,activation = keras.activations.relu))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(4,activation=keras.activations.relu))
model.add(keras.layers.Dense(1,activation=keras.activations.relu,use_bias=True))

model.compile(tf.keras.optimizers.Adam(learning_rate=0.01),loss = tf.keras.losses.Poisson())

model.summary()


#%%Train the Model
stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=25, restore_best_weights=True)


hist = model.fit(X_train,y_train,epochs=15,validation_data=[X_test,y_test],callbacks=[stopping])

#Accuracy Metrics
plt.figure()
plt.scatter(y_train,model.predict(X_train),color='red',alpha=0.5)
plt.xlabel('Real Firing Rates')
plt.ylabel('Predicted Firing Rates')
# plt.scatter(y_test,model.predict(X_test),color='blue',alpha=0.5)



predicted_sta = np.array(model.weights[0])
predicted_sta = predicted_sta.squeeze()
print(predicted_sta.shape)

plt.figure()
plt.imshow(predicted_sta,cmap='binary')
