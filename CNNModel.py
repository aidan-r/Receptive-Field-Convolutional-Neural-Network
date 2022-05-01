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
from skimage.exposure import rescale_intensity


#%%Load Data

#Change these to the respective paths
neuron_path = 'C:/Users/aidan/Desktop/grad/Neuro/Receptive-Field-Convolutional-Neural-Network/crcns-pvc2/2D_noise_natural/Stimulus_Files/'
fname = 'C:/Users/aidan/Desktop/grad/Neuro/Receptive-Field-Convolutional-Neural-Network/data/all_spike_files_2D_noise_natural.mat'
#Load the Data, X is the input image, y is the output firing rate

for j in range(200):
    (spikes,X) = util.load_neuron(j,fname,neuron_path)
    util.sta_new(spikes,X,plot=True)

    y = util.return_firing_rate(spikes,X,plot=True)





    # y = y / np.linalg.norm(y)

    X = np.reshape(X,(X.shape[0]*X.shape[1],X.shape[2]))


    #Could change to kfold after demoing

    from sklearn.model_selection import train_test_split


    X = X.transpose()
    print(X.shape)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.05)



    X_train = X_train.reshape(X_train.shape[0],12,12)
    X_test = X_test.reshape(X_test.shape[0],12,12)

    nkerns = 1

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=nkerns,kernel_size = (12,12),input_shape=(12,12,1),activation='relu',name='conv'))
    model.add(tf.keras.layers.GlobalMaxPooling2D())
    # model.add(keras.layers.Dense())
    model.add(keras.layers.Dense(32,activation=keras.activations.relu))
    model.add(keras.layers.Dense(32,activation = keras.activations.relu))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(8,activation=keras.activations.relu))
    model.add(keras.layers.Dense(1,activation=keras.activations.relu,use_bias=True))

    model.compile(tf.keras.optimizers.Adam(learning_rate=0.01),loss = tf.keras.losses.Poisson())

    model.summary()


    stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=25, restore_best_weights=True)


    hist = model.fit(X_train,y_train,epochs=200,validation_data=[X_test,y_test],callbacks=[stopping])

    #Accuracy Metrics
    plt.figure()
    plt.title('CNN Fit Neuron %s'%(j))
    plt.scatter(y_train,model.predict(X_train),color='red',alpha=0.5)
    plt.xlabel('Real Firing Rates')
    plt.ylabel('Predicted Firing Rates')
    # plt.scatter(y_test,model.predict(X_test),color='blue',alpha=0.5)



    for i in range(nkerns):
    
        plt.figure()
        plt.suptitle('Kernel %s'%(j))
        predicted_sta = np.array(model.weights[0][:,:,:,i])
        predicted_sta = predicted_sta.squeeze()
        print(predicted_sta.shape)
        plt.imshow(predicted_sta,cmap='binary')
        plt.show()
