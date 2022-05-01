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
import scipy


#%%load data

# path = 'C:/Users/aidan/Desktop/grad/Neuro/Receptive-Field-Convolutional-Neural-Network/data/2022_04_29_Noise_45deg_120s_3.mat'
path = 'C:/Users/aidan/Desktop/grad/Neuro/Receptive-Field-Convolutional-Neural-Network/data/2022_04_30_Compex_Noise_45+90deg_120s.mat'
model_data = scipy.io.loadmat(path)

stim = model_data['model_output'][0][0][2]
spikes = model_data['model_output'][0][0][6][0,:]
spikes = spikes[:120000]

#%%
stim = util.normalize_images(stim)
X = stim

X = np.reshape(X,(X.shape[0]*X.shape[1],X.shape[2]))

X = X.transpose()


print(X.shape)

y = modeling_data_processing.return_firing_rate(spikes,stim)



#%%

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)


X_train = X_train.reshape(X_train.shape[0],149,149)
X_test = X_test.reshape(X_test.shape[0],149,149)


nkerns = 4

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(filters=nkerns,kernel_size = (149,149),input_shape=(149,149,1),activation='relu',name='conv'))
model.add(tf.keras.layers.GlobalMaxPooling2D())
# model.add(keras.layers.Dense())
# model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(16,activation=keras.activations.relu))
model.add(keras.layers.Dense(16,activation = keras.activations.relu))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(8,activation=keras.activations.relu))
model.add(keras.layers.Dense(1,activation=keras.activations.relu,use_bias=True))

model.compile(tf.keras.optimizers.Adam(learning_rate=0.01),loss = tf.keras.losses.Poisson())
# tf.keras.losses.Poisson
model.summary()

#tf.keras.losses.Poisson()

stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100, restore_best_weights=True)


hist = model.fit(X_train,y_train,epochs=300,validation_data=[X_test,y_test],callbacks=[stopping])



from scipy.stats import pearsonr


r = pearsonr(y_test,model.predict(X_test)[:,0])
print(r)

plt.figure()
plt.title('CNN Predictions Training')
plt.scatter(y_train,model.predict(X_train),color='red',alpha=0.5)

plt.xlabel('Real Firing Rates')
plt.ylabel('Predicted Firing Rates')
plt.savefig('figures/cnn_corr.png')

plt.figure()
plt.title('CNN Predictions')
# plt.text(150, 100, 'R= %s'%(r[0]))
plt.scatter(y_test,model.predict(X_test),color='red',alpha=0.5)
plt.xlabel('Real Firing Rates')
plt.ylabel('Predicted Firing Rates')
plt.savefig('figures/cnn_corr.png')

plt.figure(figsize=(15,15))
for i in range(nkerns):
    
    plt.subplot(2,2,i+1)
    plt.title('Kernel %s'%(i+1))
    predicted_sta = np.array(model.weights[0][:,:,:,i])
    predicted_sta = predicted_sta.squeeze()
    print(predicted_sta.shape)
    plt.imshow(predicted_sta,cmap='Greys')
    plt.savefig('figures/learned_kernel_mult.png')
    
#%%
plt.figure()
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Poisson Loss')
plt.plot(hist.history['loss'],color='red')
plt.plot(hist.history['val_loss'],color='blue')
plt.legend(labels=['Training','Validation'])

plt.figure()
plt.title('Model Accuracy vs Shuffled Firing Rates')
plt.bar([0,1],[hist.history['val_loss']])

#%%
np.random.shuffle(y_train)
np.random.shuffle(y_test)

model.compile(tf.keras.optimizers.Adam(learning_rate=0.01),loss = tf.keras.losses.Poisson())
hist = model.fit(X_train,y_train,epochs=300,validation_data=[X_test,y_test],callbacks=[stopping])


plt.figure()
plt.title('CNN Predictions Training')
plt.scatter(y_train,model.predict(X_train),color='red',alpha=0.5)
plt.xlabel('Real Firing Rates')
plt.ylabel('Predicted Firing Rates')
plt.savefig('figures/cnn_corr.png')
