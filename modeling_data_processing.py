# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 21:34:38 2022

@author: aidan
"""

#%%import modules
import numpy as np
import matplotlib.pyplot as plt
import util
import scipy
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nrmse


#%%
path = 'C:/Users/aidan/OneDrive/Desktop/grad courses/neuro/Receptive-Field-Convolutional-Neural-Network/data/2022_04_29_Noise_45deg_120s.mat'
model_data = scipy.io.loadmat(path)

print(model_data.keys())

#%%
print(model_data['simple_model_output'][0][0][7])
print(model_data['simple_model_output'][0][0][7].shape)

#%%[0]
# print(model_data.keys())
# print(model_data['simple_model_output'][0][0][6])
# print(type(model_data['simple_model_output'][0][0][6]))
# print(model_data['simple_model_output'][0][0][6].shape)

stim = model_data['simple_model_output'][0][0][3]
spikes = model_data['simple_model_output'][0][0][7][0,:]
spikes = spikes[:120000]


#%%


def model_sta(spikes,stim,plot=True):
    
    spike_count = np.sum(spikes)
    
    #keep track of the indices in the sta save, equal to the number of spikes
    hit_count = 0
    sta_save = np.zeros(shape=(stim.shape[0],stim.shape[1],int(spike_count)))
    print(sta_save.shape)
    #STA loop 
    for i in range(spikes.shape[0]):
        # print(i,hit_count)
        
        if spikes[i] == 1:
            idx = round(i/40)
            sta_save[:,:,hit_count] = stim[:,:,idx-1]
            hit_count+=1 
                        
    sta_calc = np.mean(sta_save,axis=2)

    if plot:
        plt.figure()
        plt.title('Spike Triggered Average')
        plt.imshow(sta_calc,cmap='binary')
        plt.show()
        
    return sta_calc

sta_calc = model_sta(spikes,stim)

#%%
def isi_mean(spikes,plot=False):
    idx = np.where(spikes==1)
    isi = np.diff(idx)
    isi = isi.transpose()
    mean = np.mean(isi)
    var = np.var(isi)
    if plot:
        plt.figure()
        plt.title('Interspike Interval')
        plt.hist(isi,bins=15)
        plt.xlabel('Time')
        plt.ylabel('Counts')
        print('Mean ISI is %s'%(mean))
        print('ISI Variance is %s'%(var))

    return mean

isi_mean(spikes,plot=True)

def find_nearest(array,value,out='index'):
    idx = (np.abs(array-value)).argmin()
    if out=='index':
        return idx
    if out=='value':
        return array[value]
    
#%%

def similarity_vs_firing_rate(spikes,stim,sta,metric,tau=None,plot=False):
    if tau == None:
        tau = int(isi_mean(spikes) * 150)
    print(tau)
    
    #This will return a list of the similarity between a stimulus and spike triggered average
    ssim_save = np.zeros(shape=(stim.shape[2],))
    
    #calc the similarity
    for i in range(stim.shape[2]):
        ssim_save[i] = metric(stim[:,:,i],sta)
    
    #firing rate
    rate_save = np.zeros(shape=(stim.shape[2]))
    for i in range(stim.shape[2]):
        time_start = int((i*40))
        time_stop = int((i*40)+ tau)
    
        rate = spikes[time_start:time_stop].sum()
        rate_save[i] = rate
    if metric == mse:
        title = 'Mean Squared Error'
    elif metric == nrmse:
        title = 'Norm. Root Mean Squared Error'
    elif metric == ssim:
        title ='Structrual Similarity Index'
    if plot:
        plt.figure()
        plt.title('Similarity vs Firing Rate %s'%(title))
        plt.scatter(ssim_save,rate_save,alpha=0.5)
        plt.xlabel('Similarity to STA')
        plt.ylabel('Firing Rate')
        plt.show()
        
        
similarity_vs_firing_rate(spikes, stim, sta_calc, mse,plot=True)
similarity_vs_firing_rate(spikes, stim, sta_calc, nrmse,plot=True)
similarity_vs_firing_rate(spikes, stim, sta_calc, ssim,plot=True) 


#%%Firing rate model data

def return_firing_rate(spikes,tau=None):
    if tau is None:
        tau = int(isi_mean(spikes) * 150)
        
    rate_save = np.zeros(shape=(stim.shape[2]))
    
    for i in range(stim.shape[2]):
        time_start = int((i*40))
        time_stop = int((i*40)+ tau)
        rate = spikes[time_start:time_stop].sum()
        rate_save[i] = rate
        
    return rate_save

