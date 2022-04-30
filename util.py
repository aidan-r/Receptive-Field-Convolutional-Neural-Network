# -*- coding: utf-8 -*-
"""
<<<<<<< HEAD
Created on Thu Jan 27 16:01:06 2022

Useful functions for processing data

@author: aidan
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nrmse


#--------------------------------------
def find_nearest(array,value,out='index'):
    idx = (np.abs(array-value)).argmin()
    if out=='index':
        return idx
    if out=='value':
        return array[value]
    
    
#--------------------------------------   
def load_and_print_stim(path,num_frames = 5,plot=False):
    stim = scipy.io.loadmat(path)
    stim = stim['mov']
    print(stim.shape) 
    if plot:
        plt.figure()
        for i in range(num_frames):
            plt.imshow(stim[:,:,i],cmap='gray')
            plt.show()            
    return stim

#--------------------------------------  
def print_spike_triggered_average(spike_times,stim,frames_back=10,plot=False,idx=None):
    
    nframes = frames_back

    sta = np.zeros(shape = (stim.shape[0],stim.shape[1],spike_times.shape[0],nframes))

    #Need to convert a spike time to an index within the movies
    max_idx = find_nearest(spike_times, 1000000)
    
    for i in range(max_idx):
        spike = spike_times[i]

        spike_in_seconds = spike * 1e-4

        frame_index = int(spike_in_seconds * 24) #frames/second)
    
        for j in range(nframes):
            sta[:,:,i,j] = stim[:,:,frame_index-j]
    
    
    sta_show = np.mean(sta,axis=2)
    if plot:
        
        plt.figure(figsize=(15,17))
        if idx is not None:
            plt.suptitle('Neuron %s'%(idx))
        plt.tight_layout()
        for i in range(nframes):
            plt.subplot(5,5,i+1)
            plt.title('%s seconds Before Spike'%(round((nframes- i) / 24,2)))
            plt.imshow(sta_show[:,:,i],cmap='gray')
        plt.show()
    return sta_show

#--------------------------------------  
def load_neuron(index,fname,neuron_path):
    # fname = 'C:/Users/aidan/Desktop/grad/Neuro/data/all_spike_files_2D_noise_natural.mat'

    data = scipy.io.loadmat(fname)
    data = data['all_spike_files_2D_noise_natural']

    #num_cells = data.shape[0]
    num_cells = 10

    #This is spikes
    spikes = data[index]['events'][0]
    spikes = spikes.transpose()
    print(data[index]['name'])
    #
    
    try:
        stim_name = data[index]['name'][0][0][13:23].capitalize()
        stim_num = data[index]['name'][0][0][-6:-4].capitalize()
        path = neuron_path + '%s_%s_25hz.mat'%(stim_name,stim_num)
        stim = load_and_print_stim(path)
    except:
        stim_name = data[index]['name'][0][0][13:24].capitalize()
        stim_num = data[index]['name'][0][0][-6:-4].capitalize()
        path = neuron_path + '%s_%s_25hz.mat'%(stim_name,stim_num)
        stim = load_and_print_stim(path)


    # sta = print_spike_triggered_average(spikes, stim)
    return spikes,stim

#--------------------------------------  
def isi_mean(spikes,plot=False):
    isi = np.diff(spikes[:,0])
    mean = np.mean(isi)
    var = np.var(isi)
    if plot:
        plt.figure()
        plt.title('Interspike Interval')
        plt.hist(isi,bins=20)
        plt.xlabel('Time')
        plt.ylabel('Counts')
        print('Mean ISI is %s'%(mean))
        print('ISI Variance is %s'%(var))

    return mean

#--------------------------------------  
def similarity_vs_firing_rate(spikes,stim,tau=None,plot=False):
    if tau == None:
        tau = int(isi_mean(spikes) * 10)
    
    #This will return a list of the similarity between a stimulus and spike triggered average
    ssim_save = np.zeros(shape=(stim.shape[2],))
    
    sta = print_spike_triggered_average(spikes, stim,plot=False)
    
    for i in range(stim.shape[2]):
        ssim_save[i] = ssim(stim[:,:,i],sta[:,:,-1])
    
    rate_save = np.zeros(shape=(stim.shape[2],))
    
    #This will calc the similarity vs firing rate
    for i in range(stim.shape[2]):
        time_start = find_nearest(spikes,int((i / 24) / (1e-4)))
        time_stop = find_nearest(spikes,int((i/24) / (1e-4)) + tau)
        rate = spikes[time_start:time_stop].shape[0]
        rate_save[i] = rate
        
    if plot:
        plt.figure()
        plt.title('Similarity vs Firing Rate')
        plt.scatter(ssim_save,rate_save,alpha=0.5)
        plt.xlabel('Similarity to STA')
        plt.ylabel('Firing Rate')
        plt.show()
        
#--------------------------------------          
def compare_jitter(spikes,stim):
    sta_norm = print_spike_triggered_average(spikes, stim,plot=True)

    jitter_help = np.random.randint(-5,5,size=(spikes.shape[0],1))
    jitter_spikes = (jitter_help * isi_mean(spikes)) + spikes

    
    jitter_sta = print_spike_triggered_average(jitter_spikes, stim,plot=True)
    
 
#--------------------------------------  
def return_firing_rate(spikes,stim,tau=None,plot=False):
    if tau == None:
        tau = int(isi_mean(spikes) * 10)
        print(tau)


    rate_save = np.zeros(shape=(stim.shape[2],))

    for i in range(stim.shape[2]):
        time_start = find_nearest(spikes,int((i / 24) / (1e-4)))
        time_stop = find_nearest(spikes,int((i/24) / (1e-4)) + tau)
        rate = spikes[time_start:time_stop].shape[0]
        
        #could divid by tau here
        rate_save[i] = rate
    
    if plot:
        plt.figure()
        plt.title('Firing Rate vs Time')
        plt.plot(np.linspace(1,rate_save.shape[0],rate_save.shape[0]),rate_save)
    return rate_save


#-------------------------------------- 
def similarity_vs_firing_rate_mse(spikes,stim,tau=None,plot=False):
    if tau == None:
        tau = int(isi_mean(spikes) * 10)
    
    #This will return a list of the similarity between a stimulus and spike triggered average
    mse_save = np.zeros(shape=(stim.shape[2],))
    
    sta = print_spike_triggered_average(spikes, stim,plot=False)
    
    for i in range(stim.shape[2]):
        mse_save[i] = mse(stim[:,:,i],sta[:,:,-1])
    
    rate_save = np.zeros(shape=(stim.shape[2],))
    
    #This will calc the similarity vs firing rate
    for i in range(stim.shape[2]):
        time_start = find_nearest(spikes,int((i / 24) / (1e-4)))
        time_stop = find_nearest(spikes,int((i/24) / (1e-4)) + tau)
        rate = spikes[time_start:time_stop].shape[0]
        rate_save[i] = rate
        
    if plot:
        plt.figure()
        plt.title('Similarity vs Firing Rate')
        plt.scatter(mse_save,rate_save,alpha=0.5)
        plt.xlabel('Similarity to STA')
        plt.ylabel('Firing Rate')
        plt.show()
        
#-------------------------------------- 
def similarity_vs_firing_rate_nrmse(spikes,stim,tau=None,plot=False):
    if tau == None:
        tau = int(isi_mean(spikes) * 10)
    
    #This will return a list of the similarity between a stimulus and spike triggered average
    nrmse_save = np.zeros(shape=(stim.shape[2],))
    
    sta = print_spike_triggered_average(spikes, stim,plot=False)
    
    for i in range(stim.shape[2]):
        nrmse_save[i] = nrmse(stim[:,:,i],sta[:,:,-1])
    
    rate_save = np.zeros(shape=(stim.shape[2],))
    
    #This will calc the similarity vs firing rate
    for i in range(stim.shape[2]):
        time_start = find_nearest(spikes,int((i / 24) / (1e-4)))
        time_stop = find_nearest(spikes,int((i/24) / (1e-4)) + tau)
        rate = spikes[time_start:time_stop].shape[0]
        rate_save[i] = rate
        
    if plot:
        plt.figure()
        plt.title('Similarity vs Firing Rate')
        plt.scatter(nrmse_save,rate_save,alpha=0.5)
        plt.xlabel('Normalized Erro Input vs STA')
        plt.ylabel('Firing Rate')
        plt.show()

