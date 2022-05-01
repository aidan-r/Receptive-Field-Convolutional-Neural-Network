# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:02:24 2022

@author: aidan
"""

#%%Import modules
import pandas as pd
import numpy as np
import util
import modeling_data_processing
from skimage.exposure import rescale_intensity



neuron_path = 'C:/Users/aidan/Desktop/grad/Neuro/Receptive-Field-Convolutional-Neural-Network/crcns-pvc2/2D_noise_natural/Stimulus_Files/'
fname = 'C:/Users/aidan/Desktop/grad/Neuro/Receptive-Field-Convolutional-Neural-Network/data/all_spike_files_2D_noise_natural.mat'
ex = 'C:/Users/aidan/Desktop/grad/Neuro/Receptive-Field-Convolutional-Neural-Network/crcns-pvc2/2D_noise_natural/Stimulus_Files/Equalpower_A1_25hz.mat'

# stim = util.load_and_print_stim(ex,plot=True)

for i in range(1,200):
    (spikes,stim) = util.load_neuron(i,fname,neuron_path)
   
    # sta_old = util.print_spike_triggered_average(spikes,stim,plot=True)
    # sta = util.sta_new(spikes,stim,plot=True,idx=i)
    # print(sta)
    util.similarity_new(spikes,stim,plot=True,idx=i)



#%%
print(sta)
print(rescale_intensity(sta))