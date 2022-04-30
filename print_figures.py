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


#%%
neuron_path = 'C:/Users/aidan/Desktop/grad/Neuro/Receptive-Field-Convolutional-Neural-Network/crcns-pvc2/2D_noise_natural/Stimulus_Files/'
fname = 'C:/Users/aidan/Desktop/grad/Neuro/Receptive-Field-Convolutional-Neural-Network/data/all_spike_files_2D_noise_natural.mat'
ex = 'C:/Users/aidan/Desktop/grad/Neuro/Receptive-Field-Convolutional-Neural-Network/crcns-pvc2/2D_noise_natural/Stimulus_Files/Equalpower_A1_25hz.mat'

# stim = util.load_and_print_stim(ex,plot=True)


(spikes,stim) = util.load_neuron(20,fname,neuron_path)
#ssim = util.similarity_vs_firing_rate(spikes,stim,tau=500,plot=True)
sta = util.print_spike_triggered_average(spikes,stim,plot=True)

# isi = util.isi_mean(spikes,plot=True)

#%%
