# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:29:29 2019

@author: GHOST
"""
#%% Creating Noisy Signal and applying Detectors
import numpy as np
import matplotlib.pyplot as plt
vec_x = np.zeros(500)
vec_y = np.zeros(300)
n = np.arange(100)
wave = np.cos(2*np.pi*0.03*n)
full_vec = np.concatenate((vec_x,vec_y,wave), axis=0, out=None)
plt.figure(1)
fig = plt.subplot(2,2,1)
fig.set_title('Noiseless_Wave')
plt.plot(full_vec)
noise = np.sqrt(0.5)*np.random.randn(full_vec.size)
noisy_signal = full_vec+noise
plt.subplot(2,2,2)
plt.title('Noisy Signal')
plt.plot(noisy_signal)

Detector_1 = np.exp(-2*np.pi*1j*0.03*n)
Detection_Result = np.abs(np.convolve(Detector_1,noisy_signal,'Same'))
plt.subplot(2,2,3)
plt.title('Detection Result')
plt.plot(Detection_Result)