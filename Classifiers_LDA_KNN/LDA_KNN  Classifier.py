# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:29:29 2019

@author: GHOST
"""
#%% Creating Noisy Signal and applying Detectors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.io import loadmat
from sklearn.metrics import accuracy_score

vec_x = np.zeros(500)
vec_y = np.zeros(300)
n = np.arange(100)
wave = np.cos(2*np.pi*0.1*n)
full_vec = np.concatenate((vec_x,wave, vec_y), axis=0, out=None)
plt.figure(1)
fig = plt.subplot(2,2,1)
fig.set_title('Noiseless_Wave')
plt.plot(full_vec)
noise = np.sqrt(0.5)*np.random.randn(full_vec.size)
noisy_signal = full_vec+noise
plt.subplot(2,2,2)
plt.title('Noisy Signal')
plt.plot(noisy_signal)

#Detector_1 = np.exp(-2*np.pi*1j*0.04*n)
Detection_Result = np.abs(np.convolve(full_vec,noisy_signal,'Same'))
plt.subplot(2,2,3)
plt.title('Detection Result')
plt.plot(Detection_Result)


#%% Detection with a different Frequency

wave = np.cos(2*np.pi*0.03*n)
full_vec = np.concatenate((vec_x,wave, vec_y), axis=0, out=None)
plt.figure(2)
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


#%%  Training the classifiers
mat = loadmat('twoClassData.mat')
X = mat['X']
y = mat['y'].ravel()
train_x = np.concatenate((X[:100],X[200:300]))
test_x = np.concatenate((X[100:200],X[300:]))
train_y = np.concatenate((y[:100],y[200:300]))
test_y = np.concatenate((y[100:200],y[300:]))
# The data set needs to be selected in this way because the data from 0 to 200 is '1' and from 200 to 400 is '0'
#so inorder to train the model properly we need different type of datasets.
# KNN Classifier
train_KNN = KNeighborsClassifier()
train_KNN.fit(train_x,train_y)
pred_y_KNN = train_KNN.predict(test_x)
print('The accuracy of KNN classifier is {}'.format(accuracy_score(test_y,pred_y_KNN)*100))

# LDA Classifier

train_LDA = LinearDiscriminantAnalysis()
train_LDA.fit(train_x,train_y)
pred_y_LDA = train_LDA.predict(test_x)
print('The accuracy of LDA classifier is {}'.format(accuracy_score(test_y,pred_y_LDA)*100))
