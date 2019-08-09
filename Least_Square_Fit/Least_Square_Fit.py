# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:19:59 2019

@author: GHOST
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import get_test_data
from scipy.io import loadmat

#exercise 1 question 3

test = []
data= np.loadtxt('E:\\GitHub\\Machine_Learning_Advanced\\Least_Square_Fit\\locationData.csv')
with open ("E:\\GitHub\\Machine_Learning_Advanced\\Least_Square_Fit\\locationData.csv") as fd:
    for i in fd:
        val = i.split(" ")
        val = [float(v) for v in val]
        test.append(val)
    test = np.array(test)
    
print(np.shape(test))
print(data)



#Plotting for part 1 and 2

plt.plot(data[:,0],data[:,1])
plt.show()
ax = plt.subplot(1,1,1,projection = '3d')
ax.plot(data[:,0],data[:,1],data[:,2])
plt.show()


# exercise 1 question 4 
mat = loadmat("E:\\GitHub\\Machine_Learning_Advanced\\Least_Square_Fit\\twoClassData.mat")
X = mat['X']
y = mat['y'].ravel()

plt.plot(X[y==0,0],X[y==0,1],'ro')
plt.plot(X[y==1,0],X[y==1,1],'bo')
plt.show()

# Exxercise 1 question 5 

X1 = np.array(np.load('E:\\GitHub\\Machine_Learning_Advanced\\Least_Square_Fit\\x.npy'))
Y1 = np.array(np.load('E:\\GitHub\\Machine_Learning_Advanced\\Least_Square_Fit\\y.npy'))
A = np.vstack([X1,np.ones(len(X1))]).T

m,c = np.linalg.lstsq(A,Y1,rcond = None)[0]

plt.plot(X1,Y1,'o', label='Original Data', markersize = 5)
plt.plot(X1,m*X1+c,'r', label = 'Fitted Line')
plt.show()
