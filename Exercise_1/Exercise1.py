# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:19:59 2019

@author: GHOST
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import get_test_data


'''test = []
data= np.loadtxt('locationData.csv')
for i in data:
    test.append(i)
    test.append('\n')

test = np.array(test)
    
    
print(np.shape(test))'''
'''np.shape(data)
print(data)
plt.plot(data[:,0],data[:,1])
plt.show()
ax = plt.subplot(1,1,1,projection = '3d')
ax.plot(data[:,0],data[:,1],data[:,2])
plt.show()
'''
from scipy.io import loadmat

mat = loadmat("twoClassData.mat")





#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data

def main():
    test_list = []
    a = np.loadtxt('locationData.csv')
    with open("locationData.csv","r") as fd:
        for line in fd:
            value = line.split(" ")
            value = [float(v) for v in value]
            test_list.append(value)
        test_list = np.array(test_list)    
    print(np.shape(test_list))
    print(a)
    print("just checking")
    print(test_list)
    print(np.all(a == test_list))
main()