# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:19:59 2019

@author: GHOST
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import get_test_data


test = []
data= np.loadtxt('locationData.csv')
for i in data:
    test.append(i)
    test.append('\n')
    
'''np.shape(data)
print(data)
plt.plot(data[:,0],data[:,1])
plt.show()
ax = plt.subplot(1,1,1,projection = '3d')
ax.plot(data[:,0],data[:,1],data[:,2])
plt.show()
'''
