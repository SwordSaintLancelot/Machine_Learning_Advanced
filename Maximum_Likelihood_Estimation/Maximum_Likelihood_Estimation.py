# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:04:28 2019

@author: GHOST
"""

import numpy as np
import matplotlib.pyplot as plt

#Exercise 2 question 4
n = np.arange(100)
w = np.sqrt(0.25)*np.random.randn(100)
x = np.sin(2*3.14*0.017*n)+w
plt.plot(n,x)

# Estimating the frequency of first sine wave.
scores = [] 
frequencies = []
for f in np.linspace(0, 0.5, 1000):
# Create vector e. Assume data is in x. 
    n = np.arange(100) 
    z = -2*np.pi*f*1j*n# <compute -2*pi*i*f*n. Imaginary unit is 1j> 
    e = np.exp(z)
    score =abs(np.dot(x,e)) # <compute abs of dot product of x and e> 
    scores.append(score) 
    frequencies.append(f)
    
    
fHat = frequencies[np.argmax(scores)]


#Exercise 4 question 5


