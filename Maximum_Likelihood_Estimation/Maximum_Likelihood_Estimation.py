# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:04:28 2019

@author: GHOST
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

#question 4
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

#question 5


# Read the data

img = imread("uneven.jpg")
plt.imshow(img, cmap='gray')
plt.title("Image shape is %dx%d" % (img.shape[1], img.shape[0]))
plt.show()

# Create the X-Y coordinate pairs in a matrix
X, Y = np.meshgrid(range(1300), range(1030))
Z = img

x = X.ravel()
y = Y.ravel()
z = Z.ravel()

# ********* TODO 1 **********
# Create data matrix
# Use function "np.column_stack".
# Function "np.ones_like" creates a vector like the input.
H = np.column_stack([x**2,y**2,x*y,x,y,np.ones_like(x)])

# ********* TODO 2 **********
# Solve coefficients
# Use np.linalg.lstsq
# Put coefficients to variable "theta" which we use below.
theta, a,b,c= np.linalg.lstsq(H,z,rcond=None)
# Predict
z_pred = H @ theta
Z_pred = np.reshape(z_pred, X.shape)

# Subtract & show
S = Z - Z_pred
plt.imshow(S, cmap = 'gray')
plt.show()

