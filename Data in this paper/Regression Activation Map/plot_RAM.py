# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 00:13:33 2022

@author: Amber
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import scipy.io
for n in range(130,131):
    mat = scipy.io.loadmat('the606060_cell_209.mat')
    data= mat['the606060_cell']

    all_x= np.load('camY_15.npy',allow_pickle=True)
    
    # n=132

    # n=154
    x=all_x[n]
    x_expand=x.ravel()
    x_mean= np.mean(x)
    x_std = np.std(x)
    
    #x[x < x_mean+x_std] = 0
    #x[x >= x_mean+x_std] = 1
    df1=data[n][0]#
    volume=x
    #volume=df1
    volume_xy=volume[:,0,:]
    for i in range(1,15):  
        volume_xy+=volume[:,i,:]
    plt.figure()
    plt.imshow(volume_xy*100/(volume_xy.sum()),cmap='RdYlBu_r')
    plt.colorbar()
    #plt.rc("font", size=20)
    z=volume_xy
    
    #from mpl_toolkits.axes_grid1 import make_axes_locatable
    # Create the x, y, and z coordinate arrays.  We use 
    # numpy's broadcasting to do all the hard work for us.
    # We could shorten this even more by using np.meshgrid.
    
    sumvertical = np.sum(z, 0)
    xvert = range(np.shape(z)[1])
    
    
    sumhoriz = np.sum(z, 1)
    yhoriz = range(np.shape(z)[0])
    
    plt.figure(figsize=(12,4))
    # Random gaussian data.
    Ntotal = 1000
    data = 0.05 * np.random.randn(Ntotal) + 0.5
    
    # This is  the colormap I'd like to use.
    cm = plt.cm.get_cmap('RdYlBu_r')
    
    # Get the histogramp
    Y,X = np.histogram(data, 15, normed=1)
    Y=sumvertical
    x_span = X.max()-X.min()
    #C = [cm(((x-X.min())/x_span)) for x in X]
    y_span = Y.sum()
    C = [cm(((y-Y.min())*10/y_span)) for y in Y]
    plt.bar(X[:-1],Y,color=C,width=X[1]-X[0])
    plt.show()
    #