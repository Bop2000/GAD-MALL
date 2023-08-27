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
for n in range(0,1):
    # mat = scipy.io.loadmat('the606060_cell_Ti_train.mat')
    # data= mat['the606060_cell']
    # E=pd.read_csv('E_0_r0.csv')
    # Y=pd.read_csv('yield_0_r0.csv')
    
    
    # x2=np.load('input_0_r0.npy', allow_pickle=True)
    # input_total=[]
    # input_total = x2.reshape(len(x2), 12*12*12)
    # means =  np.mean(input_total,axis=1).reshape(len(input_total))
    # plt.figure()
    # plt.scatter(means,E)
    # E1= E['E']
    # plt.scatter(means[132],E1[132],marker='v')
    # plt.show()
    
    # plt.figure()
    # plt.scatter(means,Y)
    # Y1= Y['yield']
    # plt.scatter(means[132],Y1[132],marker='v')
    # plt.show()
    
    
    # # Make this bigger to generate a dense grid.
    # #N = 8
    # total = E
    # total['Y'] = Y['yield']
    # E500 = total[(total['E'] >480) & (total['E']<520)]
    
    # E1000 = total[(total['E'] >950) & (total['E']<1050)]
    # ## Create some random data.
    # #volume = df1
    all_x= np.load('camY_15_u500old.npy',allow_pickle=True)
    
    n=0
    # n=154
    x=all_x[n]
    x_expand=x.ravel()
    x_mean= np.mean(x)
    x_std = np.std(x)
    
    #x[x < x_mean+x_std] = 0
    #x[x >= x_mean+x_std] = 1
    # df1=data[n][0]#
    volume=x
    #volume=df1
    volume_xy=volume[:,0,:]
    for i in range(1,15):  
        volume_xy+=volume[:,i,:]
    plt.figure()
    plt.imshow(volume_xy*100/(volume_xy.sum()),cmap='RdYlBu_r')
    # ax=plt.pcolor(volume_xy*100/(volume_xy.sum()),cmap='RdYlBu_r',vmin=-0.2,vmax=0.8) #U2000
    plt.colorbar()
    #plt.rc("font", size=20)
    z=volume_xy*100/(volume_xy.sum())
    
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