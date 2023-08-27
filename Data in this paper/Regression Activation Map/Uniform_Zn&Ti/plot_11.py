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
    mat = scipy.io.loadmat('the606060_cell_U.mat')
    data= mat['the606060_cell']
    data=data.reshape(4,60*60*60)
    means =  np.mean(data,axis=1)
    # E=pd.read_csv('E_0_r0.csv')
    # Y=pd.read_csv('yield_0_r0.csv')
    
    print(data[0,100])