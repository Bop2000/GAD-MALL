import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from sklearn.mixture import GaussianMixture
import tensorflow
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import Input,Model
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
path = os.getcwd()
name1=path+'/Datasets/'
name2=path+'/Models/'
name3=path+'/Results/'

#parameters
target = 2500#Choose a elastic modulus target, such as target = 2500 MPa
sam_=10000000#Sampling number
up=0.9
n_model=10

def ensemble_predict_E(S):
    E_all=0
    for i in range(1,n_model+1):
        modelname = name2+"3dCNN_E%d.h5"%(i)
        model_E = keras.models.load_model(modelname)
        temp_E=model_E.predict(S)
        temp_E=pd.DataFrame(temp_E)
        temp_E.columns=['E']
        E_all+=temp_E
    E_all['E']/=n_model
    return E_all               

def ensemble_predict_Y(S):
    Y_all=0
    for i in range(1,n_model+1):
        modelname = name2+"3dCNN_y%d.h5"%(i)
        model_Y = keras.models.load_model(modelname)
        temp_Y=model_Y.predict(S)
        temp_Y=pd.DataFrame(temp_Y)
        temp_Y.columns=['yield']
        Y_all+=temp_Y
    Y_all['yield']/=n_model
    return Y_all  

def matrix_maker(value,n) -> np.array:
    x = [[[value for k in range(n)] for j in range(n)] for i in range(n)]
    matrix= np.array(x)
    return matrix

def density12(blocks) -> np.array:
    input_ = matrix_maker(0.1,12)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                loc=[i,j,k] 
                loc_input =  [4*loc[0]+2,4*loc[1]+2,4*loc[2]+2]
                input_[loc_input[0]-2:loc_input[0]+2,loc_input[1]-2:loc_input[1]+2,loc_input[2]-2:loc_input[2]+2] = blocks[loc[0],loc[1],loc[2]]
    return input_

def density(input_) -> np.array:
    blocks = matrix_maker(0.1, 3)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                loc=[i,j,k] 
                loc_input =  [4*loc[0]+2,4*loc[1]+2,4*loc[2]+2]
                blocks[loc[0],loc[1],loc[2]] = np.mean(input_[loc_input[0]-2:loc_input[0]+2,
                                                           loc_input[1]-2:loc_input[1]+2,
                                                           loc_input[2]-2:loc_input[2]+2]) 
    blocks=blocks.round(1)            
    return blocks  

def RanSampling(n_samples, target):
    target_upper=1.05*target
    target_lower=0.95*target
    Y_total = data['yield']
    E_data=dataE['E'][dataE['E']<target_upper]
    E_data=E_data[E_data>target_lower]
    print(E_data)
    if len(E_data) == 0:
        Y_max=65
    else:
        Y_new=data['yield'].iloc[E_data.index]
        Y_max_idx=np.argmax(Y_new)
        Y_max=Y_new.iloc[Y_max_idx]
    print('the max yield for E = {} is {}, sampling start!'.format(target, Y_max))
    batchsize = 20000
    sample_z = np.random.randint(1,8,size=[n_samples,3,3,3])/10
    sample_target=[]
    sample_Y=[]
    print('decoding started...')
    for i in tqdm(range(0, n_samples, batchsize)): 
      temp_s0=sample_z[i:i+batchsize]
      temp=[]
      for i in range(len(temp_s0)):
          x1=density12(temp_s0[i])
          temp.append(x1)
      temp_s=np.asarray(temp)
      temp_s=np.expand_dims(temp_s,axis=-1)
      temp_E=[]
      temp_E=ensemble_predict_E(temp_s)
      try:
        E_target=temp_E['E'][temp_E['E']<target_upper]
        E_target=E_target[E_target>target_lower]
        sample_=temp_s[E_target.index]
        uniform_rand = np.random.uniform(size=len(sample_))
        uniform_Y = up*Y_max + uniform_rand*(1-up)*Y_max/5
        temp_Y = ensemble_predict_Y(sample_).values
        accepted = uniform_Y.reshape(-1,1) < temp_Y.reshape(-1,1)
        acc_idx = accepted.reshape(-1)
        acc_sample_S = sample_[acc_idx]
        acc_sample_Y = temp_Y[acc_idx]
        if len(acc_sample_S)>0:
          print('strcuture sampled!',acc_sample_Y)
          sample_target.append(acc_sample_S)
          sample_Y.append(acc_sample_Y)
      except:
        continue
    print('decoding completed!') 
    try:
      sample_S_final = [item for sublist in sample_target for item in sublist]
      sample_S_final = np.asarray(sample_S_final)
      sample_Y_final =[item for sublist in sample_Y for item in sublist]
      sample_Y_final = pd.DataFrame(sample_Y_final)
      sample_Y_final.columns=['Y']
      print('size of target sample is {}'.format(sample_S_final.shape)) 
    except:
      print('no valid structure!')
      sample_Y_final=[]
      sample_S_final=[]
    return sample_S_final, sample_Y_final

#Import data
matrix=np.load(name1+"Matrix_all.npy", allow_pickle=True)
dataE=pd.read_csv(name1+"E_all.csv")
E_total = dataE['E']
data=pd.read_csv(name1+"yield_all.csv")
i=len(data)
X = matrix.reshape(i,12,12,12,1)

#Main
sample_S, sample_Y = RanSampling(n_samples=sam_, target=target )

matrix333=[]
for i in sample_S:
  matrix333.append(density(i))
matrix333=np.asarray(matrix333)#All matrices that meet the requirements

pred_E=ensemble_predict_E(sample_S)
pred_Y=ensemble_predict_Y(sample_S)
pred_E=np.asarray(pred_E)
pred_E=pred_E.reshape(-1)
pred_Y=np.asarray(pred_Y)
pred_Y=pred_Y.reshape(-1)

#Pick the matrices with the highest yield strength
top=20 
ind = np.argpartition(pred_Y, -top)[-top:]
matrix_10=matrix333[ind]#Top 10 matrices
Y_10=pred_Y[ind]#Corresponding yield strength
E_10=pred_E[ind]#Corresponding elastic modulus
print(Y_10,E_10)

np.save(name3+'TopMatrix.npy',matrix_10, allow_pickle=True)

import scipy.io as io
io.savemat(name3+'TopMatrix_RandomSearch.mat',{'data':matrix_10})

plt.figure()
plt.scatter(pred_E,pred_Y)

# ind = np.argpartition(pred_Y, -top)[-top:]

def get_distance_from_point_to_line(point, line_point1, line_point2):
    #对于两点坐标为同一点时,返回点与点的距离
    if line_point1 == line_point2:
        point_array = np.array(point )
        point1_array = np.array(line_point1)
        return np.linalg.norm(point_array -point1_array )
    #计算直线的三个参数
    A = line_point2[1] - line_point1[1]
    B = line_point1[0] - line_point2[0]
    C = (line_point1[1] - line_point2[1]) * line_point1[0] + \
        (line_point2[0] - line_point1[0]) * line_point1[1]
    #根据点到直线的距离公式计算距离
    distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A**2 + B**2))
    return distance

dist_all=[]
for i in range(len(pred_E)):
    point=[pred_E[i],pred_Y[i]]
    dist=get_distance_from_point_to_line(point, [2400,6], [2630,6])
    dist_all.append(dist)
top=20 
ind = np.argpartition(dist_all, -top)[-top:]

matrix_10=matrix333[ind]#Top 10 matrices
Y_10=pred_Y[ind]#Corresponding yield strength
E_10=pred_E[ind]#Corresponding elastic modulus
plt.scatter(E_10,Y_10)
plt.figure()
plt.scatter(E_10,Y_10)

np.save(name3+'TopMatrix.npy',matrix_10, allow_pickle=True)
io.savemat(name3+'TopMatrix_RandomSearch.mat',{'data':matrix_10})