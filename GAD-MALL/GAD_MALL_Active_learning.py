from sklearn.mixture import GaussianMixture
import tensorflow
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import Input,Model
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import math
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from prettytable import PrettyTable


#Choose a elastic modulus target, such as target = 2500 MPa
target = 2500
target_upper=1.05*target
target_lower=0.95*target
sam_=1000000#Sampling number
up=0.8
b_size=2000
n_size=6
n_accu=60
pi=3.14159265358979323846264
xxx=(1/2)*2*pi
sizeofdata0=[3,3,3]
accu=20
x_axis, y_axis,z_axis = np.linspace(n_size/(n_accu*2), n_size-n_size/(n_accu*2), n_accu),  np.linspace(n_size/(n_accu*2), n_size-n_size/(n_accu*2), n_accu),  np.linspace(n_size/(n_accu*2), n_size-n_size/(n_accu*2), n_accu)
x, y,z = np.meshgrid(x_axis, y_axis,z_axis)

#Import 3D-CAE model
autoencoder= tensorflow.keras.models.load_model('model/3D_CAE_model.h5')

#Import data
matrix=np.load("data/Matrix12.npy", allow_pickle=True)
dataE=pd.read_csv("data/E.csv")
E_total = dataE['E']
data=pd.read_csv("data/yield.csv")
i=len(data)
X = matrix.reshape(i,12,12,12,1)

encoded_input = Input(shape=(1,1,1,8))
deco = autoencoder.layers[-7](encoded_input)
deco = autoencoder.layers[-6](deco)
deco = autoencoder.layers[-5](deco)
deco = autoencoder.layers[-4](deco)
deco = autoencoder.layers[-3](deco)
deco = autoencoder.layers[-2](deco)
deco = autoencoder.layers[-1](deco)
# create the decoder model
decoder = Model(encoded_input, deco)

input = Input(shape=(12,12,12,1))
enco = autoencoder.layers[1](input)
enco = autoencoder.layers[2](enco)
enco = autoencoder.layers[3](enco)
enco = autoencoder.layers[4](enco)
enco = autoencoder.layers[5](enco)
enco = autoencoder.layers[6](enco)
enco = autoencoder.layers[7](enco)
# create the encoder model
encoder = Model(input, enco)

embed=encoder(X)
embed_all=embed[:,0].numpy()
embed_all=embed_all[:,0]
embed_all=embed_all[:,0]

#Average negative log likelihood
scores=[]
for i in range(1,12):
  gm = GaussianMixture(n_components=i, random_state=0, init_params='kmeans').fit(embed_all)
  print('Average negative log likelihood:', -1*gm.score(embed_all))
  scores.append(-1*gm.score(embed_all))
plt.figure()
plt.scatter(range(1,12), scores)
plt.plot(range(1,12),scores)
gm = GaussianMixture(n_components=4, random_state=0, init_params='kmeans').fit(embed_all) #plot a n_components v.s. Average negative log likelihood
print('Average negative log likelihood:', -1*gm.score(embed_all))

def Structure(x1,decoder):
  x1=np.expand_dims(x1,axis=1)
  x1=np.expand_dims(x1,axis=1)
  x1=np.expand_dims(x1,axis=1)
  recon=decoder(x1)
  new_x=recon.numpy()
  new_x1=new_x.round(1)
  return new_x1

def ensemble_predict_E(S):
    modelname = "model/3dCNN_E.h5"
    model_E = keras.models.load_model(modelname)
    E=model_E.predict(S)
    E=pd.DataFrame(E)
    E.columns=['E']
    return E               

def ensemble_predict_Y(S):
    modelname = "model/3dCNN_Y.h5"
    model_Y = keras.models.load_model(modelname)
    Y=model_Y.predict(S)
    Y=pd.DataFrame(Y)
    Y.columns=['yield']
    return Y  

def matrix_maker(value,n) -> np.array:
    temp_x = [[[value for k in range(n)] for j in range(n)] for i in range(n)]
    matrix= np.array(temp_x)
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

def density2(input_,blocks) -> np.array:
    for i in range(3):
        for j in range(3):
            for k in range(3):
                loc=[i,j,k] 
                loc_input =  [4*loc[0]+2,4*loc[1]+2,4*loc[2]+2]
                input_[loc_input[0]-2:loc_input[0]+2,loc_input[1]-2:loc_input[1]+2,loc_input[2]-2:loc_input[2]+2] = blocks[loc[0],loc[1],loc[2]]
    return input_

def findneighbour(inputdata,position):
    neighbourhoods=np.zeros((3,3,3))
    neighbourhoods[:,:,:]=np.nan
    r=len(inputdata)
    flag=0
    for i in range(r):
        if inputdata[i,0]==position[0] and inputdata[i,1]==position[1] and inputdata[i,2]==position[2]:
            flag=1
    if flag!=0:
        for i in range(r):
            dertax=inputdata[i,0]-position[0]
            dertay=inputdata[i,1]-position[1]
            dertaz=inputdata[i,2]-position[2]
            if abs(dertax)<=1 and abs(dertay)<=1 and abs(dertaz)<=1:
                neighbourhoods[int(dertax+1),int(dertay+1),int(dertaz+1)]=inputdata[i,3]
    return neighbourhoods

def createunitofv(datainput,positon,nofv,dofv):
    neibourhoods=findneighbour(datainput,positon)
    unitofv=np.ones((nofv-2*dofv,nofv-2*dofv,nofv-2*dofv))
    if not np.isnan(neibourhoods[1,1,1]):
        unitofv=unitofv*neibourhoods[1,1,1]
    else:
        unitofv=np.zeros((nofv,nofv,nofv))
        unitofv[:,:,:]=np.nan
        return unitofv
    if np.isnan(neibourhoods[2,1,1]):
        neibourhoods[2,1,1]=neibourhoods[1,1,1]
    if np.isnan(neibourhoods[0,1,1]):
        neibourhoods[0,1,1]=neibourhoods[1,1,1]
    if np.isnan(neibourhoods[1,2,1]):
        neibourhoods[1,2,1]=neibourhoods[1,1,1]
    if np.isnan(neibourhoods[1,0,1]):
        neibourhoods[1,0,1]=neibourhoods[1,1,1]
    if np.isnan(neibourhoods[1,1,2]):
        neibourhoods[1,1,2]=neibourhoods[1,1,1]
    if np.isnan(neibourhoods[1,1,0]):
        neibourhoods[1,1,0]=neibourhoods[1,1,1]
    if np.isnan(neibourhoods[2,2,1]):
        neibourhoods[2,2,1]=(neibourhoods[2,1,1]+neibourhoods[1,2,1])/2
    if np.isnan(neibourhoods[2,0,1]):
        neibourhoods[2,0,1]=(neibourhoods[2,1,1]+neibourhoods[1,0,1])/2
    if np.isnan(neibourhoods[0,2,1]):
        neibourhoods[0,2,1]=(neibourhoods[0,1,1]+neibourhoods[1,2,1])/2
    if np.isnan(neibourhoods[0,0,1]):
        neibourhoods[0,0,1]=(neibourhoods[0,1,1]+neibourhoods[1,0,1])/2
    if np.isnan(neibourhoods[2,1,2]):
        neibourhoods[2,1,2]=(neibourhoods[2,1,1]+neibourhoods[1,1,2])/2
    if np.isnan(neibourhoods[2,1,0]):
        neibourhoods[2,1,0]=(neibourhoods[2,1,1]+neibourhoods[1,1,0])/2
    if np.isnan(neibourhoods[0,1,2]):
        neibourhoods[0,1,2]=(neibourhoods[0,1,1]+neibourhoods[1,1,2])/2
    if np.isnan(neibourhoods[0,1,0]):
        neibourhoods[0,1,0]=(neibourhoods[0,1,1]+neibourhoods[1,1,0])/2
    if np.isnan(neibourhoods[1,2,2]):
        neibourhoods[1,2,2]=(neibourhoods[1,2,1]+neibourhoods[1,1,2])/2
    if np.isnan(neibourhoods[1,2,0]):
        neibourhoods[1,2,0]=(neibourhoods[1,2,1]+neibourhoods[1,1,0])/2
    if np.isnan(neibourhoods[1,0,2]):
        neibourhoods[1,0,2]=(neibourhoods[1,0,1]+neibourhoods[1,1,2])/2
    if np.isnan(neibourhoods[1,0,0]):
        neibourhoods[1,0,0]=(neibourhoods[1,0,1]+neibourhoods[1,1,0])/2
    if np.isnan(neibourhoods[0,0,0]):
        neibourhoods[0,0,0]=(neibourhoods[0,1,1]+neibourhoods[1,0,1]+neibourhoods[1,1,0])/3
    if np.isnan(neibourhoods[2,0,0]):
        neibourhoods[2,0,0]=(neibourhoods[2,1,1]+neibourhoods[1,0,1]+neibourhoods[1,1,0])/3
    if np.isnan(neibourhoods[0,2,0]):
        neibourhoods[0,2,0]=(neibourhoods[0,1,1]+neibourhoods[1,2,1]+neibourhoods[1,1,0])/3
    if np.isnan(neibourhoods[0,0,2]):
        neibourhoods[0,0,2]=(neibourhoods[0,1,1]+neibourhoods[1,0,1]+neibourhoods[1,1,2])/3
    if np.isnan(neibourhoods[0,2,2]):
        neibourhoods[0,2,2]=(neibourhoods[0,1,1]+neibourhoods[1,2,1]+neibourhoods[1,1,2])/3
    if np.isnan(neibourhoods[2,0,2]):
        neibourhoods[2,0,2]=(neibourhoods[2,1,1]+neibourhoods[1,0,1]+neibourhoods[1,1,2])/3
    if np.isnan(neibourhoods[2,2,0]):
        neibourhoods[2,2,0]=(neibourhoods[2,1,1]+neibourhoods[1,2,1]+neibourhoods[1,1,0])/3
    if np.isnan(neibourhoods[2,2,2]):
        neibourhoods[2,2,2]=(neibourhoods[2,1,1]+neibourhoods[1,2,1]+neibourhoods[1,1,2])/3
    for i in range(dofv):
        nownumber=neibourhoods[1,1,1]+i*(neibourhoods-neibourhoods[1,1,1])/(2*dofv+1)
        temp=np.zeros((1,nofv-2*dofv+2*i,nofv-2*dofv+2*i))
        temp[:,:,:]=nownumber[2,1,1]
        unitofv=np.concatenate((unitofv,temp),axis=0)#x+
        temp[:,:,:]=nownumber[0,1,1]
        unitofv=np.concatenate((temp,unitofv),axis=0)#x-
        temp=np.zeros((nofv-2*dofv+2*i+2,1,nofv-2*dofv+2*i))
        temp[:,:,:]=nownumber[1,2,1]
        unitofv=np.concatenate((unitofv,temp),axis=1)#y+
        temp[:,:,:]=nownumber[1,0,1]
        unitofv=np.concatenate((temp,unitofv),axis=1)#y-
        temp=np.zeros((nofv-2*dofv+2*i+2,nofv-2*dofv+2*i+2,1))
        temp[:,:,:]=nownumber[1,1,2]
        unitofv=np.concatenate((unitofv,temp),axis=2)#z+
        temp[:,:,:]=nownumber[1,1,0]
        unitofv=np.concatenate((temp,unitofv),axis=2)#z-      
        unitofv[[-1],[-1],:]=nownumber[2,2,1]#x+,y+
        unitofv[0,0,:]=nownumber[0,0,1]#x-,y-
        unitofv[[-1],0,:]=nownumber[2,0,1]#x+,y-
        unitofv[0,[-1],:]=nownumber[0,2,1]#x,y+  
        unitofv[[-1],:,[-1]]=nownumber[2,1,2]
        unitofv[0,:,0]=nownumber[0,1,0]
        unitofv[[-1],:,0]=nownumber[2,1,0]
        unitofv[0,:,[-1]]=nownumber[0,1,2]    
        unitofv[:,[-1],[-1]]=nownumber[1,2,2]
        unitofv[:,0,0]=nownumber[1,0,0]
        unitofv[:,[-1],0]=nownumber[1,2,0]
        unitofv[:,0,[-1]]=nownumber[1,0,2]
        unitofv[[-1],[-1],[-1]]=nownumber[2,2,2]
        unitofv[0,[-1],[-1]]=nownumber[0,2,2]
        unitofv[[-1],0,[-1]]=nownumber[2,0,2]
        unitofv[[-1],[-1],0]=nownumber[2,2,0]
        unitofv[[-1],0,0]=nownumber[2,0,0]
        unitofv[0,[-1],0]=nownumber[0,2,0]
        unitofv[0,0,[-1]]=nownumber[0,0,2]
        unitofv[0,0,0]=nownumber[0,0,0]
    return unitofv

def createv_2(data,sizeofdata,nofv,dofv):
    v=[]
    for k in range(sizeofdata[2]):
        temp2=[]
        for j in range(sizeofdata[1]):
            temp1=[]
            for i in range(sizeofdata[0]):
                position=[i,j,k]
                varray=createunitofv(data,position,nofv,dofv)
                if i<1:
                    temp1=varray
                else:
                    temp1=np.concatenate((temp1,varray),axis=0)
            if j<1:
                temp2=temp1
            else:
                temp2=np.concatenate((temp2,temp1),axis=1)
        if k<1:
            v=temp2
        else:
            v=np.concatenate((v,temp2),axis=2)
    return v

############
r1=np.zeros((27,3))
for a in range(3):
    for b in range(3):
        for c in range(3):
            r1[9*a+3*b+c,0]=a
            r1[9*a+3*b+c,1]=b
            r1[9*a+3*b+c,2]=c
#############
oo=np.sin(pi*x)*np.cos(pi*y)+np.sin(pi*y)*np.cos(pi*z)+np.sin(pi*z)*np.cos(pi*x)
def To60(matrix):
    the606060=[]
    N=len(matrix)
    # r1_100=np.tile(r1, (N,1,1))
    finished=(10*(1-matrix).reshape(N,27,1))*0.282-0.469
    # print(finished.shape)
    # data_all=np.concatenate((r1_100,finished),axis=2)
    for l in range(N):
        r2=finished[l]
        data0=np.concatenate((r1,r2),axis=1)
        v=createv_2(data0,sizeofdata0,accu,3)
        ov=oo+v
        the606060.append(ov)
    the606060_cell=np.asarray(the606060)
    the606060_cell=np.where(the606060_cell<0.9,1,0)
    return the606060_cell

matrix=matrix.reshape(len(matrix),12,12,12)
input_=[]
for i in range(len(matrix)):
    temp_x=matrix[i]
    xx=density(temp_x)
    input_.append(xx)
matrix2=np.array(input_)
matrix60=To60(matrix2)
mean_try=np.mean(matrix60.reshape(len(matrix60),60*60*60),axis=1)

def rejSampling(gm, n_samples, target):
    target_upper=1.05*target
    target_lower=0.95*target
    Y_total = data['yield']
    E_data=dataE['E'][dataE['E']<target_upper]
    E_data=E_data[E_data>target_lower]
    print(E_data)
    if len(E_data) == 0:
        Y_max=24
    else:
        Y_new=data['yield'].iloc[E_data.index]
        Y_max_idx=np.argmax(Y_new)
        Y_max=Y_new.iloc[Y_max_idx]
    print('the max yield for E = {} is {}, sampling start!'.format(target, Y_max))
    batchsize = b_size
    sample_z = gm.sample(n_samples)[0]
    sample_target=[]
    sample_Y=[]
    print('decoding started...')
    for i in tqdm(range(0, n_samples, batchsize)): 
      temp_s0=Structure(sample_z[i:i+batchsize],decoder)
      temp_s3=[]
      for i in range(len(temp_s0)):
          temp_x=density(temp_s0[i])
          temp_s3.append(temp_x)
      temp_s=np.asarray(temp_s3)
      temp_s3=np.asarray(temp_s3)
      temp_s60=To60(temp_s3)
      temp_E=[]
      temp_E=ensemble_predict_E(temp_s60)
      try:
        E_target=temp_E['E'][temp_E['E']<target_upper]
        E_target=E_target[E_target>target_lower]
        sample_=temp_s[E_target.index]
        sample_60=np.asarray(sample_)
        sample_60=To60(sample_60)
        uniform_rand = np.random.uniform(size=len(sample_))
        uniform_Y = up*Y_max + uniform_rand*(1-up)*Y_max
        temp_Y = ensemble_predict_Y(sample_60).values
        accepted = uniform_Y.reshape(-1,1) < temp_Y.reshape(-1,1)
        acc_idx = accepted.reshape(-1)
        acc_sample_S = sample_[acc_idx]
        acc_sample_Y = temp_Y[acc_idx]
        if len(acc_sample_S)>0:
          print('strcuture sampled!')
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

###main
sample_S, sample_Y = rejSampling(gm, n_samples=sam_, target=target)
if len(sample_S)>1000:
    top=1000
    ind = np.argpartition(sample_Y, -top)[-top:]
    sample_S=sample_S[ind]
sample_Y.columns=['yield']#Corresponding yield strength
matrix333=sample_S.reshape(len(sample_S),27)
matrix_x=np.unique(matrix333,axis=0)
for i in range(27):
    matrix_x=matrix_x[(matrix_x[:,i]>0)&(matrix_x[:,i]<0.9)]
matrix_x=matrix_x.reshape(len(matrix_x),3,3,3,1)
X60=To60(matrix_x)
pred_E=ensemble_predict_E(X60)
pred_Y=ensemble_predict_Y(X60)
pred_E=np.asarray(pred_E)
pred_Y=np.asarray(pred_Y)
pred_Y=pred_Y.reshape(-1)

#Pick the matrices with the highest yield strength
top=20
ind = np.argpartition(pred_Y, -top)[-top:]
matrix_20=matrix_x[ind]#Top 20 porosity matrices
Y_20=pred_Y[ind]#Corresponding yield strength
E_20=pred_E[ind]#Corresponding elastic modulus
np.save('results/PorosityMatrices_top20.npy',matrix_20,allow_pickle=True)#Save top 20 porosity matrices and use Matlab to generate STL file for finite element simulation.
print('Sampling completed!')
table=PrettyTable(['No.','Elastic modulus (MPa)','Yield strength (MPa)'])
for i in range(len(E_20)):
    table.add_row([i+1,E_20[i][0],Y_20[i]])
print(table)
