from tensorflow.compat.v1 import ConfigProto, InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth=True
session=InteractiveSession(config=config)
import seaborn as sns
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from bayes_opt import BayesianOptimization
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
path = os.getcwd()
name1=path+'/Datasets/'
name2=path+'/Models/'
name3=path+'/Results/'

targetE=2500

init_points=500
n_iter=4000

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

###load models
n_model=10
model_E1= keras.models.load_model(name2+"3dCNN_E1.h5")
model_E2= keras.models.load_model(name2+"3dCNN_E2.h5")
model_E3= keras.models.load_model(name2+"3dCNN_E3.h5")
model_E4= keras.models.load_model(name2+"3dCNN_E4.h5")
model_E5= keras.models.load_model(name2+"3dCNN_E5.h5")
model_E6= keras.models.load_model(name2+"3dCNN_E6.h5")
model_E7= keras.models.load_model(name2+"3dCNN_E7.h5")
model_E8= keras.models.load_model(name2+"3dCNN_E8.h5")
model_E9= keras.models.load_model(name2+"3dCNN_E9.h5")
model_E10= keras.models.load_model(name2+"3dCNN_E10.h5")

model_y1= keras.models.load_model(name2+"3dCNN_y1.h5")
model_y2= keras.models.load_model(name2+"3dCNN_y2.h5")
model_y3= keras.models.load_model(name2+"3dCNN_y3.h5")
model_y4= keras.models.load_model(name2+"3dCNN_y4.h5")
model_y5= keras.models.load_model(name2+"3dCNN_y5.h5")
model_y6= keras.models.load_model(name2+"3dCNN_y6.h5")
model_y7= keras.models.load_model(name2+"3dCNN_y7.h5")
model_y8= keras.models.load_model(name2+"3dCNN_y8.h5")
model_y9= keras.models.load_model(name2+"3dCNN_y9.h5")
model_y10= keras.models.load_model(name2+"3dCNN_y10.h5")

def predprocessE(model,S):
    temp_E=model.predict(S)
    temp_E=pd.DataFrame(temp_E)
    temp_E.columns=['E']
    return temp_E

def predprocessY(model,S):
    temp_Y=model.predict(S)
    temp_Y=pd.DataFrame(temp_Y)
    temp_Y.columns=['yield']
    return temp_Y

def ensemble_predict_E(S):
    temp1=predprocessE(model_E1,S)
    temp2=predprocessE(model_E2,S)
    temp3=predprocessE(model_E3,S)
    temp4=predprocessE(model_E4,S)
    temp5=predprocessE(model_E5,S)
    temp6=predprocessE(model_E6,S)
    temp7=predprocessE(model_E7,S)
    temp8=predprocessE(model_E8,S)
    temp9=predprocessE(model_E9,S)
    temp10=predprocessE(model_E10,S)
    E_all=temp1+temp2+temp3+temp4+temp5+temp6+temp7+temp8+temp9+temp10
    E_all['E']/=n_model
    return E_all               

def ensemble_predict_Y(S):
    temp1=predprocessY(model_y1,S)
    temp2=predprocessY(model_y2,S)
    temp3=predprocessY(model_y3,S)
    temp4=predprocessY(model_y4,S)
    temp5=predprocessY(model_y5,S)
    temp6=predprocessY(model_y6,S)
    temp7=predprocessY(model_y7,S)
    temp8=predprocessY(model_y8,S)
    temp9=predprocessY(model_y9,S)
    temp10=predprocessY(model_y10,S)
    Y_all=temp1+temp2+temp3+temp4+temp5+temp6+temp7+temp8+temp9+temp10
    Y_all['yield']/=n_model
    return Y_all 

def train_model(x1,
                x2,
                x3, 
                x4,
                x5, 
                x6, 
                x7, 
                x8,
                x9,
                x10,
                x11,
                x12,
                x13,
                x14,
                x15,
                x16,
                x17,
                x18,
                x19,
                x20,
                x21,
                x22,
                x23,
                x24,
                x25,
                x26,
                x27,
                ):
    params = {
        "x1": round(x1,1),
        'x2': round(x2,1),
        'x3': round(x3,1),
        'x4': round(x4,1),
        'x5': round(x5,1),
        'x6': round(x6,1),
        'x7': round(x7,1),
        'x8': round(x8,1),
        'x9': round(x9,1),
        'x10': round(x10,1),
        'x11': round(x11,1),
        'x12': round(x12,1),
        'x13': round(x13,1),
        'x14': round(x14,1),
        'x15': round(x15,1),
        'x16': round(x16,1),
        'x17': round(x17,1),
        'x18': round(x18,1),
        'x19': round(x19,1),
        'x20': round(x20,1),
        'x21': round(x21,1),
        'x22': round(x22,1),
        'x23': round(x23,1),
        'x24': round(x24,1),
        'x25': round(x25,1),
        'x26': round(x26,1),
        'x27': round(x27,1),
                 }
    print(params)

#    model = LGBMRegressor(nfolds=5,**params)
#    model.fit(X_train, Y_train)
    x_test = [params['x1'],params['x2'],params['x3'],params['x4'],params['x5']
    ,params['x6'],params['x7'],params['x8'],params['x9'],params['x10'],params['x11']
    ,params['x12'],params['x13'],params['x14'],params['x15'],params['x16'],params['x17']
    ,params['x18'],params['x19'],params['x20'],params['x21'],params['x22'],params['x23']
    ,params['x24'],params['x25'],params['x26'],params['x27']]
    x_test=np.array(x_test)
    x_test=x_test.round(1)
    x_test=x_test.reshape(1,3,3,3)
    temp=[]
    for i in range(len(x_test)):
        x1=density12(x_test[i])
        temp.append(x1)
    x_test12=np.asarray(temp)
    x_test12=x_test12.reshape(1,12,12,12,1)
    predE = ensemble_predict_E(x_test12)
    predE1=predE['E'][0]
    target_upper=1.05*targetE
    target_lower=0.95*targetE
    if predE1<target_lower or predE1>target_upper:
        score=-2000-abs(predE1-targetE)
    else:
        predY = ensemble_predict_Y(x_test12)
        predY1=predY['yield'][0]
        # print(predY1)
        score=-(200-predY1)
    return score

bounds = {
          'x1': (0.1, 0.8),
          'x2': (0.1, 0.8),
          'x3': (0.1, 0.8),
          'x4': (0.1, 0.8),
          'x5': (0.1, 0.8),
          'x6': (0.1, 0.8),
          'x7': (0.1, 0.8),
          'x8': (0.1, 0.8),
          'x9': (0.1, 0.8), 
          'x10': (0.1, 0.8),
          'x11': (0.1, 0.8),
          'x12':(0.1,0.8),
          'x13':(0.1,0.8),
          'x14':(0.1,0.8),
          'x15':(0.1,0.8),
          'x16':(0.1,0.8),
          'x17':(0.1,0.8),
          'x18':(0.1,0.8),
          'x19':(0.1,0.8),
          'x20':(0.1,0.8),
          'x21':(0.1,0.8),
          'x22':(0.1,0.8),
          'x23':(0.1,0.8),
          'x24':(0.1,0.8),
          'x25':(0.1,0.8),
          'x26':(0.1,0.8),
          'x27':(0.1,0.8),
          
          }
optimizer = BayesianOptimization(
    f=train_model,
    pbounds=bounds,
    random_state=1,
)
optimizer.maximize(init_points=init_points, n_iter=n_iter)

optimizer.max

print(optimizer.max)

target_all=[]
params_all=[]
for i, res in enumerate(optimizer.res):
    target=res['target']
    target_all.append(target)
    
    params=[]
    for n in range(1,28):                                 
        temp=res['params']["x%d"%(n)]
        params.append(temp)
    params=np.array(params)
    params=params.round(1)
    params_all.append(params)
    
target_all=np.array(target_all)
params_all=np.array(params_all)


matrix12=[]
for i in params_all:
  matrix12.append(density12(i.reshape(3,3,3)))
matrix12=np.asarray(matrix12)#All matrices that meet the requirements

pred_E=ensemble_predict_E(matrix12)
pred_Y=ensemble_predict_Y(matrix12)
pred_E=np.asarray(pred_E)
pred_E=pred_E.reshape(-1)
pred_Y=np.asarray(pred_Y)
pred_Y=pred_Y.reshape(-1)

plt.figure()
plt.scatter(pred_E[init_points:],pred_Y[init_points:])

######
# whe=np.where(pred_E<2500)
# pred_E=pred_E[whe[0]]
# pred_Y=pred_Y[whe[0]]
# target_all=target_all[whe[0]]
######

top=20
ind = np.argpartition(target_all, -top)[-top:]
target_top=target_all[ind]
params_top=params_all[ind]
Y_10=pred_Y[ind]#Corresponding yield strength
E_10=pred_E[ind]#Corresponding elastic modulus

plt.figure()
plt.scatter(E_10,Y_10)

plt.figure()
# plt.scatter(pred_E[init_points:],pred_Y[init_points:])
plt.scatter(pred_E,pred_Y)
plt.scatter(E_10,Y_10)

p_top=params_top.reshape(len(params_top),3,3,3)
np.save(name3+'TopMatrix.npy',p_top, allow_pickle=True)

import scipy.io as io
io.savemat(name3+'TopMatrix_BO.mat',{'data':p_top})
