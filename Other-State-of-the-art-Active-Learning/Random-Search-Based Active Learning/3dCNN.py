from tensorflow.compat.v1 import ConfigProto, InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth=True
session=InteractiveSession(config=config)
import seaborn as sns
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn import metrics
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
path = os.getcwd()
name1=path+'/Datasets/'
name2=path+'/Models/'

n_model=10

p_len=12
patience=50

def get_model(width=12, height=12, depth=12):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="elu",padding='same')(inputs)
    x = layers.MaxPool3D(pool_size=2,padding='same')(x)
    x = layers.Conv3D(filters=32, kernel_size=3, activation="elu",padding='same')(x)
    x = layers.MaxPool3D(pool_size=2,padding='same')(x)
    x = layers.Conv3D(filters=16, kernel_size=3, activation="elu",padding='same')(x)
    x = layers.MaxPool3D(pool_size=2,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=8, kernel_size=3, activation="elu",padding='same')(x)
    x = layers.MaxPool3D(pool_size=2,padding='same')(x)
    x = layers.Conv3D(filters=4, kernel_size=3, activation="elu",padding='same')(x)
    x = layers.MaxPool3D(pool_size=2,padding='same')(x)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=128, activation="elu")(x)
    # x = layers.Dense(units=64, activation="elu")(x)
    # x = layers.Dense(units=32, activation="elu")(x)
    outputs = layers.Dense(units=1, activation="linear")(x)
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

def get_model1(width=12, height=12, depth=12):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="elu",padding='same')(inputs)
    x = layers.MaxPool3D(pool_size=2,padding='same')(x)
    x = layers.Conv3D(filters=32, kernel_size=3, activation="elu",padding='same')(x)
    x = layers.MaxPool3D(pool_size=2,padding='same')(x)
    x = layers.Conv3D(filters=16, kernel_size=3, activation="elu",padding='same')(x)
    x = layers.MaxPool3D(pool_size=2,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=8, kernel_size=3, activation="elu",padding='same')(x)
    x = layers.MaxPool3D(pool_size=2,padding='same')(x)
    x = layers.Conv3D(filters=4, kernel_size=3, activation="elu",padding='same')(x)
    x = layers.MaxPool3D(pool_size=2,padding='same')(x)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=128, activation="elu")(x)
    # x = layers.Dense(units=64, activation="elu")(x)
    # x = layers.Dense(units=32, activation="elu")(x)
    outputs = layers.Dense(units=1, activation="linear")(x)
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

def plot_p(model,X,Y,i):
    pred=model.predict(X)
    R2=stats.pearsonr(pred.reshape(-1), Y)[0]
    R2=np.asarray(R2)
    R2=R2.round(5)
    plt.subplot(2,round(n_model/2),i)
    sns.set()
    sns.regplot(x=pred, y=Y, color='k') 
    plt.title(R2)

def cal_r2(model,X_test,y_test):
    y_pred=model.predict(X_test)
    acc=stats.pearsonr(y_pred.reshape(-1), y_test)
    acc=np.asarray(acc)
    acc=acc.reshape(1,2)
    y_pred_copy= y_pred.reshape(len(y_pred))
    y_test_copy=y_test.reshape(len(y_test))
    MAE= metrics.mean_absolute_error(y_test_copy, y_pred_copy)
    return acc,MAE

def fit_3dcnn(data):
    Val_MAE=[]
    Test_MAE=[]
    R2=np.empty(shape=(0,2))
    R2_test=np.empty(shape=(0,2))
    input_matrix=np.load(name1+"Matrix_train.npy", allow_pickle=True)
    input_matrix=input_matrix.reshape(len(input_matrix),p_len,p_len,p_len,1)
    m_test=np.load(name1+"Matrix_test.npy", allow_pickle=True)
    X_test2 = m_test.reshape(len(m_test),p_len,p_len,p_len,1)
    if data.columns=='E':
        data1=data['E'].values
        data_test=pd.read_csv(name1+"E_test.csv")
        y_test2= data_test['E'].values
    if data.columns=='yield':
        data1=data['yield'].values
        data_test=pd.read_csv(name1+"yield_test.csv")
        y_test2= data_test['yield'].values

    plt.figure()
    for i in range(1,n_model+1):
        X_train, X_test, y_train, y_test = train_test_split(input_matrix, data1, test_size=0.2, random_state=i+10)
        
        if data.columns=='E':
            model = get_model1(width=p_len)
            mc = ModelCheckpoint(name2+"3dCNN_E%d.h5"%(i), monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        if data.columns=='yield':
            model = get_model(width=p_len)
            mc = ModelCheckpoint(name2+"3dCNN_y%d.h5"%(i), monitor='val_loss', mode='min', verbose=1, save_best_only=True)
       
        model.summary()
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=["mean_absolute_error"])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=patience)
        history=model.fit(X_train, y_train, validation_data=(X_test, y_test),  batch_size=32, epochs=5000, callbacks=[es,mc])
        temp_R2,MAE=cal_r2(model,X_test,y_test)
        R2=np.append(R2,temp_R2,axis=0)
        Val_MAE.append(MAE)
        temp_R2,MAE=cal_r2(model,X_test2,y_test2)
        R2_test=np.append(R2_test,temp_R2,axis=0)
        Test_MAE.append(MAE)
        plot_p(model,X_test2,y_test2,i)
    return Val_MAE,R2,Test_MAE, R2_test


#####Main
#####E
DataE=pd.read_csv(name1+"E_train.csv")
MAE_valE,R2_valE,MAE_testE, R2_testE=fit_3dcnn(DataE)

#### Y
DataY=pd.read_csv(name1+"yield_train.csv")
MAE_valY,R2_valY,MAE_testY, R2_testY=fit_3dcnn(DataY)


model = get_model(width=p_len)
model.summary()