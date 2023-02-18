from tensorflow.compat.v1 import ConfigProto, InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth=True
session=InteractiveSession(config=config)
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#Import
matrix=np.load('data/3D_CAE_Train.npy', allow_pickle=True)
Ran=range(len(matrix))
X = matrix.reshape(17835,12,12,12,1)

#Split
X_train, X_test, y_train, y_test = train_test_split(X, Ran, test_size=0.2, random_state=1)

#Model parameters
b_size=64
k_size =4.31648385243556
f_size =60.3907825791688
lr =0.000753014797772
k_size = int(k_size)
f_size = int(f_size)
b_size=int(b_size)

#Model architecture
input_gyroid = keras.Input(shape = (12,12,12,1))
x = layers.Conv3D(f_size, (k_size,k_size,k_size), activation='elu', padding='same')(input_gyroid)
x = layers.MaxPooling3D((2,2,2), padding='same')(x)
x = layers.Conv3D(f_size/2, (k_size,k_size,k_size), activation='elu', padding='same')(x)
x = layers.MaxPooling3D((2,2,2), padding='same')(x)
x = layers.Conv3D(f_size/4, (k_size,k_size,k_size), activation='elu', padding='same')(x)
x = layers.Conv3D(8, (k_size,k_size,k_size), activation='elu', padding='same')(x)
encoded = layers.MaxPooling3D((3,3,3), padding='same', name='encoder')(x)
x = layers.Conv3D(f_size/4, (k_size,k_size,k_size), activation='elu', padding='same')(encoded)
x = layers.UpSampling3D((2,2,2))(x)
x = layers.Conv3D(f_size/2, (k_size,k_size,k_size), activation='elu', padding='same')(x)
x = layers.UpSampling3D((3,3,3))(x)
x = layers.Conv3D(f_size, (k_size,k_size,k_size), activation='elu', padding='same')(x)
x = layers.UpSampling3D((2,2,2))(x)
decoded = layers.Conv3D(1, (3,3,3), activation='linear', padding='same')(x)
autoencoder = keras.Model(input_gyroid, decoded)

#Training
optimizer = keras.optimizers.Adam(learning_rate=lr)
autoencoder.compile(optimizer=optimizer, loss='mse')
re=ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0,mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc = ModelCheckpoint('model/3D_CAE_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
history = autoencoder.fit(X_train, X_train, epochs=500, batch_size=b_size, shuffle=True, validation_data=(X_test, X_test),callbacks=[es,mc,re])