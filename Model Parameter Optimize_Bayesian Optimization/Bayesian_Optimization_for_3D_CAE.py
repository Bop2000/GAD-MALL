import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.layers import Flatten, add
from tensorflow.keras.layers import BatchNormalization 
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

matrix=np.load('AE_data/3D_CAE_Train.npy', allow_pickle=True)
Emax=range(len(matrix))
X = matrix.reshape(17835,12,12,12,1)
X_train, X_test, y_train, y_test = train_test_split(X, Emax, test_size=0.2, random_state=1)

def train_preprocessing(volume, affinity):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    #volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, affinity

def validation_preprocessing(volume, affinity):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, affinity
t = time.localtime()
table = pd.DataFrame(columns=['target', 'k_size', 'f_size', 'pool_size', 'drop_size', 'dense_size'])

def Conv2d_BN(x, nb_filter, kernel_size, strides=1, padding='same', name=None):
    x = layers.Conv2D(nb_filter, kernel_size, padding=padding,  strides=strides,
                     activation=LeakyReLU())(x)
    x = BatchNormalization()(x)
    return x

def identity_Block(inpt, nb_filter, kernel_size, strides=1, with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides,
                             kernel_size=kernel_size)
        x = Dropout(0.5)(x)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x
    
def res_model(k_size, f_size):
    """Build a 3D convolutional neural network model."""

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
    
   # Define the model.
    model = keras.Model(input_gyroid,decoded, name="3dautoencoder")

    return model

def train_model(k_size, f_size, lr):
    k_size = int(k_size)
    f_size = int(f_size)
    # b_size = int(b_size)
    # l_rate = int(l_rate)
    # module__w = int(module__w) # number of hidden layers
    b_size = 64
    # Augment the on the fly during training.
    autoencoder = res_model(k_size, f_size)
    
    # autoencoder = keras.Model(input_gyroid, decoded)
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    autoencoder.compile(optimizer=optimizer, loss='mse')
    re=ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=10, verbose=0,
        mode='auto', min_delta=0.0001, cooldown=0, min_lr=0
        )
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    history = autoencoder.fit(X_train, X_train, epochs=500, batch_size=b_size, shuffle=True, validation_data=(X_test, X_test),callbacks=[es,re])
    autoencoder.summary()
    min_loss = min(history.history['val_loss'])
    # h_params=[k_size, f_size, lr, min_loss]
    # h_params=pd.DateFrame(h_params)
    # h_params.to_csv('h_params_{}_{}_{}_{}.csv'.format(min_loss, k_size, f_size, lr)
    return -min_loss
            
bounds = {'k_size':(3,8), 'f_size':(16,64), 'lr':(0.0001,0.01)}
optimizer = BayesianOptimization(
    f=train_model,
    pbounds=bounds,
    random_state=1,
)

optimizer.maximize(init_points=100, n_iter=100)
print(optimizer.max)
table = pd.DataFrame(columns=['target', 'k_size', 'f_size', 'lr'])

for res in optimizer.res:
    table=table.append(pd.DataFrame({'target':[res['target']],'k_size':[res['params']['k_size']],
                                      'f_size':[res['params']['f_size']],'lr':[res['params']['lr']]}),ignore_index=True)
table.to_csv('AE_results/AE_performance.csv',index=False)
