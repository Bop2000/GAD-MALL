from bayes_opt import BayesianOptimization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
matrix=np.load('CNN_data/Matrix60.npy', allow_pickle=True)
Emax=pd.read_csv('CNN_data/E.csv')
i=1
X = matrix.reshape(len(matrix),60,60,60,1)
Y=Emax['E'].values

X_train, X_test, y_train, y_test = train_test_split(X, Emax['E'].values, test_size=0.2, random_state=1)

def cnn3d_model(f_size, k_size):
    """Build a 3D convolutional neural network model."""
    width=60
    height=60
    depth=60
    inputs = keras.Input((width, height, depth, 1))
    x = layers.Conv3D(filters=f_size, kernel_size=k_size, activation="elu",padding='same')(inputs)
    x = layers.MaxPool3D(pool_size=2,padding='same')(x)
    x = layers.Conv3D(filters=f_size/2, kernel_size=k_size, activation="elu",padding='same')(x)
    x = layers.MaxPool3D(pool_size=2,padding='same')(x)
    x = layers.Conv3D(filters=f_size/4, kernel_size=k_size, activation="elu",padding='same')(x)
    x = layers.MaxPool3D(pool_size=2,padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation="elu")(x)
    x = layers.Dense(units=64, activation="elu")(x)
    x = layers.Dense(units=32, activation="elu")(x)
    outputs = layers.Dense(units=1, activation="linear")(x)
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

def train_model(k_size, f_size, lr):
    k_size = int(k_size)
    f_size = int(f_size)
    # b_size = int(b_size)
    # module__w = int(module__w) # number of hidden layers
    b_size = 16
    # Augment the on the fly during training.
    cnn3d = cnn3d_model(k_size, f_size)
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    cnn3d.compile(optimizer=optimizer, loss='mse')
    re=ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=10, verbose=0,
        mode='auto', min_delta=0.0001, cooldown=0, min_lr=0
        )
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
    history = cnn3d.fit(X_train,y_train, epochs=5000, batch_size=b_size, shuffle=True, validation_data=(X_test, y_test),callbacks=[es,re])
    cnn3d.summary()
    min_loss = min(history.history['val_loss'])
    return -min_loss

bounds = {'k_size':(2,8), 'f_size':(4,32), 'lr':(0.0005,0.01)}
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
table.to_csv('CNN_results/CNN_performance.csv',index=False)

