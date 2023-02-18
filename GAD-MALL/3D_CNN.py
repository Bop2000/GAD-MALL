from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

#Build a 3D convolutional neural network model
def get_model(width=60, height=60, depth=60):
    inputs = keras.Input((width, height, depth, 1))
    x = layers.Conv3D(filters=8, kernel_size=3, activation="elu",padding='same')(inputs)
    x = layers.MaxPool3D(pool_size=2,padding='same')(x)
    x = layers.Conv3D(filters=4, kernel_size=3, activation="elu",padding='same')(x)
    x = layers.MaxPool3D(pool_size=2,padding='same')(x)
    x = layers.Conv3D(filters=2, kernel_size=3, activation="elu",padding='same')(x)
    x = layers.MaxPool3D(pool_size=2,padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation="elu")(x)
    x = layers.Dense(units=64, activation="elu")(x)
    x = layers.Dense(units=32, activation="elu")(x)
    outputs = layers.Dense(units=1, activation="linear")(x)
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


#1. 3D CNN model for elastic modulus predicting

#Import data
matrix=np.load("data/Matrix60.npy", allow_pickle=True)
Data=pd.read_csv("data/E.csv")
X = matrix.reshape(len(Data),60,60,60,1)

#Split
X_train, X_test, y_train, y_test = train_test_split(X, Data['E'].values, test_size=0.2, random_state=0)

# Build model.
model = get_model(width=60, height=60, depth=60)

#Training
optimizer = keras.optimizers.Adam(learning_rate=0.005)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=["mean_absolute_error"])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=30)
mc = ModelCheckpoint("model/3dCNN_E.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)
history=model.fit(X_train, y_train, validation_data=(X_test, y_test),  batch_size=32, epochs=500, callbacks=[es,mc])


#2. 3D CNN model for yiled strength predicting
 
#Import data   
matrix=np.load("data/Matrix60.npy", allow_pickle=True)
Data2=pd.read_csv("data/yield.csv")
X = matrix.reshape(len(Data2),60,60,60,1)

#Split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, Data2['yield'].values, test_size=0.2, random_state=1)
# Build model.
model2 = get_model(width=60, height=60, depth=60)
#Training
optimizer = keras.optimizers.Adam(learning_rate=0.005)
model2.compile(optimizer=optimizer, loss='mean_squared_error', metrics=["mean_absolute_error"])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=30)
mc = ModelCheckpoint("model/3dCNN_Y.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)
history=model2.fit(X_train2, y_train2, validation_data=(X_test2, y_test2),  batch_size=16, epochs=500, callbacks=[es,mc])

  