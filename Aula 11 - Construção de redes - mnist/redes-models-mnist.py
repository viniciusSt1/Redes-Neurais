import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.datasets import mnist
import numpy as np

(train_X,train_y),(test_X,test_y) = mnist.load_data()

def Densa(input_size,classes):
    inputShape = input_size
    model = Sequential()
    model.add(Dense(64, input_shape=(inputShape,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    #model.add(Dropout(taxa))
    model.add(Dense(classes, activation='softmax'))

    adam = optimizers.Adam(lr = 0.001)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def CNN(width, height, channels, classes):
    inputShape = (width, height, channels)
    model = Sequential()
    model.add(Conv2D(6, (3,3), activation='relu', padding='same', input_shape = inputShape)) #numero de mascaras, tamanho das mascaras
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(6, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(6, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())    #Passagem para as densas

    model.add(Dense(128, input_shape=(inputShape,), activation='relu'))
    model.add(Dense(64, input_shape=(inputShape,), activation='relu'))
    #model.add(Dropout(taxa))
    model.add(Dense(classes, activation='softmax'))

    adam = optimizers.Adam(lr = 0.001)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def LSTM_model(width, height, channels, classes):
    inputShape = (width, channels)
    model = Sequential()
    model.add(LSTM(100, activation='tanh', input_shape=inputShape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(classes, activation='softmax'))

    adam = optimizers.Adam(lr = 0.001)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

#--------------- DENSA ------------------#
X_train_dense = np.reshape(train_X,(train_X.shape[0],train_X.shape[1]*train_X.shape[2]))
X_test_dense = np.reshape(test_X,(test_X.shape[0],test_X.shape[1]*test_X.shape[2]))

y = train_y.reshape(-1,1)

model_densa = Densa(X_train_dense.shape[1],10)

model_densa.fit(X_train_dense,y,epochs=10)

#--------------- CNN --------------#
model_CNN = CNN(train_X.shape[1],train_X.shape[2],1,10)

model_CNN.fit(train_X,train_y,epochs=3)

#----------------- LSTM ------------#
model_LSTM = LSTM_model(train_X.shape[1],1,train_X.shape[2],10) 

model_LSTM.fit(train_X,train_y, epochs=5)
