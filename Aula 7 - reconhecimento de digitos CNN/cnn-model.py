import tensorflow as tf
import keras
import numpy as np

from keras.datasets import mnist
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score,f1_score ,precision_score, recall_score

#Recebendo as imagens do dataset em matrizes
(train_x,train_y),(test_x,test_y) = mnist.load_data()

def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(6,(5,5), padding='same', activation="relu", input_shape=(28,28,1)),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(6,(5,5), activation="relu"),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(120,activation="relu"),
        keras.layers.Dense(64,activation="relu"),
        keras.layers.Dense(10,activation="softmax")
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    return model

callbacks = [tf.keras.callbacks.EarlyStopping(monitor="accuracy",patience=5)]

model = create_model()
X_train = train_x.reshape(train_x.shape[0],28,28,1)
X_test = test_x.reshape(test_x.shape[0],28,28,1)
model.fit(train_x,train_y,epochs=5,callbacks=callbacks)

resultados = model.predict(test_x)
y_pred=np.argmax(resultados,axis=1)
accuracy = accuracy_score(y_pred,test_y)
precision = precision_score(y_pred,test_y,average="macro")
recall = recall_score(y_pred,test_y,average="macro")

model.summary() #Como o modelo está feito

