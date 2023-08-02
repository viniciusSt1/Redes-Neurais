import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score,f1_score ,precision_score, recall_score

def CNN(Xtrain):
    input_ = keras.layers.Input(shape=Xtrain.shape[1:])

    cnn_cv1 = keras.layers.Conv2D(filters=6,kernel_size=(5,5),activation='relu',padding='same')(input_)
    cnn_maxp = keras.layers.MaxPooling2D(pool_size=2)(cnn_cv1)

    cnn_cv2 = keras.layers.Conv2D(filters=6,kernel_size=(5,5),activation='relu',padding='same')(cnn_maxp)
    cnn_maxp2 = keras.layers.MaxPooling2D(pool_size=2)(cnn_cv2)

    cnn_fla = keras.layers.Flatten()(cnn_maxp2)
    cnn_d1 = keras.layers.Dense(units=120,activation='relu')(cnn_fla)
    cnn_d2 = keras.layers.Dense(units=64,activation='relu')(cnn_d1)
    output_ = keras.layers.Dense(units=10,activation='softmax')(cnn_d2)

    model = Model(inputs=[input_],outputs=[output_])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    return model

model = CNN(X_train)
H = model.fit(X_train,train_y,epochs=5)

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred,axis=1)

print(classification_report(test_y, y_pred))
print(confusion_matrix(test_y,y_pred))

model.summary()
