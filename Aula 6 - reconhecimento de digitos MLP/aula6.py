import tensorflow as tf
import keras
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score,f1_score ,precision_score, recall_score

#Modelo para a rede neural
def create_model(camadas,dropout_rate,num_class):
    # Definindo estrutura da rede
    model = keras.Sequential([
        keras.layers.Dense(camadas[0],input_shape=(784,)),
        #keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(camadas[1],activation=("relu")),
        #keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(camadas[2],activation=("tanh")),
        #keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(camadas[3],activation=("relu")),
        #keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(camadas[4],activation=("relu")),
        #keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(num_class,activation='softmax')
    ])

    #Compilando modelo
    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

    return model

#Recebendo as imagens do dataset em matrizes
(train_x,train_y),(test_x,test_y) = mnist.load_data()

#Transformando matriz em um vetor para ser nossa entrada da rede neural (redimensionando)
#Para treino
new_train_x = np.reshape(train_x,(train_x.shape[0],train_x.shape[1]*train_x.shape[2]))
#Para teste
new_test_x = np.reshape(test_x,(test_x.shape[0],test_x.shape[1]*test_x.shape[2]))

#Binarizando vetor de saidas para treino e teste
binarizer = LabelBinarizer()
binarizer.fit(train_y)

bin_train_y = binarizer.transform(train_y)
bin_teste_y = binarizer.transform(test_y)

#treinamento da rede neural
callbacks = [tf.keras.callbacks.EarlyStopping(monitor="loss",patience=10)]
epochs = 25
model=create_model((128,256,1024,256,128),0.2,10)
H = model.fit(new_train_x,bin_train_y,batch_size=64,epochs=epochs, callbacks=callbacks)

#Resultados
resultados = model.predict(new_test_x)
y_pred=np.argmax(resultados,axis=1)

#sum(y_pred==test_y)#Quantidade de acertos nas entradas de teste

#Análise da rede
class_report = classification_report(y_pred,test_y)
conf_matrix = confusion_matrix(y_pred,test_y)
accuracy = accuracy_score(y_pred,test_y)
precision = precision_score(y_pred,test_y,average="macro")
f1score = f1_score(y_pred,test_y,average="macro")
recall = recall_score(y_pred,test_y,average="macro")

#Plotando gráfico Loss/Accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,epochs),H.history["loss"],label="train_loss")
plt.plot(np.arange(0,epochs),H.history["accuracy"],label="train_acc")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
