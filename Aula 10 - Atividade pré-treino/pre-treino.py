import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def get_base_model(neurons, trainX, trainY):
	model = Sequential()
	model.add(Dense(neurons, input_dim=784, activation='tanh', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))

	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(trainX, trainY, epochs=50, verbose=1)
	return model

# evaluate a fit model
def evaluate_model(model, trainX, testX, trainy, testy):
	_, train_acc = model.evaluate(trainX, trainy, verbose=0)
	_, test_acc = model.evaluate(testX, testy, verbose=0)
	return train_acc, test_acc

# add one new layer and re-train only the new layer
def add_layer(neurons, model, trainX, trainy):
	output_layer = model.layers[-1]
	model.pop()
	for layer in model.layers:
		layer.trainable = False
	model.add(Dense(neurons, activation='tanh', kernel_initializer='he_uniform'))
	model.add(output_layer)
	model.fit(trainX, trainy, epochs=100, verbose=1)

def best_score(scores):
    output_array_scores = np.array(list(scores.values()))
    max_score = 0
    for i in range(len(output_array_scores)):
        if (output_array_scores[i][1] > max_score) and (output_array_scores[i+1][1] < output_array_scores[i][1]):
            max_score = output_array_scores[i][1]
            return max_score, i+2

if __name__ == '__main__':
    (trainX, trainY), (testX, testY) = mnist.load_data()
    
    trainX = np.reshape(trainX, (trainX.shape[0],trainX.shape[1]*trainX.shape[2]))
    testX = np.reshape(testX, (testX.shape[0],testX.shape[1]*testX.shape[2]))
    
    #trainY = to_categorical(trainY)
    #testY = to_categorical(testY)

    neurons = 16
    model = get_base_model(neurons, trainX, trainY)
    scores = dict() #avalie o modelo base
    train_acc, test_acc = evaluate_model(model, trainX, testX, trainY, testY)
    print('> layers=%d, train=%.3f, test=%.3f' % (len(model.layers), train_acc, test_acc))
    scores[len(model.layers)] = (train_acc, test_acc)
   
    n_layers = 2 # quantidade maxima de camadas adicionadas
    
    for i in range(n_layers):
     	add_layer(neurons,model, trainX, trainY)    #avalie o modelo
     	train_acc, test_acc = evaluate_model(model, trainX, testX, trainY, testY)
     	print('> layers=%d, train=%.3f, test=%.3f' % (len(model.layers), train_acc, test_acc))
     	# store scores for plotting
     	scores[len(model.layers)] = (train_acc, test_acc)
    
    saida = np.array(list(scores.values()))
    # plot number of added layers vs accuracy
    plt.plot(list(scores.keys()), [scores[k][0] for k in scores.keys()], label='train', marker='.')
    plt.plot(list(scores.keys()), [scores[k][1] for k in scores.keys()], label='test', marker='.')
    plt.legend()
    plt.show()