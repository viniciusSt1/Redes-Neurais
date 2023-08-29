# example of training a gan on mnist
from numpy import expand_dims, zeros, ones, vstack
from numpy.random import randn, randint
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, \
  LeakyReLU, Dropout, MaxPooling2D, UpSampling2D
from matplotlib import pyplot
import sys

def define_discriminator(in_shape=(28,28,1)):
  model = Sequential()
  model.add(Conv2D(16, (5,5), activation='relu', padding='same', input_shape = in_shape))
  model.add(MaxPooling2D((2,2)))
  model.add(Conv2D(16, (5,5), activation='relu', padding='same'))
  model.add(MaxPooling2D((2,2)))
  model.add(Flatten())
  model.add(Dense(32, activation='relu'))
  model.add(Dense(32, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  
  model.compile(optimizer=Adam(learning_rate = 0.001), loss='binary_crossentropy', metrics=['accuracy'])
  
  return model

def define_generator(latent_dim):
  model = Sequential()
  
  model.add(Dense(128*7*7, activation='relu', input_dim=latent_dim))
  model.add(Reshape((7,7,128)))  
  model.add(UpSampling2D(interpolation='bilinear')) #7*2 = 14x14
  model.add(Conv2D(32,(5, 5), padding='same', activation='relu'))
  model.add(UpSampling2D(interpolation='bilinear')) #14*2 = 28x28
  model.add(Conv2D(32,(5, 5), padding='same', activation='relu'))
  model.add(Conv2D(1, (5, 5), padding='same', activation='tanh'))
  
  model.compile(optimizer=Adam(learning_rate = 0.001), loss='binary_crossentropy', metrics=['accuracy'])
  
  return model

def define_gan(g_model, d_model):
  # make weights in the discriminator not trainable
  d_model.trainable = False
  model = Sequential()
  model.add(g_model)
  model.add(d_model)
  # Muda os nomes dos layers
  model.layers[0]._name="gerador"
  model.layers[1]._name="discriminador"
  opt = Adam(learning_rate=0.0002, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt)
  return model

def load_real_samples():
  (trainX, trainy), (_, _) = load_data()
  trainX = trainX[(trainy==7)]
  del trainy
  X = expand_dims(trainX, axis=-1)
  X = X.astype('float32')
  X = X / 255.0
  return X

def generate_real_samples(dataset, n_samples):
  ix = randint(0, dataset.shape[0], n_samples)
  X = dataset[ix]
  y = ones((n_samples, 1))
  return X, y

def generate_latent_points(latent_dim, n_samples):
  x_input = randn(n_samples, latent_dim)
  return x_input

def generate_fake_samples(g_model, latent_dim, n_samples):
  x_input = generate_latent_points(latent_dim, n_samples)
  X = g_model.predict(x_input)
  y = zeros((n_samples, 1))
  return X, y

def save_plot(examples, epoch, n=5):
  for i in range(n * n):
    pyplot.subplot(n, n, 1 + i)
    pyplot.axis('off')
    pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
  filename = 'generated_plot_e%03d.png' % (epoch+1)
  pyplot.savefig(filename)
  pyplot.close()

def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
  X_real, y_real = generate_real_samples(dataset, n_samples)
  _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
  x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
  _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
  print('Acuracia do discriminador em amostras reais: %.0f%%, falsas: %.0f%%' % (acc_real*100, acc_fake*100))
  save_plot(x_fake, epoch)

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=100):
  bat_per_epo = int(dataset.shape[0] / n_batch)
  half_batch = int(n_batch / 2)
  # manually enumerate epochs
  for i in range(n_epochs): #i==epoch
    # enumerate batches over the training set
    for j in range(bat_per_epo):
      X_real, y_real = generate_real_samples(dataset, half_batch)
      X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
      X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
      d_loss, d_accuracy = d_model.train_on_batch(X, y)
      X_gan = generate_latent_points(latent_dim, n_batch)
      # create inverted labels for the fake samples
      y_gan = ones((n_batch, 1))
      g_loss = gan_model.train_on_batch(X_gan, y_gan)
      if (j+1==bat_per_epo):
        print('epoch=%d, batch=%d/%d, d_acc=%.3f, g_loss=%.3f (bin_crossentr)' 
          % (i+1, j+1, bat_per_epo, d_accuracy, g_loss))
    if (i+1) % 10 == 0:
      summarize_performance(i, g_model, d_model, dataset, latent_dim)
    if i+1==n_epochs:
      filename = 'generator_model_%03d.h5' % (i + 1)
      g_model.save(filename)

# size of the latent space
latent_dim = 10
d_model = define_discriminator()
g_model = define_generator(latent_dim)
gan_model = define_gan(g_model, d_model)
dataset = load_real_samples()
save_plot(dataset,-1)
train(g_model, d_model, gan_model, dataset, latent_dim)

