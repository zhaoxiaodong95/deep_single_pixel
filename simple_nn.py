import keras
from keras.models import Sequential
from keras.layers import Conv2D, Reshape, MaxPooling2D, Dense, Activation, Dropout, GaussianDropout, ActivityRegularization, Flatten
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization
from keras import initializers
from keras import regularizers
from keras.activations import softmax, relu, elu
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import load_model

import sys
import random 
from sys import exit
import numpy as np
np.random.seed(1234)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


def nn_model(input_shape,output_shape):

    padding = 'same'

    """
    may need to play around with this in order to get
    correct output shape
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu', padding=padding,
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))
    #model.add(Reshape((output_shape[0],output_shape[1],32)))

    model.add(Conv2D(64, (5, 5), activation='relu',
                     padding=padding))
    model.add(MaxPooling2D(pool_size=(8, 8), strides=(8, 8)))

    model.add(Conv2D(1, (5, 5), activation='relu',
                     padding=padding))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    # add this if doing classification
    #model.add(Flatten())
    #model.add(Dense(1000, activation='relu'))
    #model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    return model

class AccuracyHistory(keras.callbacks.Callback):
    """
    Get accurate history of model after training
    """
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

def run_cnn(model, x_train, y_train, x_val, y_val, x_test, y_test):
    # compile model
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.SGD(lr=0.01),
                  metrics=['accuracy'])

    # print summary of network archetecture
    model.summary()

    history = AccuracyHistory()

    # fit model
    hist = model.fit(x_train, y_train,
              batch_size=16,
              epochs=10,
              verbose=1,
              validation_data=(x_val, y_val),
              callbacks=[history])

    # evaluate results
    eval_results = model.evaluate(x_test, y_test,
                                  sample_weight=None,
                                  batch_size=16, verbose=1)

    # predict output
    preds = model.predict(x_test)   
 
    return model, hist, eval_results, preds

# make random gaussian blob for testing purposes
def gauss_2d():
    x, y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
    d = np.sqrt(x*x+y*y)
    sigma, mu = 1.0, 0.0
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    print("Made 2D Gaussian-like array")
    #print(g)

    return g

def main():
    # define parameters
    input_shape = [1024,512,1]
    output_shape = [64,64]
   
    #y_train = []
    #for i in range(num_train):
    #    y_train.append(gauss_2d)


    # load data
    

    # define model
    model = nn_model(input_shape, output_shape)
    exit()

    # train model
    model, hist, eval_results, preds = run_cnn(model, x_train, y_train, x_val, y_val, x_test, y_test)

main()
