from __future__ import print_function
import pandas as pd
import random
import numpy as np

batch_size = 100
num_classes = 10
epochs = 50

# input image dimensions

# # the data, shuffled and split between train and test sets
# mndata = MNIST('data/MNIST')

# x_train_images, y_train_labels = mndata.load_training()
# x_test_images, y_test_labels = mndata.load_testing()

# x_train = x_train_images
# y_train = y_train_labels
# x_test = x_test_images
# y_test = y_test_labels


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, AveragePooling2D, Activation
from keras import backend as K
from keras.datasets import mnist
 
# Load pre-shuffled MNIST data into train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(96, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(96, (3, 3), activation='relu'))
model.add(Conv2D(96, (3, 3), activation='relu', strides=2))
model.add(Conv2D(192, (3, 3), activation='relu'))
model.add(Conv2D(192, (3, 3), activation='relu'))
model.add(Conv2D(192, (3, 3), activation='relu', strides=2))
model.add(Conv2D(192, (3, 3), activation='relu', padding='valid'))
model.add(Conv2D(192, (1, 1), activation='relu'))
model.add(Conv2D(10, (1, 1), activation='relu'))
model.add(AveragePooling2D(pool_size=(1, 1), strides=None, padding='valid', data_format=None))
model.add(Activation('softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=1e-5,decay =0.000001),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
