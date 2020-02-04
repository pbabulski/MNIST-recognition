# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 20:38:11 2019

@author: Przemek
"""

import tensorflow as tf
import keras
from keras.datasets import mnist
from matplotlib import pyplot as plt
from random import randint
import numpy

#importowanie zbioru 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#wyswietla liczebnosc zbioru x i rozmiar zdjec
x_train.shape
print(x_train.shape)

#zmiana ksztaltu tablicy do 4-wymiarowej zeby mogla zeby mogla pracowac z KERAS
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

#upewniamy sie, ze wartosc jest zmiennoprzecinkowa aby mozna bylo uzyskac warto∟ci dziesietne po podziale
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#normalizowanie wartosci RGB poprzez podzielenie przez maksymalna wartosc
x_train /= 255
x_test /= 255

#wyswietlanie liczby x_train i x_test
print('x_train shape: ', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

#importowanie modulow z KERAS zawierajacych modele i warstwy
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
#tworzenie sekwencyjnego modelu i dodawanie warstw
model = Sequential()
model.add(Conv2D(32,kernel_size=(5,5),input_shape=input_shape,activation=tf.nn.relu)) 
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Flatten()) #splaszczenie dwuwymiarowych tablic dla w pelni polaczonych warstw. 
                     #Stosowany w celu zachowania wag przełączając model na inny format danych
model.add(Dense(64,activation=tf.nn.sigmoid))
model.add(Dense(64,activation=tf.nn.sigmoid))
model.add(Dropout(0.25))
model.add(Dense(10,activation=tf.nn.softmax))

#kompilowanie i dopasowywanie modelu 
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#trening algorytmu
history = model.fit(x_train,y_train, validation_split=0.33, epochs=45)

#ocena modelu
model.evaluate(x_test, y_test)

#wywietlenie wykresów porównujących wartoci celnosci i bledy w treningu i 
#walidacji
plt.subplot(2,1,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('trafnosc modelu')
plt.ylabel('trafnosc')
plt.xlabel('iteracje')
plt.legend(['trening','test'],loc='upper left')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('blad modelu')
plt.ylabel('blad')
plt.xlabel('iteracje')
plt.legend(['trening', 'test'], loc='upper left')
plt.show()
