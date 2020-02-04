# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 17:43:20 2019

@author: Przemek
"""

import keras
from keras.models import load_model
from matplotlib import pyplot as plt
import snap7
import snap7.client as c
from snap7.util import *
from time import sleep


#nawiazanie polaczenia ze sterownikiem
myplc = snap7.client.Client()
myplc.connect('192.168.0.1',0,1)
myplc.get_connected() 

#wczytanie zapisanego modelu
model = load_model('CNN_model.h5') 

#sprawdzenie cyfry image_index i predykcji modelu pred.argmax()
image_index = 500
fig = plt.figure()
plt.imshow(x_test[image_index].reshape(28, 28),cmap='gray', interpolation='none')
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())
fig

#zmiana stanu wyjć cyfrowych w zaleznosci od rozpoznanego obrazu
if pred.argmax() == 0:
    print("Zero")
    print(pred.argmax())
    WriteOutput(myplc,0,0,True)
    sleep(3)
    WriteOutput(myplc,0,0,False)
    
elif pred.argmax() == 1:
    print("Jedynka")
    print(pred.argmax())
    WriteOutput(myplc,0,1,True)
    sleep(3)
    WriteOutput(myplc,0,1,False)
    
elif pred.argmax() == 2:
    print("Dwójka")
    print(pred.argmax())
    WriteOutput(myplc,0,2,True)
    sleep(3)
    WriteOutput(myplc,0,2,False)
    
elif pred.argmax() == 3:
    print("Trójka")
    print(pred.argmax())
    WriteOutput(myplc,0,3,True)
    sleep(3)
    WriteOutput(myplc,0,3,False)
    
elif pred.argmax() == 4:
    print("Czwórka")
    print(pred.argmax())
    WriteOutput(myplc,0,4,True)
    sleep(3)
    WriteOutput(myplc,0,4,False)
    
elif pred.argmax() == 5:
    print("Piątka")
    print(pred.argmax())
    WriteOutput(myplc,0,5,True)
    sleep(3)
    WriteOutput(myplc,0,5,False)
     
elif pred.argmax() == 6:
    print("Szóstka")
    print(pred.argmax())
    WriteOutput(myplc,0,6,True)
    sleep(3)
    WriteOutput(myplc,0,6,False)
    
elif pred.argmax() == 7:
    print("Siódemka")
    print(pred.argmax())
    WriteOutput(myplc,0,7,True)
    sleep(3)
    WriteOutput(myplc,0,7,False)
    
elif pred.argmax() == 8:
    print("Ósemka")
    print(pred.argmax())
    WriteOutput(myplc,1,0,True)
    sleep(3)
    WriteOutput(myplc,1,0,False)
    
elif pred.argmax() == 9:
    print("Dziewiątka")
    print(pred.argmax())
    WriteOutput(myplc,1,1,True)
    sleep(3)
    WriteOutput(myplc,1,1,False)
    
   
    