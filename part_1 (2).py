from tensorflow import keras

(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data(path="mnist.npz")

print(x_test.shape)
print(x_train.shape)
print(y_test.shape)
print(y_train.shape)

#!pip install opencv-contrib-python-headless==4.1.2.30 --quiet
#!pip install cvlib --quiet

#!pip install tensorflow-cpu

import matplotlib.pyplot as plt
plt.imshow(x_train[20],cmap='gray')

x_train=keras.utils.normalize(x_train)
import matplotlib.pyplot as plt
plt.imshow(x_train[20],cmap='gray')

x_train[0]

#from keras.utils import to_categorical()
#to_categorical(y_train[0])
import tensorflow as tf
tf.keras.utils.to_categorical(y_train[0])

model=keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(500,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))

model.compile(loss="sparse_categorical_crossentropy",optimizer='adam',metrics=['accuracy'])

history=model.fit(x_train,y_train,epochs=5)

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])

model.fit(x_train,y_train,epochs=20,validation_data=(x_train,y_train))

import pandas as pd
loss=pd.DataFrame(model.history.history)
loss.plot()

import numpy as np
y_pred=model.predict(x_test)  # classification gives output in form of probability
y_pred=np.argmax(y_pred,axis=1) # 
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_pred,y_test)

accuracy_score(y_pred,y_test)

model.save('DIGIT.hdf5')

