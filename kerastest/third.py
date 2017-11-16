#!/usr/bin/env python
# coding=utf8
# http://www.lining0806.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8Bkeras%E5%85%A5%E9%97%A8/

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

# Generate dummy data, with 2 classes (binary classification)
import numpy as np

x_train = np.random.random((1000, 20)) #生成随机数 1000个 1*20维向量
# print len(x_train), len(x_train[0])
# print np.shape(x_train[0]) #(20,)

y_train = np.random.randint(2, size=(1000, 1))  #生成【0，2）范围内的随机数 1000个 维度是1
# print np.shape(y_train[0]) # (1,) 表示1维

x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# Train the model, iterating on the data in batches of 128 samples
model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)

# score = model.evaluate(x_test, y_test, batch_size=128)
# print x_test[0]
# print x_test[0].reshape(4,5)  # reshape 是将一组数据的维度变换 例如 1*24的数据 可以转变为 3*8  4*6 等
classes = model.predict( x_test[0].reshape(1,20), batch_size=1)
print classes