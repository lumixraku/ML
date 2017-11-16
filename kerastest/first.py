#!/usr/bin/env python
# coding=utf8

from PIL import Image
from numpy import *
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()


model.add(Dense(units=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(units=10))
model.add(Activation("softmax"))

# 编译模型时必须指明损失函数和优化器
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])




#暂时使用 sklearn 的数据
from sklearn import datasets
from sklearn.externals import joblib

digits = datasets.load_digits()
x_train = digits.data[:-1]
y_train = digits.target[:-1]


# train data
print x_train, x_train[0], len(x_train[0])
model.fit(x_train, y_train, epochs=5, batch_size=32)


# 评估
x_test = x_train
y_test = y_train
# loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

#预测
my8 = [0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 10, 10, 1, 0, 0,
       0, 0, 10, 0, 0, 10, 0, 0,
       0, 0, 3, 10, 10, 3, 0, 0,
       0, 0, 3, 10, 10, 3, 0, 0,
       0, 0, 10, 0, 0, 10, 0, 0,
       0, 0, 1, 10, 10, 1, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0]
my82D = np.asarray(my8).reshape(1, 64)

classes = model.predict(my82D, batch_size=128)