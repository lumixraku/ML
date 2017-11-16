#!/usr/bin/env python
# coding=utf8



# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_yaml
import matplotlib.pyplot as plt
import numpy
import os
# fix random seed for reproducibility
numpy.random.seed(7)

dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
model = None
if os.path.exists('./pima_model.yaml'):
    yaml_file = open('pima_model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()

    model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    model.load_weights("model.h5")
else:
    # load pima indians dataset
    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(X, Y, epochs=1500, batch_size=10)
    # evaluate the model
    scores = model.evaluate(X, Y)
    # print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open("pima_model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


# x_test = numpy.asarray([7,100,0,0,0,30.0,0.484,32])
# x_test = numpy.asarray([7,107,74,0,0,29.6,0.254,31])
x_test = dataset[:,0:8]
y_test = dataset[:,8]
rowidx = 0
matchcount = 0

y_pred = model.predict_classes(x_test, batch_size=1)

print(len(x_test), len(y_test))
# plt.scatter(x_test, y_test)
# plt.plot(x_test, y_pred)
# plt.show()
# print(y_pred)
# for row in x_test:
#     rs = [[0]]
#     rs = model.predict_classes( row.reshape(1,8), batch_size=1)
#     y_test_label = y_test[rowidx]
#     if int(y_test_label) == rs[0][0]:
#         matchcount = matchcount +1
#     print(rowidx, rs[0][0], y_test_label,int(y_test_label) == rs[0][0])
#     rowidx = rowidx + 1

# print matchcount
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))