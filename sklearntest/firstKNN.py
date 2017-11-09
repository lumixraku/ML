#!/usr/bin/env python
# coding=utf8
import constants
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
digits = datasets.load_digits()
# print( type (digits.data[0][0])  )
# print(digits.data)
# print(digits.target)


def showresult(data):
    if type(data) == list:
        data = np.array(data)
    data = data.reshape(8, 8)
    plt.imshow(data, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()

# digits.images 是一个3维数组， 每一个元素是一个二维数组(8*8)表示数字图形 （灰阶）
# 0是颜色最浅方块 数值越大颜色越深
# digits.images[0] 就是数字0


def draw0123():
    print digits.images[0]

    images_and_labels = list(zip(digits.images, digits.target))
    for index, (image, label) in enumerate(images_and_labels[:4]):
        plt.subplot(2, 4, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Training: %i' % label)

    plt.show()


def savemodel(clf):
    """
    模型持久化
    """
    # s = pickle.dumps(clf)
    # clf2 = pickle.loads(s)
    # rs = clf2.predict(constants.my82D)
    # print "clf2 rs:", rs
    joblib.dump(clf, 'number.pkl')




def knntest():
    # clf classifier
    clf = KNeighborsClassifier(n_neighbors=3)
    # 使用训练数据对分类器进行训练，它将会返回分类器的某些参数设置

    # digits 是一堆二位数组，每个元素是表示数字的 64 长度的数组 （images 的一维表示） [numpy]
    # print type(digits.data[:-1]), type(digits.data[:-1][0])

    # print digits.data[:-1][0]
    clf.fit(digits.data[:-1], digits.target[:-1])  # model.fit(): 实际上就是训练，对于监督模型来说是 fit(X, y)，对于非监督模型是 fit(X)。
    # print( len (digits.data[:-1][0])  )
    # print clf

    # 用于计算的部分代码已被隐藏，以下是用于预测的未知数据
    # 你可以改变这个数据中的数字，但必须保证数组元素个数为64，否则将会出错
    test = [0, 0, 10, 14, 8, 1, 0, 0,
            0, 2, 16, 14, 6, 1, 0, 0,
            0, 0, 15, 15, 8, 15, 0, 0,
            0, 0, 5, 16, 16, 10, 0, 0,
            0, 0, 12, 15, 15, 12, 0, 0,
            0, 4, 16, 6, 4, 16, 6, 0,
            0, 8, 16, 10, 8, 16, 8, 0,
            0, 1, 8, 12, 14, 12, 1, 0]

    # my 8
    test =  constants.my8

    print("对图片的预测结果为：")
    print(clf.predict( constants.my82D ))
    # print(clf.predict(np.asarray(test).reshape(1, 64)))

    # 下面这个函数将自动完成绘图任务
    # showresult(test)

    savemodel(clf)


def loadFileModel():
    clf2 = joblib.load('number.pkl')
    rs = clf2.predict(constants.my82D)
    print "clf2 rs:", rs


knntest()
loadFileModel()
