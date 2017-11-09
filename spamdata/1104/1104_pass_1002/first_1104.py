#!/usr/bin/env python
# coding=utf8

import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.externals import joblib

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




def svmtest():
    # SVM 分类器
    from sklearn import svm
    # 建立SVM分类器
    clf = svm.SVC(gamma=0.001, C=100.)
    filldata = []
    filllabel = []


    def filldataFn(fname):

        f0 = open(fname)
        line = f0.readline()
        while line:
            label = line.split(' ')[0]
            arrstr = line.split(' ')[1]
            arrstr = arrstr.replace('L', '')
            arrstr = arrstr.replace('[', '')
            arrstr = arrstr.replace(']', '')
            arrstr = arrstr.replace(', ', ',')
            arrstr = arrstr.replace('\n', '')
            arrlist = [float(n) for n in arrstr.split(',')]

            # nparr = np.asarray(arrlist)

            # print label, nparr
            filldata.append(arrlist)
            filllabel.append(label)

            line = f0.readline()

    filldataFn('0.txt')
    filldataFn('1002.txt')
    npdata = np.array(filldata)
    nplabel = np.array(filllabel)
    # print type(npdata), type(npdata[0])
    # for idx, item in enumerate(filldata):
        # print item, filllabel[idx]


    clf.fit(npdata, nplabel)

    test = [1,0L,52424L,12325L,10540L,0,62964L,15,0] #0
    test = [1,2742L,143625L,19414L,6886L,0,150511L,17,1.7] #1002

    test = [1, 0L, 51056L, 10153L, 2116L, 0, 53172L, 14, 0] #0 other
    # print np.shape(np.array(test))

    # test predict
    count0 = 0
    count1002 = 0
    filldata = []
    filldataFn('other0.txt')
    testnpdata = np.array(filldata)
    for npdata in testnpdata:
        rs = clf.predict( npdata.reshape(1,9) )
        if rs[0] == '0':
            count0 = count0 + 1
    print 'test0', count0

    filldata = []
    filldataFn('other1002.txt')
    testnpdata = np.array(filldata)
    for npdata in testnpdata:
        rs = clf.predict( npdata.reshape(1,9) )
        if rs[0] == '1002':
            count1002 = count1002 + 1
    print 'test1002', count1002

def knntest():
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=3)
    filldata = []
    filllabel = []


    def filldataFn(fname):

        f0 = open(fname)
        line = f0.readline()
        while line:
            label = line.split(' ')[0]
            arrstr = line.split(' ')[1]
            arrstr = arrstr.replace('L', '')
            arrstr = arrstr.replace('[', '')
            arrstr = arrstr.replace(']', '')
            arrstr = arrstr.replace(', ', ',')
            arrstr = arrstr.replace('\n', '')
            arrlist = [float(n) for n in arrstr.split(',')]

            # nparr = np.asarray(arrlist)

            # print label, nparr
            filldata.append(arrlist)
            filllabel.append(label)

            line = f0.readline()

    filldataFn('0.txt')
    filldataFn('1002.txt')
    npdata = np.array(filldata)
    nplabel = np.array(filllabel)
    # print type(npdata), type(npdata[0])
    # for idx, item in enumerate(filldata):
        # print item, filllabel[idx]


    clf.fit(npdata, nplabel)

    test = [1,0L,52424L,12325L,10540L,0,62964L,15,0] #0
    test = [1,2742L,143625L,19414L,6886L,0,150511L,17,1.7] #1002

    test = [1, 0L, 51056L, 10153L, 2116L, 0, 53172L, 14, 0] #0 other
    # print np.shape(np.array(test))

    # test predict
    count0 = 0
    count1002 = 0
    filldata = []
    filldataFn('other0.txt')
    testnpdata = np.array(filldata)
    for npdata in testnpdata:
        rs = clf.predict( npdata.reshape(1,9) )
        if rs[0] == '0':
            count0 = count0 + 1
    print 'test0', count0

    filldata = []
    filldataFn('other1002.txt')
    testnpdata = np.array(filldata)
    for npdata in testnpdata:
        rs = clf.predict( npdata.reshape(1,9) )
        if rs[0] == '1002':
            count1002 = count1002 + 1
    print 'test1002', count1002




svmtest()
# knntest()

