# -*- coding: utf-8 -*-

# http://blog.csdn.net/hujingshuang/article/details/61917359


import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
from PIL import Image
import struct
from keras.datasets import mnist as kerasmnist
mnist = input_data.read_data_sets('mnist_data/', one_hot=True)

(x_train, y_train), (x_test, y_test) = kerasmnist.load_data()   #x是数据 28*28 的图像构成的集合  y 是 label
x_train = x_train[0:1000]
y_train = y_train[0:1000]

# y_actual = W * x + b
x = tf.placeholder(tf.float32, [None, 784])                 # 占位符
y_actual = tf.placeholder(tf.float32, shape=[None, 10])     # 占位符（实际值）
W = tf.Variable(tf.zeros([784, 10]))                        # 初始化权值W
b = tf.Variable(tf.zeros([10]))                             # 初始化偏置b

# 建立抽象模型
y_predict = tf.nn.softmax(tf.matmul(x, W) + b)              # 加权变换并进行softmax回归，得到预测值
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual * tf.log(y_predict), reduction_indices=1))   # 求交叉熵
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)     # 用梯度下降法使得残差最小

y_predictmax = tf.argmax(y_predict, 1)
y_actualmax = tf.argmax(y_actual, 1)
# 建立测试训练模型
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1))  # 若预测值与实际值相等则返回boolen值1，不等为0
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))                 # 将返回的beelen数组转换为float类型，并求均值，即得到准确度

# 初始化所有变量
init = tf.initialize_all_variables()


def printImg(data):
    """
    data shape 为(784, ) 也就是一个1*784列
    """
    tmp = np.reshape(data, (28,28))
    tmp = mnist.test.images[0]
    tmp = [int(round(x)) for x in tmp]
    print np.reshape(tmp, (28, 28))
    # return tmp

def showImg(data):


# https: // stackoverflow.com / questions / 37558523 / converting - 2d - numpy - array - of - grayscale - values - to - a - pil - image
#  The Image.fromarray(____, 'L') function seems to only work properly with an array of integers between 0 and 255. I use the np.uint8 function for this.
    columns = 28
    rows = 28
    data = np.reshape(data, (columns, rows))
    # print np.shape(data), data
    mat = np.uint8(data * 255)
    new_im = Image.fromarray(mat, 'L')
    new_im.show()


# 在一切操作之后，都用sess来run它们
with tf.Session() as sess:
    sess.run(init)

    # mnist.test.images 测试数据10000条  每条都是 784个元素 的数组
    # print len(mnist.test.images), np.shape(mnist.test.images[0])
    # 注意 label 的表示形式 array([0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.] 表示数字7  是因为在7这个位置的概率是1
    # 之所以这么表示 可能是因为 softmax 吧 （毕竟这个的输出结果 所有概率和为1）
    print(printImg(mnist.test.images[0]), mnist.test.labels[0], '===\n')
    # showImg(mnist.test.images[0])



    for i in range(1000):       # 训练阶段，迭代1000次
        batch_xs, batch_ys = mnist.train.next_batch(100)        # 按批次训练，每批100行数据

        # 执行训练（此处为占位符x, y_actual载入数据，然后使用配置好的train来训练）
        sess.run(train, feed_dict={x: batch_xs, y_actual: batch_ys})

        if i % 500 == 0:        # 每训练500次，测试一次
            # print("correct_prediction", sess.run(correct_prediction, feed_dict={x: mnist.test.images, y_actual: mnist.test.labels}))

            # mnist.test数据共有10000个 labels 也是10000个
            predictrs = sess.run(y_predictmax, feed_dict={x: mnist.test.images, y_actual: mnist.test.labels})

            # print mnist.test.labels[0:100], '>--->', predictrs[0:100]

            actualrs = sess.run(y_actualmax, feed_dict={x: mnist.test.images, y_actual: mnist.test.labels})
            # print actualrs[0:100]
            print("accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y_actual: mnist.test.labels}))
