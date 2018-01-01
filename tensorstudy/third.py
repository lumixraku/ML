# -*- coding: utf-8 -*-

# http://blog.csdn.net/hujingshuang/article/details/61917359


import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np

# 尝试下载数据包
mnist = input_data.read_data_sets('mnist_data/', one_hot=True)

# y_actual = W * x + b
x = tf.placeholder(tf.float32, [None, 784])                 # 占位符
y_actual = tf.placeholder(tf.float32, shape=[None, 10])     # 占位符（实际值）
W = tf.Variable(tf.zeros([784, 10]))                        # 初始化权值W
b = tf.Variable(tf.zeros([10]))                             # 初始化偏置b

# 建立抽象模型
# 构建Softmax 回归模型
y_predict = tf.nn.softmax(tf.matmul(x, W) + b)              # 加权变换并进行softmax回归，得到预测值


# 指定最小化误差用的损失函数，我们的损失函数是目标类别和预测类别之间的交叉熵。
# 求交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual * tf.log(y_predict), reduction_indices=1))

# 我们用最速下降法让交叉熵下降，步长为0.5.
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)     # 用梯度下降法使得残差最小

y_predictmax = tf.argmax(y_predict, 1)
y_actualmax = tf.argmax(y_actual, 1)
# 建立测试训练模型
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1))  # 若预测值与实际值相等则返回boolen值1，不等为0
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))                 # 将返回的beelen数组转换为float类型，并求均值，即得到准确度

# 初始化所有变量
init = tf.initialize_all_variables()

# 在一切操作之后，都用sess来run它们
with tf.Session() as sess:
    sess.run(init)

    # mnist.test.images 测试数据10000条  每条都是 784个元素 的数组
    # print len(mnist.test.images), np.shape(mnist.test.images[0])


    for i in range(1000):       # 训练阶段，迭代1000次
        batch_xs, batch_ys = mnist.train.next_batch(100)        # 按批次训练，每批100行数据

        # print np.shape(batch_xs[0]), batch_xs[0]
        # print np.shape(batch_ys[0]), batch_ys[0]
        # print '............'
        # 执行训练（此处为占位符x, y_actual载入数据，然后使用配置好的train来训练）
        sess.run(train, feed_dict={x: batch_xs, y_actual: batch_ys})

        if i % 500 == 0:        # 每训练500次，测试一次
            # print("correct_prediction", sess.run(correct_prediction, feed_dict={x: mnist.test.images, y_actual: mnist.test.labels}))
            predictrs = sess.run(y_predictmax, feed_dict={x: mnist.test.images, y_actual: mnist.test.labels})
            print predictrs[0:100]

            actualrs = sess.run(y_actualmax, feed_dict={x: mnist.test.images, y_actual: mnist.test.labels})
            # print actualrs[0:100]
            # print("accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y_actual: mnist.test.labels}))