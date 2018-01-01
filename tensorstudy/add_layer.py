# -*- coding: utf-8 -*-
#

# https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/3-2-create-NN/

import tensorflow as tf
import numpy as np

import tensorflow.examples.tutorials.mnist.input_data as input_data


def main():

    mnist = input_data.read_data_sets('mnist_data/', one_hot=True)
    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])  # 表示 y_actual

    # 定义隐藏层
    l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

    # 定义输出层
    # l2 = add_layer(l1, 10, 10, activation_function=tf.nn.relu)

    prediction = add_layer(l1, 10, 1, activation_function=None)

    # 计算预测值prediction和真实值的误差，对二者差的平方求和再取平均。
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
    # loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=1))

    # loss = tf.reduce_mean(tf.reduce_sum(tf.abs(ys - prediction),
                                        # reduction_indices=1))


    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # correct_prediction = tf.equal(
    #     tf.argmax(prediction, 1), tf.argmax(ys, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    init = tf.global_variables_initializer()

    # start----------------
    sess = tf.Session()
    sess.run(init)

    # np.linspace(-1000, 1000, 3000, dtype=np.float32) 生成 -1000 到 1000的 3000个随机数
    x_data=np.linspace(-1, 1, 300,
                       dtype=np.float32)[:, np.newaxis]

    noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
    y_data = np.square(x_data) - 0.5 + noise
    for i in range(1000):       # 训练阶段，迭代1000次

        # (注意：当运算要用到placeholder时，就需要feed_dict这个字典来指定输入。)
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

        # 每训练n次，测试一次
        n = 50
        if i % n == 0:
            print("loss:", sess.run(loss, feed_dict={
                  xs: x_data, ys: y_data}))


def add_layer(inputs, in_size, out_size, activation_function=None):
    """
    在没有激励函数的情况下 addLayer返回 Wx + b 有激励函数 返回 fn(Wx+b)
    定义添加神经层的函数def add_layer(), 它有四个参数：输入值、输入的大小、输出的大小和激励函数，我们设定默认的激励函数是None。
    """
    # 因为在生成初始参数时，随机变量(normal distribution)会比全部为0要好很多，所以我们这里的weights为一个in_size行, out_size列的随机变量矩阵。
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))

    # 在机器学习中，biases的推荐值不为0，所以我们这里是在0向量的基础上又加了0.1。
    biases = tf.Variable(tf.zeros([out_size]) + 0.1)
    # 我们定义Wx_plus_b, 即神经网络未激活的值。其中，tf.matmul()是矩阵的乘法。
    Wx_plus_b = tf.matmul(inputs, Weights) + biases  # y_predict？

    # 当activation_function——激励函数为None时，输出就是当前的预测值——Wx_plus_b，不为None时，就把Wx_plus_b传到activation_function()函数中得到输出。
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs


if __name__ == "__main__":
    main()
