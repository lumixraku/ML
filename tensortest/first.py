#!/usr/bin/env python
# coding=utf8

from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base
import glob2
import tensorflow as tf
from numpy import *


def GetImage(filelist):
    width=28
    height=28
    value=zeros([1,width,height,1])
    value[0,0,0,0]=-1
    label=zeros([1,10]) #一个 ndarray  长度为1 元素是一个10维数组

    # print len(label), shape(label[0])
    label[0,0]=-1

    for filename in filelist:
        img=array(Image.open(filename).convert("L"))
        width,height=shape(img);
        index=0
        tmp_value=zeros([1,width,height,1])
        for i in range(width):
            for j in range(height):
                tmp_value[0,i,j,0]=img[i,j]
                index+=1

        if(value[0,0,0,0]==-1):
            value=tmp_value
        else:
            value=concatenate((value,tmp_value))

        tmp_label=zeros([1,10])
        index=int(filename.strip().split('/')[2][0])
        # print "input:",index
        tmp_label[0,index]=1
        if(label[0,0]==-1):
            label=tmp_label
        else:
            label=concatenate((label,tmp_label))

    return array(value),array(label)






sess = tf.Session()


# One type of the Tensor nodes is Constant
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(sess.run([node1, node2])) # [3.0, 4.0]
# One type of the Tensor nodes is Variable
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run([W, b]))
# [array([ 0.30000001], dtype=float32), array([-0.30000001], dtype=float32)]


# https://juejin.im/post/58b424e2570c350069343ebb
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Create the model
# 判断数字
# tf.zeros 表示初始化为 0。
x = tf.placeholder(tf.float32, [None, 784])  #28*28  图片
W = tf.Variable(tf.zeros([784, 10]))  #权重  因为要判断10个数字
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

#  Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])  # 这是 label
# print x

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


# Train
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images,y_: mnist.test.labels})) #accuracy 计算的是一个整体的精确度。





#//////////////////////////////////////////////////////////////////////
print("Start Test Images")

dir_name = "./test_num"
files = glob2.glob(dir_name + "/*.png")
cnt = len(files)
for i in range(cnt):
    # print(files[i])
    test_img, test_label = GetImage([files[i]])
    testDataSet = DataSet(test_img, test_label, dtype=tf.float32)

    res = accuracy.eval({x: testDataSet.images, y_: testDataSet.labels})
    print len(test_img[0]), test_label

    print("output: ",  res)
    print("----------")
