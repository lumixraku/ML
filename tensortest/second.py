#!/usr/bin/env python
# coding=utf8

# https://community.rapidminer.com/t5/General-Chit-Chat/What-is-TensorFlow-A-typical-flow-of-TensorFlow/td-p/43513

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.misc import imread
from sklearn.metrics import accuracy_score
import tensorflow as tf
import pylab


# Let’s set a seed value, so that we can control our models randomness
seed = 128
rng = np.random.RandomState(seed)

root_dir = os.path.abspath('../..')
data_dir = os.path.join(root_dir, 'data')
sub_dir = os.path.join(root_dir, 'sub')

cur_dir_path = os.path.dirname(os.path.realpath(__file__))



# check for existence
os.path.exists(root_dir)
os.path.exists(data_dir)
os.path.exists(sub_dir)

# 随机生成1000个数据
data = pd.Series(np.random.randn(1000),index=np.arange(1000))
# 为了方便观看效果, 我们累加这个数据
data.cumsum()
# pandas 数据可以直接观看其可视化形式
data.plot()
plt.show()

# train = pd.read_csv(os.path.join(cur_dir_path, 'mnist_data', 'mnist_csv_label', 'mnist_train.csv'))
# test = pd.read_csv(os.path.join(cur_dir_path, 'mnist_data', 'mnist_csv_label', 'mnist_test.csv'))
# sample_submission = pd.read_csv(os.path.join(data_dir, 'Sample_Submission.csv'))

# train.head()

print train
# Let us see what our data looks like! We read our image and display it.

# img_name = rng.choice(train.filename)
# filepath = os.path.join(data_dir, 'Train', 'Images', 'train', img_name)

# img = imread(filepath, flatten=True)

# pylab.imshow(img, cmap='gray')
# pylab.axis('off')
# pylab.show()