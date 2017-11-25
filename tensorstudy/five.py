#!/usr/bin/env python
# coding=utf8

# https://jizhi.im/blog/post/aia-1



# 载入扩展库，并输出TensorFlow的版本
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.cross_validation import train_test_split

print(tf.__version__)


# 载入数据，划分训练/测试集
iris = datasets.load_iris() #鸢yuan尾属植物

train_X, test_X, train_y, test_y = train_test_split(iris.data, iris.target, test_size = 5, random_state = 0)

print(len(train_X), train_X[0]) # 145 第一个数据为 [ 6.3  3.3  6.   2.5]
print(train_y[0])


#特征选择与模型搭建

feature_name = "flower_features"
feature_columns = [tf.feature_column.numeric_column(feature_name,
                                                    shape=[4])]

classifier = tf.estimator.LinearClassifier(
    feature_columns=feature_columns,
    n_classes=3,
    model_dir="./iris_model")


# 输入函数，讲导入的数据转换为TensorFlow数据类型
def input_fn(set_split='train'):
    def _fn():
        if set_split == "test":
            features = {feature_name: tf.constant(test_X)}
            label = tf.constant(test_y)
        else:
            features = {feature_name: tf.constant(train_X)}
            label = tf.constant(train_y)
        return features, label
    return _fn




# 训练（拟合）模型
classifier.train(input_fn=input_fn(),
                 steps=1000)
print('fit done')


# 评估准确率
accuracy_score = classifier.evaluate(input_fn=input_fn('test'),
                                     steps=100)["accuracy"]
print('\nAccuracy: {0:f}'.format(accuracy_score))



new_samples = np.array(
      [[6.4, 3.2, 4.5, 1.5],
     [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"flower_features": new_samples},
    num_epochs=1,
    shuffle=False)

#得到一个 generator 对象 需要迭代取出数据
#后面 predict 是来自于 https://www.tensorflow.org/get_started/estimator
gen_rs = classifier.predict(input_fn=predict_input_fn)
predictions = list(gen_rs)
# print(predictions[0]["classes"])
predicted_classes = [p["classes"] for p in predictions]

print('predict', predicted_classes)


