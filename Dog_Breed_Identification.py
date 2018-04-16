import os
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf

def read_resize_data(file_route, width, height):
    files = os.listdir(file_route)
    data = np.zeros((len(files), width, height, 3)) # 保存处理后的图像
    i = 0 # 计数
    for file in files:
        image = cv2.imread(file_route +"/" +file)
        image = cv2.resize(image, (width, height))
        # image = tf.reshape(image, [width, height, 3])
        image = np.array(image)
        image = image / image.max() # 归一化

        data[i] = image
    return data

def one_hot(labels):
    labels_one_hot = pd.get_dummies(labels)
    return labels_one_hot

def next_batch(data, labels, num):
    len = data.shape[0]
    indices = np.arange(len)
    np.random.shuffle(indices)
    indices = indices[:num]
    train_mini = data[indices]
    labels_mini = one_hot(labels["breed"])
    labels_mini = labels_mini.loc[indices,]

    return train_mini, labels_mini

def weights_init(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def bias_init(len):
    return tf.Variable(tf.constant(0.5, shape=len))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def max_pool(x):
    # [batch, weight, height, channels]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

train_route = "C:/Song-Code/Practice/Dog Breed Identification/train"
test_route = "C:/Song-Code/Practice/Dog Breed Identification/test"
labels = pd.read_csv("C:/Song-Code/Practice/Dog Breed Identification/labels.csv")
train = read_resize_data(train_route, 60, 60)
test = read_resize_data(test_route, 60, 60)

x = tf.placeholder("float", shape=[None, 60, 60, 3])
y_ = tf.placeholder("float", shape=[None, 120])

# 设计神经网络结构
w_conv1 = weights_init([5, 5, 3, 32])
b_conv1 = bias_init([32])

h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1)

w_conv2 = weights_init([5, 5, 32, 64])
b_conv2 = bias_init([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2)

w_fc1 = weights_init([64*15*15, 1024])
b_fc1 = bias_init([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 64*15*15])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = weights_init([1024, 120])
b_fc2 = bias_init([120])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# 损失函数和梯度优化方法
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        train_mini, labels_mini = next_batch(train, labels, 50)
        train_step.run(feed_dict={x:train_mini, y_:labels_mini, keep_prob:0.5})
    saver = tf.train.Saver()
    model_path = "C:\Song-Code\model\Dog_Breed_Identification.ckpt"
    save_path = saver.save(sess, model_path)
    saver.restore(sess, model_path)
    result = sess.run(y_conv, feed_dict={x:test, keep_prob:1.0})
