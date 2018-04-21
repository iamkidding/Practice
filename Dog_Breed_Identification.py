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
        image = image - image.mean() # 零均值
        image /= np.std(image) # 归一化

        data[i] = image
        i += 1
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

def weights_init(name,shape):
    # return tf.Variable(tf.truncated_normal(shape, stddev=0.01))
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_init(len):
    return tf.Variable(tf.constant(0.0, shape=len))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def max_pool(x):
    # [batch, weight, height, channels]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

train_route = "C:/Song-Code/Practice/Dog Breed Identification/train"
test_route = "C:/Song-Code/Practice/Dog Breed Identification/test"
labels = pd.read_csv("C:/Song-Code/Practice/Dog Breed Identification/labels.csv")
classes = pd.get_dummies(labels["breed"]).columns

test_id = os.listdir(test_route) # 文件名带着扩展名
for file in test_id:
    file = file.split(".")[0]

train = read_resize_data(train_route, 64, 64)
test = read_resize_data(test_route, 64, 64)

x = tf.placeholder("float", shape=[None, 64, 64, 3])
y = tf.placeholder("float", shape=[None, 120])

# 设计神经网络结构
is_training = tf.placeholder("bool")
# 卷积第一层
w_conv1 = weights_init("w_conv1", [5, 5, 3, 32])
# w_conv1 = tf.get_variable("w_conv1", shape=[5, 5, 3, 32], initializer=tf.contrib.layers.xavier_initializer())
b_conv1 = bias_init([32])
bn1 = tf.layers.batch_normalization(conv2d(x, w_conv1) + b_conv1, training=is_training)
h_conv1 = tf.nn.relu(bn1)
h_pool1 = max_pool(h_conv1)
# 第二层
w_conv2 = weights_init("w_conv2", [5, 5, 32, 64])
# w_conv2 = tf.get_variable("w_conv2", shape=[5, 5, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
b_conv2 = bias_init([64])
bn2 = tf.layers.batch_normalization(conv2d(h_pool1, w_conv2) + b_conv2, training=is_training)
h_conv2 = tf.nn.relu(bn2)
h_pool2 = max_pool(h_conv2)
# 第三层
w_conv3 = weights_init("w_conv3", [5, 5, 64, 128])
# w_conv3 = tf.get_variable("w_conv3", shape=[5, 5, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
b_conv3 = bias_init([128])
bn3 = tf.layers.batch_normalization(conv2d(h_pool2, w_conv3) + b_conv3, training=is_training)
h_conv3 = tf.nn.relu(bn3)
h_pool3 = max_pool(h_conv3)
# 全连接层第一层
w_fc1 = weights_init("w_fc1", [128*8*8, 1024])
# w_fc1 = tf.get_variable("w_fc1", shape=[128*8*8, 1024], initializer=tf.contrib.layers.xavier_initializer())
b_fc1 = bias_init([1024])
h_pool3_flat = tf.reshape(h_pool3, [-1, 128*8*8])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1) + b_fc1)
keep_prob = tf.placeholder("float")  # dropout的比例
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# 第二层
w_fc2 = weights_init("w_fc2", [1024, 120])
# w_fc2 = tf.get_variable("w_fc2", shape=[1024, 120], initializer=tf.contrib.layers.xavier_initializer())
b_fc2 = bias_init([120])

# y_ = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
y_ = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

# 损失函数
# cross_entropy = -tf.reduce_sum(y_*tf.log(y_+1e-10))
loss_function = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)
cross_entropy = tf.reduce_sum(loss_function)

# 梯度优化方法
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
# (batch, lr, loss) (50, 1, 250左右)  (50, 1e-3, 240左右) (50, 1e-10, 11000) (50, 1e-5, 开始下降最后稳定在230）(50, 5e-7, 240）
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        train_mini, labels_mini = next_batch(train, labels, 50)
        if i % 100 == 0: # 每100次进行一次验证
            train_accuracy = accuracy.eval(feed_dict={x: train_mini, y: labels_mini, keep_prob: 1.0, is_training: True})
            loss = cross_entropy.eval(feed_dict={x: train_mini, y: labels_mini, keep_prob: 1.0, is_training: True})
            print("step %d, training accuracy %g, loss %g" % (i, train_accuracy, loss))
        train_step.run(feed_dict={x:train_mini, y:labels_mini, keep_prob:0.5, is_training:True})

    saver = tf.train.Saver()
    model_path = "C:\Song-Code\model\Dog_Breed_Identification.ckpt"
    save_path = saver.save(sess, model_path)
    saver.restore(sess, model_path)

    result = sess.run(y_, feed_dict={x:test, keep_prob:1.0, is_training: False})
    re_pd = pd.DataFrame(result)
    re_pd.columns = classes # 重名列
    re_pd.insert(0, "id", test_id) # 插入id
    re_pd.to_csv("C:/Song-Code/Practice/Dog Breed Identification/sub.csv", index=False)
