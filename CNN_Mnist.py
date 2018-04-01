import pandas as pd
import numpy as np
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data

# 读取数据
def read_data(file_name):
    # 将数据shuffle
    data = pd.read_csv(file_name).sample(frac=1.0)
    data = data.reset_index(drop=True)
    labels = data["label"]
    # 无通用性，pandas列切片要使用列名，列名为字符串不能用整数1:9之类index的切片
    data = data.loc[:, "pixel0":]
    num_classes = pd.unique(labels).size
    # 将label转换成one_hot形式
    labels_one_hot = np.zeros((labels.size, num_classes))
    for i in range(labels.size):
        labels_one_hot[i][labels[i]] = 1
    return data, labels_one_hot

def to_excel(data):
    data_set = pd.Series(data)
    index = [i for i in range(1, data_set.shape[0]+1)]
    submission = pd.DataFrame({'ImageId':pd.Series(index), 'Label':data_set[0]})
    submission.to_csv("C:/Song-Code/Practice/CNN_Mnist_sub.csv", index=False)

# convert class labels from scalars to one-hot vectors e.g. 1 => [0 1 0 0 0 0 0 0 0 0]
# 另一种转换标签的方法，来自kaggle
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def one_hot_to_dense(data):
    # axis: Bydefault, the index is into the flattened array, otherwise along
    # the specified axis. axis=1按行，axis=0,列
    return data.argmax(axis=1)

def weight_variable(shape):
    # 截尾正态分布
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def min_max_norm(data):
    norm = data / data.max()
    norm = norm.fillna(0)
    return norm

def next_batch(num, train, labels):
    train_mini = train.sample(num)
    labels_mini = labels[train_mini.index]
    train_mini.reset_index(drop=True)

    return train_mini, labels_mini

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # strides[0]和strides[3]必须为1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
train, labels = read_data("C:\Song-Code\Practice\Digit Recognizer/train.csv")
test = pd.read_csv("C:\Song-Code\Practice\Digit Recognizer/test.csv")
train = min_max_norm(train)
test = min_max_norm(test)
x = tf.placeholder("float", shape=[None, 784])
# x_test = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# -1表示该位置由函数自动计算得出，由总的x的个数除以其他维数
x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

w_conv2 = weight_variable([5, 5, 32, 64]) # ?为什么是[5, 5, 32, 64]
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        train_mini, labels_mini = next_batch(50, train, labels)
        # if i % 100 == 0:
        #     train_accuracy = accuracy.eval(feed_dict={x:train_mini, y_:labels_mini, keep_prob:1.0})
        #     print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x:train_mini, y_:labels_mini, keep_prob:0.5})
    saver = tf.train.Saver()
    model_path = "C:\Song-Code\model\CNN_Mnist.ckpt"
    save_path = saver.save(sess, model_path)
    saver.restore(sess, model_path)
    result = sess.run(y_conv, feed_dict={x:test, keep_prob:1.0})

#     for i in range(20000):
#         batch = mnist.train.next_batch(50)
#         if i%100 == 0:
#             train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
#             print("step %d, training accuracy %g" % (i, train_accuracy))
#         train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
#     print("test accuracy %g" % accuracy.eval(feed_dict={
#         x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
