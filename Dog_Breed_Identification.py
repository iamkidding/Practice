import os
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf

def read_resize_train_val(file_route, width, height):
    files = os.listdir(file_route)
    val_num = 1000 # 选1000个图片做validation,通过labels，前9222个图片包括所有的标签
    train_num = len(files) - 1000
    train = np.zeros((train_num, width, height, 3)) # 保存处理后的图像
    val = np.zeros((val_num, width, height, 3)) # 保存处理后的图像
    i = 0 # 计数
    for file in files:
        image = cv2.imread(file_route +"/" +file)
        image = cv2.resize(image, (width, height))
        # image = tf.reshape(image, [width, height, 3])
        image = np.array(image)
        image = image - image.mean() # 零均值
        image /= np.std(image) # 归一化

        if i < train_num:
            train[i] = image
        else:
            val[i-train_num] = image
        i += 1
    return train, val

def read_resize_test(file_route, width, height):
    files = os.listdir(file_route)
    data = np.zeros((len(files), width, height, 3)) # 保存处理后的图像
    i = 0 # 计数
    for file in files:
        image = cv2.imread(file_route +"/" +file)
        image = cv2.resize(image, (width, height))
        image = np.array(image)
        image = image - image.mean() # 零均值
        image /= np.std(image) # 归一化

        data[i] = image
        i += 1
    return data

def image_augmentation(data, labels, num): # 暂时只实现水平翻转
    for i in range(num):
        random_index = np.random.randint(len(data))
        # method = np.random.randint(1)
        # if method == 0:
        image_aug = cv2.flip(data[random_index], 1)
        # else:
        #     image_aug = cv2.
        image_aug = np.reshape(image_aug, (1, 64, 64, 3))
        data = np.concatenate((data, image_aug))
        if i == 0: # 第一轮之后，labels的type由DataFrame变为npdarray
            labels = np.concatenate((labels, labels.loc[random_index,:].values.reshape(1, 2)))
        else:
            labels = np.concatenate((labels, np.reshape(labels[random_index,:],(1, 2))))

    return data, labels

def one_hot(labels):
    labels_one_hot = pd.get_dummies(labels)
    return labels_one_hot

def next_batch(data, labels, num, is_df):
    len = data.shape[0]
    indices = np.arange(len)
    np.random.shuffle(indices)
    indices = indices[:num]
    train_mini = data[indices]
    # 训练的labels是nparray， 验证集的是df
    if is_df is not True:
        labels_mini = one_hot(labels[:, 1])
    else:
        labels_mini = one_hot(labels["breed"])
    #经过one_hot函数后变为df
    labels_mini = labels_mini.loc[indices]

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
train_labels = labels.loc[:9222, :]
val_labels = labels.loc[9222:, :].reset_index(drop=True)
classes = pd.get_dummies(labels["breed"]).columns

test_id = os.listdir(test_route) # 文件名带着扩展名
for i in range(len(test_id)):
    test_id[i] = test_id[i].split(".")[0]

train, val = read_resize_train_val(train_route, 64, 64)
train, train_labels = image_augmentation(train, train_labels, 1000)
test = read_resize_test(test_route, 64, 64)

x = tf.placeholder("float", shape=[None, 64, 64, 3])
y = tf.placeholder("float", shape=[None, 120])
keep_prob = tf.placeholder("float")  # dropout的比例
is_training = tf.placeholder("bool")

# 设计神经网络结构
if is_training is not False:
    x = tf.nn.dropout(x, keep_prob) # 在训练时将x drop掉一半
bnx = tf.layers.batch_normalization(x, training=is_training)
# 卷积第一层
w_conv1 = weights_init("w_conv1", [5, 5, 3, 32])
b_conv1 = bias_init([32])
bn1 = tf.layers.batch_normalization(conv2d(x, w_conv1) + b_conv1, training=is_training)
h_conv1 = tf.nn.relu(bn1)
h_conv1_drop = tf.nn.dropout(h_conv1, keep_prob)
h_pool1 = max_pool(h_conv1_drop)
# 第二层
w_conv2 = weights_init("w_conv2", [5, 5, 32, 64])
b_conv2 = bias_init([64])
bn2 = tf.layers.batch_normalization(conv2d(h_pool1, w_conv2) + b_conv2, training=is_training)
h_conv2 = tf.nn.relu(bn2)
h_conv2_drop = tf.nn.dropout(h_conv2, keep_prob)
h_pool2 = max_pool(h_conv2_drop)
# 第三层
w_conv3 = weights_init("w_conv3", [5, 5, 64, 128])
b_conv3 = bias_init([128])
bn3 = tf.layers.batch_normalization(conv2d(h_pool2, w_conv3) + b_conv3, training=is_training)
h_conv3 = tf.nn.relu(bn3)
h_conv3_drop = tf.nn.dropout(h_conv3, keep_prob=keep_prob)
h_pool3 = max_pool(h_conv3_drop)
# 全连接层第一层
w_fc1 = weights_init("w_fc1", [128*8*8, 1024])
b_fc1 = bias_init([1024])
h_pool3_flat = tf.reshape(h_pool3, [-1, 128*8*8])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# 第二层
w_fc2 = weights_init("w_fc2", [1024, 120])
b_fc2 = bias_init([120])

# y_ = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2) # 预测时候用
y_ = tf.matmul(h_fc1_drop, w_fc2) + b_fc2  # 训练时候用

# 损失函数
# cross_entropy = -tf.reduce_sum(y_*tf.log(y_+1e-10))
# 使用下面的损失函数，在网络中的最后输出为参数*全连接层加上偏置，而不是softmax的结果
loss_function = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)
cross_entropy = tf.reduce_sum(loss_function)

# 梯度优化方法
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
# (batch, lr, loss) (50, 1, 250左右)  (50, 1e-3, 240左右) (50, 1e-10, 11000) (50, 1e-5, 开始下降最后稳定在230）(50, 5e-7, 240）
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(16000):
        train_mini, labels_mini = next_batch(train, train_labels, 50, False)
        if i % 100 == 0: # 每100次进行一次验证
            val_mini, vl_mini = next_batch(val, val_labels, 50, True)  # 验证集
            train_accuracy = accuracy.eval(feed_dict={x: train_mini, y: labels_mini, keep_prob: 1.0, is_training: True})
            val_loss = cross_entropy.eval(feed_dict={x: val_mini, y: vl_mini, keep_prob: 1.0, is_training: False})
            loss = cross_entropy.eval(feed_dict={x: train_mini, y: labels_mini, keep_prob: 1.0, is_training: True})
            print("step %d, training acc %g, validation loss %g, loss %g" % (i, train_accuracy, val_loss, loss))
        train_step.run(feed_dict={x:train_mini, y:labels_mini, keep_prob:0.5, is_training:True})
        if i % 1000 == 0: # 每1000次存一下模型
            saver = tf.train.Saver()
            model_path = "C:\Song-Code\model\Dog_Breed_Identification1.ckpt"
            save_path = saver.save(sess, model_path)

    saver = tf.train.Saver()
    model_path = "C:\Song-Code\model\Dog_Breed_Identification.ckpt"
    saver.restore(sess, model_path)
    validation = sess.run(y_, feed_dict={x: val, keep_prob:1.0, is_training:False})
    # for i in range(11):
    #     if i < 10:
    #         temp = test[i*1000: (i+1)*1000]
    #     else:
    #         temp = test[i*1000:]
    #     result = sess.run(y_, feed_dict={x:temp, keep_prob:1.0, is_training: False})
    #     if i == 0:
    #         re_pd = pd.DataFrame(result)
    #     else:
    #         temp2 = pd.DataFrame(result)
    #         re_pd = pd.concat([re_pd, temp2], ignore_index=True)
    #
    # re_pd.columns = classes # 重名列
    # re_pd.insert(0, "id", test_id) # 插入id
    # re_pd.to_csv("C:/Song-Code/Practice/Dog Breed Identification/sub.csv", index=False)
