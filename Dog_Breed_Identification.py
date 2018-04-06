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
        image = np.array(image)
        image = image / image.max() # 归一化

        data[i] = image
    return data

def weights_init(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def bias_init(len):
    return tf.Variable(tf.constant(0.5, shape=len))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 3, 3, 1], padding="SAME")

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 3, 3, 1], padding="SAME")

train_route = "C:/Song-Code/Practice/Dog Breed Identification/train"
test_route = "C:/Song-Code/Practice/Dog Breed Identification/test"
labels = pd.read_csv("C:/Song-Code/Practice/Dog Breed Identification/labels.csv")
train = read_resize_data(train_route, 60, 60)
test = read_resize_data(test_route, 60, 60)
