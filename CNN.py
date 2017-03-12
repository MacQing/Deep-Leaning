
# coding: utf-8

# In[17]:

# CNN
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sklearn.model_selection import train_test_split


# In[18]:

image_width = 28
labels_num = 10
# note that, the shape of MNIST's image is [1, 28*28]
# when apply conv operator, we need reshape the vector into a matrix
x = tf.placeholder(dtype="float", shape=[None, image_width*image_width])
y = tf.placeholder(dtype="float", shape=[None, labels_num])
image_x = tf.reshape(tensor=x, shape=[-1, image_width, image_width, 1])

# 1st conv+pooling
# 32 filter with shape of [5, 5, 1]
W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(value=0.1, shape=[1, 32]))
layer_conv1 = tf.nn.conv2d(input=image_x, filter=W_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1
layer_conv1 = tf.nn.relu(layer_conv1)
layer_pool1 = tf.nn.max_pool(layer_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# 2st conv+pooling
# 64 filter with shape of [5, 5, 1]
W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(value=0.1, shape=[1, 64]))
layer_conv2 = tf.nn.conv2d(input=layer_pool1, filter=W_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2
layer_conv2 = tf.nn.relu(layer_conv2)
layer_pool2 = tf.nn.max_pool(layer_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# full connect layer
W_fc1 = tf.Variable(tf.truncated_normal(shape=[7*7*64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(value=0.1, shape=[1, 1024]))
layer_pool2_flat = tf.reshape(tensor=layer_pool2, shape=[-1, 7*7*64])
layer_fc1 = tf.nn.relu(tf.matmul(layer_pool2_flat, W_fc1) + b_fc1)
# dropout
keep_prob = tf.placeholder(dtype="float", shape=None)
layer_fc1_dropout = tf.nn.dropout(layer_fc1, keep_prob=keep_prob)

# output layer
W_out = tf.Variable(tf.truncated_normal(shape=[1024, labels_num], stddev=0.1))
b_out = tf.Variable(tf.constant(value=0.1, shape=[1, labels_num]))
layer_out = tf.matmul(layer_fc1_dropout, W_out) + b_out


# In[19]:

# cross-entropy
loss = tf.nn.softmax_cross_entropy_with_logits(logits=layer_out, labels=y)
# accuracy
acc = tf.equal(tf.arg_max(y, 1), tf.arg_max(layer_out, 1))
acc = tf.reduce_mean(tf.cast(acc, "float"))


# In[20]:

# load MNIST data
mnist = input_data.read_data_sets("/Data", one_hot=True)
train_data, test_data, train_labels, test_labels = train_test_split(mnist.train.images, mnist.train.labels, test_size=0.4, random_state=1)


# In[ ]:

# optimization method
train_step = tf.train.AdadeltaOptimizer(0.01).minimize(loss)

# training
num_epochs, size_batch = 20, 100
num_batchs = int(train_data.shape[0] / size_batch)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for epoch in range(num_epochs):
    for batch in range(num_batchs):
        index_batch = [j + (batch) * size_batch for j in range(size_batch)]
        #printnt("index_batch:", index_batch)
        sess.run(train_step, feed_dict={x: train_data[index_batch], y: train_labels[index_batch], keep_prob:0.5})
        
    # record the loss, training accuracy and validation accuracy
    loss_val = sess.run(loss, feed_dict={x: train_data, y: train_labels})
    train_acc = sess.run(acc, feed_dict={x: train_data, y: train_labels})
    val_acc = sess.run(acc, feed_dict={x: test_data, y: test_labels})
    print("Epoch", epoch + 1, "/", num_epochs, ":", "loss", loss_val, "train_acc", train_acc, "val_acc", val_acc)


# In[ ]:



