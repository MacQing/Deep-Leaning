
# coding: utf-8

# In[306]:

# MLP for MNIST
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sklearn.model_selection import train_test_split, KFold


# In[307]:

# buld MLP
dim_x, dim_y = 28*28, 10
x = tf.placeholder(dtype="float", shape=[None, dim_x])
y = tf.placeholder(dtype="float", shape=[None, dim_y])

num_nodes = {"h1": 100, "h2": 100}
wight = {
    "h1": tf.Variable(tf.random_normal([dim_x, num_nodes["h1"]])),
    "h2": tf.Variable(tf.random_normal([num_nodes["h1"], num_nodes["h2"]])),
    "output": tf.Variable(tf.random_normal([num_nodes["h2"], dim_y]))
}
bias = {
    "h1": tf.Variable(tf.random_normal([1, num_nodes["h1"]])),
    "h2": tf.Variable(tf.random_normal([1, num_nodes["h2"]])),
    "output": tf.Variable(tf.random_normal([1, dim_y])),
}

layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, wight["h1"]), bias["h1"]))
layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, wight["h2"]), bias["h2"]))
layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, wight["output"]), bias["output"]))
y_ = tf.nn.softmax(layer3)


# In[308]:

# define loss function: cross-entropy
loss = -tf.reduce_sum(y * tf.log(y_))

# define optimization method
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# define accuracy
accuracy = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(accuracy, "float"))


# In[309]:

# load MNIST data
mnist = input_data.read_data_sets("/Data", one_hot=True)
train_data, test_data, train_labels, test_labels = train_test_split(mnist.train.images, mnist.train.labels, test_size=0.4, random_state=1)


# In[310]:

# training
num_epochs, size_batch = 10, 100
num_batchs = int(train_data.shape[0] / size_batch)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for epoch in range(num_epochs):
    for batch in range(num_batchs):
        index_batch = [j + (batch) * size_batch for j in range(size_batch)]
        #printnt("index_batch:", index_batch)
        sess.run(train_step, feed_dict={x: train_data[index_batch], y: train_labels[index_batch]})
        
    # record the loss, training accuracy and validation accuracy
    loss_val = sess.run(loss, feed_dict={x: train_data, y: train_labels})
    train_acc = sess.run(accuracy, feed_dict={x: train_data, y: train_labels})
    val_acc = sess.run(accuracy, feed_dict={x: test_data, y: test_labels})
    print("Epoch", epoch + 1, "/", num_epochs, ":", "loss-", loss_val, "train_acc-", train_acc, "val_acc-", val_acc)


# In[311]:

test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
print("The accuracy in test data:",  test_acc)


# In[ ]:



