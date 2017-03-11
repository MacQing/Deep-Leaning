
# coding: utf-8

# In[517]:

# MLP for MNIST
# 在写该程序时遇到的最大困惑是：为什么网络最后不加softmax推断层，并且交叉熵用softmax_cross_entropy_with_logits()
# 最后得出的结论是：如果加了softmax推断层，那么交叉熵公式变成了tf.reduce_mean(tf.reduce_sum(-y * log(y_), 1))
# 这样，由于有的y_会无限趋于0，那么log(y_)会出现数值计算问题。
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sklearn.model_selection import train_test_split, KFold


# In[518]:

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

pickin = 0.8
layer1 = tf.nn.relu(tf.add(tf.matmul(x, wight["h1"]), bias["h1"]))
#layer1 = tf.nn.dropout(x=layer1,keep_prob=pickin)
layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, wight["h2"]), bias["h2"]))
#layer2 = tf.nn.dropout(x=layer2,keep_prob=pickin)
# sotmax前面的一层，即最后一个隐含层必须是线性
# 后面不需要加softmax层，否则在自己写交叉熵的程序时会遇到数值计算的问题（y_无穷接近0，造成nan）
y_ = (tf.add(tf.matmul(layer2, wight["output"]), bias["output"]))


# In[519]:

# define loss function: cross-entropy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_, y))

# define optimization method
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# define accuracy
accuracy = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(accuracy, "float"))


# In[520]:

# load MNIST data
mnist = input_data.read_data_sets("/Data", one_hot=True)
train_data, test_data, train_labels, test_labels = train_test_split(mnist.train.images, mnist.train.labels, test_size=0.4, random_state=1)


# In[521]:

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
        sess.run(train_step, feed_dict={x: train_data[index_batch], y: train_labels[index_batch]})
        
    # record the loss, training accuracy and validation accuracy
    loss_val = sess.run(loss, feed_dict={x: train_data, y: train_labels})
    train_acc = sess.run(accuracy, feed_dict={x: train_data, y: train_labels})
    val_acc = sess.run(accuracy, feed_dict={x: test_data, y: test_labels})
    print("Epoch", epoch + 1, "/", num_epochs, ":", "loss", loss_val, "train_acc", train_acc, "val_acc", val_acc)


# In[522]:

test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
print("The accuracy in test data:",  test_acc)

