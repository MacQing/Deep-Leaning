
# coding: utf-8

# In[50]:

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


# In[39]:

# 建立分类模型
dimx, dimy = 28 * 28, 10
x = tf.placeholder("float", [None, dimx])
W = tf.Variable(tf.zeros([dimx, dimy]))
b = tf.Variable(tf.zeros([1, dimy]))
y = tf.nn.softmax(tf.matmul(x, W) + b)


# In[40]:

# 设置优化方法

y_ = tf.placeholder("float", [None, dimy])
# 交叉熵
cross_entropy = - tf.reduce_sum(y_ * tf.log(y))
# 反向传播算法
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


# In[41]:

# 建立评估模型
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# In[57]:

# 初始化所有变量
init = tf.global_variables_initializer()
# 启动一个Session，并初始化所有变量
sess = tf.Session()
sess.run(init)
# 开始训练模型
mnist = input_data.read_data_sets("/Data", one_hot = True)

num_epoch = 10
for epoch in range(num_epoch):
    # 划分训练集和验证集
    train_data, test_data, train_lables, test_labels = train_test_split(mnist.train.images, mnist.train.labels, test_size=0.4, random_state = 0)
    
    train_data_size, batch_size = train_data.shape[0], 100
    num_batch = int(train_data_size / batch_size)
    for i in range(num_batch):
        # 每个batch在训练集中的索引
        batch_index = [j + (i)* batch_size for j in range(batch_size)]
        sess.run(train_step, feed_dict = {x: train_data[batch_index], y_: train_lables[batch_index]})
        
    # 计算训练误差和验证误差
    train_acc = sess.run(accuracy, feed_dict={x: train_data, y_: train_lables})
    valid_acc = sess.run(accuracy, feed_dict={x: test_data, y_: test_labels})
    print("Epoch ", epoch, "-", "train_acc: ", train_acc, "valid_acc: ", valid_acc)


# In[58]:

# 计算验证集误差
print("The accuracy in validation data: ", sess.run(accuracy, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels}))


# In[ ]:



