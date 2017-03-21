
# coding: utf-8

# In[65]:

import numpy as np
from keras.models import  Sequential
from keras.layers import LSTM, Dense, Flatten
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[66]:

x = np.linspace(0, 10, 1000)
y = np.sin(x)
# construct datasets
time_steps = 9
data_size = x.shape[0] - time_steps
data_x = np.zeros((data_size, time_steps))
data_y = np.zeros(data_size)
for i in range(data_size):
    data_x[i] = x[i : i + time_steps]
    data_y[i] = y[i + time_steps]
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.4, random_state = 1)
train_x = np.expand_dims(train_x, axis=2)
test_x = np.expand_dims(test_x, axis=2)
train_x.shape, train_y.shape, test_x.shape, test_y.shape

plt.plot(data_y)
plt.show()


# In[82]:

seq = Sequential()
seq.add(LSTM(units=50, input_shape=(time_steps, 1), return_sequences=True))
seq.add(LSTM(units=50, return_sequences=False))
seq.add(Dense(1, activation='linear'))
seq.summary()
seq.compile(optimizer='Adadelta',
              loss='mean_squared_error', metrics=["accuracy"])


# In[83]:

seq.fit(train_x, train_y, epochs=10, batch_size=10)


# In[84]:

predicted_y = seq.predict(np.expand_dims(data_x, axis=2))
plt.plot(predicted_y)
plt.hold(True)
plt.plot(data_y)
plt.show()


# In[ ]:



