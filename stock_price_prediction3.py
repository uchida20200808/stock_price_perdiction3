#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Building a Stock Price Predictor Using Python
https://www.section.io/engineering-education/stock-price-prediction-using-python/
'''


# In[2]:


'''
What is a RNN?

When you read this text, 
you understand each word based on previous words in your brain. 

You wouldn’t start thinking from scratch,
rather your thoughts are cumulative. 
Recurrent Neural Networks implement the same concept using machines; 

they have loops and allow information to persist where traditional neural networks can’t.

Let’s use a few illustrations to demonstrate how a RNN works.
'''


# In[3]:


'''
What is LSTM?

Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture 
that you can use in the deep learning field. 

In LSTM, you can process an entire sequence of data. 
For example, handwriting generation, question answering or speech recognition, and much more.

Unlike the traditional feed-forward neural network, 
that passes the values sequentially through each layer of the network, 
LSTM has a feedback connection that helps it remember preceding information, 
making it the perfect model for our needs to do time series analysis.
'''


# In[4]:


'''
Choosing data

In this tutorial, 
I will use a TESLA stock dataset from Yahoo finance, 
that contains stock data for ten years. 
You can download it for free from here.

I’m also going to use Google Colab 
because it’s a powerful tool, 
but you are free to use whatever you are comfortable with.
'''


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout


# In[6]:


'''
We are going to use numpy for scientific operations,
pandas to modify our dataset, 
matplotlib to visualize the results, 
sklearn to scale our data, 
and keras to work as a wrapper on low-level libraries like TensorFlow or Theano high-level neural networks library.
'''


# In[7]:


'''
Let’s build our app!

First of all, 
if you take a look at the dataset, 
you need to know that 
the “open” column represents the opening price for the stock at that “date” column, 
and the “close” column is the closing price on that day. 
The “High” column represents the highest price reached that day, 
and the “Low” column represents the lowest price.
'''


# In[8]:


#we need to make a data frame:
df = pd.read_csv('TSLA.csv')


# In[9]:


df.shape


# In[10]:


#To make it as simple as possible we will just use one variable which is the “open” price.
df = df['Open'].values
#1列のデータに変換する。行数は変換前のデータ数から計算される。
#https://qiita.com/yosshi4486/items/deb49d5a433a2c8a8ed4
df = df.reshape(-1, 1)


# In[11]:


df.shape


# In[12]:


'''
The reshape allows you to add dimensions or change the number of elements in each dimension. 
We are using reshape(-1, 1)
because we have just one dimension in our array, 
so numby will create the same number of our rows and add one more axis: 
1 to be the second dimension.
'''


# In[13]:


#intは整数値に変換する
#https://www.javadrive.jp/python/function/index3.html
dataset_train = np.array(df[:int(df.shape[0]*0.8)])
dataset_test = np.array(df[int(df.shape[0]*0.8):])
print(dataset_train.shape)
print(dataset_test.shape)


# In[14]:


'''
We will use the MinMaxScaler to scale our data between zero and one. 
In simpler words, 
the scaling is converting the numerical data represented in a wide range into a smaller one.
'''


# In[15]:


scaler = MinMaxScaler(feature_range=(0,1))
dataset_train = scaler.fit_transform(dataset_train)
dataset_train[:5]


# In[16]:


dataset_test = scaler.transform(dataset_test)
dataset_test[:5]


# In[17]:


#Next, we will create the function that will help us to create the datasets:
def create_dataset(df):
    x = []
    y = []
    for i in range(50, df.shape[0]):
        x.append(df[i-50:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y 


# In[18]:


'''
For the features (x), 
we will always append the last 50 prices, 
and for the label (y), 
we will append the next price. 
Then we will use numpy to convert it into an array.

Now we are going to create our training and testing data 
by calling our function for each one:
'''


# In[19]:


x_train, y_train = create_dataset(dataset_train)
x_test, y_test = create_dataset(dataset_test)


# In[20]:


'''
Next, we need to reshape our data to make it a 3D array in order to use it in LSTM Layer.
'''


# In[ ]:





# In[21]:


model = Sequential()
model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=96,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96))
model.add(Dropout(0.2))
model.add(Dense(units=1))


# In[22]:


'''
First, we initialized our model as a sequential one with 96 units in the output’s dimensionality. 
We used return_sequences=True 
to make the LSTM layer with three-dimensional input and input_shape to shape our dataset.

Making the dropout fraction 0.2 drops 20% of the layers. 
Finally, we added a dense layer with a value of 1 
because we want to output one value.

After that, we want to reshape our feature for the LSTM layer, 
because it is sequential_3 
which is expecting 3 dimensions, not 2:
'''


# In[23]:


x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[24]:


model.compile(loss='mean_squared_error', optimizer='adam')


# In[25]:


'''
We used loss='mean_squared_error' 
because it is a regression problem, 
and the adam optimizer to update network weights iteratively based on training data.

We are ready!

Let’s save our model and start the training:
'''


# In[26]:


model.fit(x_train, y_train, epochs=25, batch_size=32)
model.save('stock_prediction.h5')


# In[27]:


'''
Every epoch refers to one cycle 
through the full training dataset, 
and batch size refers to the number of training examples utilized in one iteration.
'''


# In[28]:


model = load_model('stock_prediction.h5')


# In[29]:


'''
Results visualization

The last step is to visualize our data. 
If you are new to data visualization 
please consider going through our Getting Started with Data Visualization 
using Pandas tutorial first.
'''


# In[30]:


predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

fig, ax = plt.subplots(figsize=(16,8))
ax.set_facecolor('#000041')
ax.plot(y_test_scaled, color='red', label='Original price')
plt.plot(predictions, color='cyan', label='Predicted price')
plt.legend()


# In[ ]:




