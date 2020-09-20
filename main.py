# This prgram determines price of stocks using AI and ML

import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Collect and show data of stocks
df=web.DataReader ('GOOGL',data_source='yahoo',start='2010-01-01',end='2020-9-17')

# Show the data

print(df)

# Show no of rows and coloumn

print(df.shape)

# Visualize the data

plt.figure(figsize=(16,8))
plt.title=('Close price history')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Closing price USD ', fontsize=18)
plt.show()

# Create a new dataframe with only a new Close Coloumn

data = df.filter(['Close'])

# Convert dataframe to a numpy array
dataset = data.values

# Get the row numbers to train the model

training_data_len = math.ceil( len(dataset)*.8)
print (training_data_len)

# Scale the data

scaler = MinMaxScaler (feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
print (scaled_data)

# Create training and scaled dataset

train_data=scaled_data [0: training_data_len, :]

# Splitting data into X_train and Y_train

x_train = []
y_train = []

for i in range (60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()

# Converting x_train & y_train in numpy array

x_train,y_train= np.array(x_train), np.array(y_train)

# Reshape the data
x_train =np.reshape (x_train, (x_train.shape[0], x_train.shape[1], 1))
print (x_train.shape)

# Build the lstm model

model = Sequential()
model.add (LSTM(50, return_sequences= True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model

# Train the model

model.fit(x_train, y_train, batch_size= 1, epochs= 1)

# Create a testing dataset

test_data = scaled_data[training_data_len-60: , :]

# Creating datasets x_test and y_test

x_test= []
y_test= dataset[training_data_len: , :]
for i in range (60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert the data into numpy array

x_test = np.array(x_test)

x_test= np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
print (x_test.shape)

# Get the predicted price

predictions =  model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get root mean squared error
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
print(rmse)

# Plot and visualize the data

train = data[:training_data_len]
valid= data [training_data_len:]
valid ['Predictions']= predictions

plt.figure(figsize=(16,8))
plt.title = ('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Closing price in USD', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc = 'lower right')
plt.show()

# Show valid price

print (valid)

company_quote = web.DataReader('GOOGL', data_source='yahoo',start='2010-01-01',end='2020-9-17')

new_df = company_quote.filter(['Close'])

# Get last 60 days of dataframe & cobvert to an array
last_60_days = new_df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)

x_test=[]
x_test.append(last_60_days_scaled)
x_test=np.array(x_test)
x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_price = model.predict(x_test)

# Undo the scaling
predicted_price = scaler.inverse_transform(predicted_price)
print ("The predicted value is" ,predicted_price)

# Get the quote
company_quote2 = web.DataReader('GOOGL', data_source='yahoo',start='2020-09-18',end='2020-9-18')
print("The actual price of the given date is" ,company_quote2['Close'])