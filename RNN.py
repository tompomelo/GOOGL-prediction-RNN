# Import Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import The Training Set

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Data Structure (/w 60 timestep and 1 output)

X_train = []
y_train = []
for i in range(60, 1258):
  X_train.append(training_set_scaled[i-60:i, 0])
  y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build RNN

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Init RNN

regressor = Sequential()

# First Layer
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Second Layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Third Layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Fouth Layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Output Layer
regressor.add(Dense(units = 1))

# Compile
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fit RNN to Training Set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
