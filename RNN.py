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
for i in range(60, 1258)
  X_train.append
