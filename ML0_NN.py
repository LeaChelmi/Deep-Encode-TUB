# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 17:05:47 2023

@author: Léa
"""

import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

# Import the dataset 
x0 = pd.read_excel('C:\\Users\\Léa\\Videos\\features.xlsx')
y0 = pd.read_excel('C:\\Users\\Léa\Videos\\Label.xlsx')



x=x0[['width','height', 'spatial_info','temporal_info']]
y=y0['Bitrate']

#Split the dataset between train and test (70/30)

x_train, x_test = train_test_split(x, test_size=0.3)
y_train, y_test = train_test_split(y, test_size=0.3)



# Model definition
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),  # Input layer with 4 neurons corresponding to xi
    tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer with 64 neurons
    tf.keras.layers.Dense(1)  # Output layer with 1 neuron corresponding to y
])

# Model compilation
model.compile(optimizer='adam', loss='mean_squared_error')

# Model training
model.fit(x_train, y_train, epochs=100)

# Prediction
predictions = model.predict(x_test)

# Displaying predictions
for i in range(len(predictions)):
    print('Input:', x_test[i], 'Predicted y:', predictions[i])

