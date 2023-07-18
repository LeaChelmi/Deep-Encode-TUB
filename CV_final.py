# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 20:27:52 2023

@author: emili
"""

import pandas as pd
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# Load data from a CSV file
data = pd.read_csv('C:\\Users\\emili\\OneDrive\\Documents\\3A\\Cours\\deep encode\\merged_table_cleaned.csv')

# Separate features (X) and labels (Y)
X = data[['Bitrate','width','height','spatial_info','temporal_info','fps','entropy']].values
Y = data['label'].values                  
X = X.astype(float)
Y = Y.astype(float)

# Perform data normalization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create a function that builds the model
def build_model(neurons_input, neurons_output, num_layers, neurons_layer):
    model = Sequential()
    model.add(Dense(neurons_layer, input_dim=neurons_input, activation='relu'))

    for _ in range(num_layers - 1):
        model.add(Dense(neurons_layer, activation='relu'))

    model.add(Dense(neurons_output, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Create the KerasRegressor model for cross-validation
model = KerasRegressor(build_fn=build_model, verbose=0)

# Define the hyperparameters to tune
param_grid = {
    'neurons_input': [X.shape[1]],
    'neurons_output': [1],
    'num_layers': [1, 2, 3],
    'neurons_layer': [4, 8, 16, 32,64],
    'batch_size': [16, 32, 64],
    'epochs': [100,200,400] 
}

# Perform cross-validation with grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_result = grid.fit(X, Y)

# Display the grid search results
print("Best MSE:", -grid_result.best_score_)
print("Best Parameters:", grid_result.best_params_)
