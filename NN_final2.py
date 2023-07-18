# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 20:14:43 2023

@author: emili
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load data from the CSV file
data = pd.read_csv('C:\\Users\\emili\\OneDrive\\Documents\\3A\\Cours\\deep encode\\merged_table_cleaned.csv')

# Separate features (X) and labels (Y)
X = data[['Bitrate','width','height','spatial_info','temporal_info','fps','entropy']].values
Y = data['label'].values                  
X = X.astype(float)
Y = Y.astype(float)
                   

#normalization

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Normalize the features (X)
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# Create the model
model = Sequential()

# define the optimal parameters (CV results)
neuronsLayer = 64
nbLayers = 3
batchSize = 32
ep = 200


# Add the input layer
model.add(Dense(neuronsLayer, input_dim=7, activation='relu'))

# Add hidden layers
for _ in range(nbLayers):
    model.add(Dense(neuronsLayer, activation='relu'))

# Add the output layer
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_absolute_error', optimizer='adam')

# Train the model
model.fit(X_train, Y_train, epochs = ep, batch_size=batchSize, verbose=1)

# Evaluate the model on the testing set
loss = model.evaluate(X_test, Y_test)
print('Mean Absolute Error:', loss)

# # Make predictions on the test data
predictions = model.predict(X_test)
predictions = predictions.round().astype(int)  # Arrondir les valeurs prédites à l'entier le plus proche

#print('Predictions:', predictions)

# Calculate MSE
mse = mean_squared_error(Y_test, predictions)
print("MSE:", mse)

# Calculate R-squared
r_squared = r2_score(Y_test, predictions)

print("R-squared:", r_squared)


# Create a DataFrame to display the actual and predicted values
df_predictions = pd.DataFrame({'Actual Values': Y_test.flatten(), 'Predicted Values': predictions.flatten()})

print(df_predictions) 

# Plot a graph of actual values vs. predicted values 
fig, ax = plt.subplots(figsize=(15, 15))
ax.scatter(Y_test, predictions,s=20)
ax.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--', label='Diagonale line')
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')
ax.set_title('Comparison of Actual and Predicted Values')
ax.legend()
plt.show()
