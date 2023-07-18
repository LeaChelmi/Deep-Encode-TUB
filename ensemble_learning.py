from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin

# Load data from the CSV file
data = pd.read_csv('C:\\Users\\emili\\OneDrive\\Documents\\3A\\Cours\\deep encode\\merged_table_cleaned.csv')

# Separate features (X) and labels (Y)
X = data[['Bitrate', 'width', 'height', 'spatial_info', 'temporal_info', 'fps', 'entropy']].values
Y = data['label'].values
X = X.astype(float)
Y = Y.astype(float)

# Normalize the features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# # Create the decision tree regressor
tree_regressor = DecisionTreeRegressor(random_state=0)

# Create the neural network regressor
neural_network_regressor = MLPRegressor(hidden_layer_sizes=(64, 64, 64), activation='relu', solver='adam',max_iter=500)


# Create the ensemble learning model with VotingRegressor
ensemble_model = VotingRegressor(
    estimators=[('tree', tree_regressor), ('neural_network', neural_network_regressor)],
    weights=[1, 1]  # Adjust the weights to give more importance to one model over the other
)

# Train the ensemble learning model
ensemble_model.fit(X_train, Y_train)

# Predict with the ensemble learning model
ensemble_predictions = ensemble_model.predict(X_test)

# Calculate the MSE
mse = mean_squared_error(Y_test, ensemble_predictions)
print("Mean Squared Error:", mse)

# Calculate R-squared
r_squared = r2_score(Y_test, ensemble_predictions)
print("R-squared:", r_squared)
