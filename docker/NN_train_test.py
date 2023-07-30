import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import os

#Neural Network
#folder with features and labels
#initiates def mergeFeaturesAndLabels

def runNeuralNetwork(folder):
    
    """
    Parameters:
        folder (str): The path to the folder containing 'features_and_labels.csv'.

    """
    
    # Load data from the CSV file
    # data is a table with features and labels
    data = pd.read_csv(os.path.join(folder, 'final_dataset_merged.csv'))
    
    # Separate features (X) and labels (Y)
    X = data[['Bitrate','width','height','spatial_info','temporal_info','fps','entropy']]
    Y = data['Label']                  
    X = X.astype(float)
    Y = Y.astype(float)
                       
    print(f'X:\n{X}')
    print(f'Y:\n{Y}')
    
    #normalization
    
    # Create a MinMaxScaler object
    scaler = MinMaxScaler()
    
    # Normalize the features (X)
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X)
    
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
    model.fit(X_train, Y_train, epochs = ep, batch_size=batchSize, verbose=0)
    
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
    df_predictions = pd.DataFrame({'Actual Values': Y_test.values, 'Predicted Values': predictions.flatten()})
    
    #print(df_predictions) 
    
    #table with test set, so that we can identify for which videos to calculate the vmaf
    
    # test_table = X_test.copy()
    test_table = pd.DataFrame()
    test_table['Name'] = data['Name'].iloc[X_test.index]
    test_table['Label'] = Y_test
    test_table['Prediction'] = predictions
    #save file
    # Specify the file path and name for the CSV file
    csv_file_path = os.path.join(folder, 'predictions_for_eval_NN.csv')
    
    # Save the DataFrame to a CSV file
    test_table.to_csv(csv_file_path, index=False)
    
    
    # Plot a graph of actual values vs. predicted values 
    # fig, ax = plt.subplots(figsize=(15, 15))
    # ax.scatter(Y_test, predictions,s=20)
    # ax.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--', label='Diagonale line')
    # ax.set_xlabel('Actual Values')
    # ax.set_ylabel('Predicted Values')
    # ax.set_title('Comparison of Actual and Predicted Values')
    # ax.legend()
    # plt.show()

    # return model