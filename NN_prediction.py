
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#gets a link to the videos for which the prediction should be made
#Step 0: detect scene, folder structure
#Step 1: run feature extraction
#Step 2: take the csv files generated earlier to train the model
#Step 3: get csv that was saved after feature extraction and ask the model to predict

#STEP 2 and 3 :
    
def runPredictionBasedOnOurDataSetNN(path, path_to_features_and_labels):
    
    #path is the path to the features extracted from the videos for which we need prediction
    
    """
    Parameters:
        path (str): The path to the directory containing the features extracted from videos for which predictions are needed.
        path_to_features_and_labels (str): The path to the directory containing the 'features_and_labels.csv' file, representing the dataset with features and labels.

    Returns:
        pd.DataFrame: A DataFrame containing the predicted bitrates.
    """

    # Load data from the CSV file
    data = pd.read_csv(f"{path_to_features_and_labels}/features_and_labels.csv")
    # df is a table with features and labels 
    
    # Separate features (X) and labels (Y)
    X = data[['Bitrate','width','height','spatial_info','temporal_info','fps','entropy']]
    Y = data['label']                  
    X = X.astype(float)
    Y = Y.astype(float)
                       
    
    #normalization
    
    # Create a MinMaxScaler object
    scaler = MinMaxScaler()
    
    # Normalize the features (X)
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X)
      
    
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
    model.fit(X, Y, epochs = ep, batch_size=batchSize, verbose=1)
    
    #Predict
    df_for_prediction = pd.read_csv(f"{path}/features.csv")
    
    #part of the table with featuresonly, wthout columnName
    df_for_prediction_bis = df_for_prediction.iloc[:, 1:]


    
    
    # # Make predictions on the test data
    predictions = model.predict(df_for_prediction_bis)
    predictions = predictions.round().astype(int)  # Arrondir les valeurs prédites à l'entier le plus proche
    
    df_for_prediction['Predicted bitrate'] = predictions
                          
    #save file
    # Specify the file path and name for the CSV file
    csv_file_path = f"{path}/predictions_NN.csv"

    # Save the DataFrame to a CSV file
    df_for_prediction.to_csv(csv_file_path, index=False)
                 
    return df_for_prediction
    