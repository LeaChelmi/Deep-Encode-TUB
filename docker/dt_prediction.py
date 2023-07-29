
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split



#gets a link to the videos for which the prediction should be made
#Step 0: detect scene, folder structure
#Step 1: run feature extraction
#Step 2: take the csv files generated earlier to train the model
#Step 3: get csv that was saved after feature extraction and ask the model to predict

#THIS IS STEP 2 and 3

def runPredictionBasedOnOurDataSet(path, path_to_features_and_labels):
    #path is the path to the features extracted from the videos for which we need prediction
    
    """
    Parameters:
        path (str): The path to the directory containing the features extracted from videos for which predictions are needed.
        path_to_features_and_labels (str): The path to the directory containing the 'features_and_labels.csv' file, representing the dataset with features and labels.

    Returns:
        pd.DataFrame: A DataFrame containing the predicted bitrates.
    """


    #path_to_features_and_labels = os.path.join(os.getcwd(), 'default_dataset')
    
    df = pd.read_csv(f"{path_to_features_and_labels}/features_and_labels.csv")
    # df is a table with features and labels 
    #where label is in the last column called Label
    #Name is the first column called Name
    #OTHER COLUMNS ARE FEATURES

    #names of the columns with features, 
    #can also just take all columns apart from first with name and last with label
    
      
            
    df_copy= df.copy()
    df_copy.drop(columns=["Name", "Label"], inplace=True)
    list_of_features = df_copy.columns.tolist()
    X = df[list_of_features] #Features
    y = df.Label #Labels

 
    # create a regressor object
    regressor = DecisionTreeRegressor(random_state = 0) 

    # fit the regressor with X and Y data
    regressor.fit(X, y)
    
    #Predict
    df_for_prediction = pd.read_csv(f"{path}/features.csv")
    
    #part of the table with featuresonly, wthout columnName
    df_for_prediction_a = df_for_prediction.iloc[:, 1:]

    y_predicted = regressor.predict(df_for_prediction_a)

    df_for_prediction['Predicted bitrate'] = y_predicted
                      
    #save file
    # Specify the file path and name for the CSV file
    csv_file_path = f"{path}/dt_predictions.csv"

    # Save the DataFrame to a CSV file
    df_for_prediction.to_csv(csv_file_path, index=False)
                 
    return df_for_prediction


