

import os
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


#DECISION TREE
#folder with features and labels
#initiates def mergeFeaturesAndLabels

def runDecisionTreeRegressor(folder):
    """
    Runs a Decision Tree Regressor on the data in the specified folder, predicts the labels, and calculates the Mean Squared Error (MSE).

    Parameters:
        folder (str): The path to the folder containing 'features_and_labels.csv'.

    Returns:
        regressor: The trained Decision Tree Regressor object.
    """
    
    df = pd.read_csv(f"{folder}/features_and_labels.csv")
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

    #split the data into train and test 

    #Split the data into training set and test set
    # 67% training and 33% test
    
    #random state can be removed if we dont want to see the same split each time
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 
    # create a regressor object
    regressor = DecisionTreeRegressor(random_state = 0) 

    # fit the regressor with X and Y data
    regressor.fit(X_train, y_train)

    # Predict
    y_predicted = regressor.predict(X_test)

    # Calculate the MSE
    mse = mean_squared_error(y_test, y_predicted)
    print("Mean Squared Error:", mse)


    # Get the feature importances
    feature_importances = regressor.feature_importances_

    # Print the feature importances
    print("Features importance \n")
    for item1, item2 in zip(list_of_features, feature_importances):
        print(f"{item1}: {item2}")
    

    test_table = X_test.copy() 
    test_table['Name'] = df['Name'].iloc[X_test.index]
    test_table['Label'] = y_test
    test_table['Prediction'] = y_predicted
    #save file
    # Specify the file path and name for the CSV file
    csv_file_path = "predictions_for_eval.csv"

    # Save the DataFrame to a CSV file
    test_table.to_csv(csv_file_path, index=False)

    return regressor




