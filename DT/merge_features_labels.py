
import pandas as pd
import os

#merge tables with features and table with labels
#names should be same

#we have a path to the main folder 

def mergeFeaturesAndLabels(folder):
    
    """
    Merges the features and labels from CSV files in the specified folder and saves the result in a new CSV file.

    Parameters:
        folder (str): The path to the folder containing 'features.csv' and 'labels.csv' files.

    Returns:
        Merged DataFrame.
    """

    path_to_features = f'{folder}/features.csv'
    path_to_labels = f'{folder}/labels.csv'

    #upload 
    df_features = pd.read_csv(path_to_features)
    df_labels = pd.read_csv(path_to_labels)
    
    #merge tables 
    df = pd.merge(df_features, df_labels, on=['Name'])

    #raname tables if needed
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    #df.drop('Unnamed: 0_y', axis=1, inplace=True)

    #delete rows where label -1
    df = df[df['Label'] != -1]

    #save file
    # Specify the file path and name for the CSV file
    csv_file_path = f'{folder}/features_and_labels.csv'

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)
    return df






