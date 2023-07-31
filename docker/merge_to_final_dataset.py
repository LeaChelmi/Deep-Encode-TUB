import pandas as pd
import os


def merge_to_final_dataset(path):
    """
    Merge the features and labels of a new dataset with an existing final dataset CSV file.

    Parameters:
    path (str): The path to the directory containing the new dataset's 'features_and_labels.csv' file.
    path_to_final_dataset (str): The path to the directory containing the existing final dataset CSV file.

    Returns:
    None: The function saves the merged DataFrame as 'final_dataset.csv' in the specified directory.

    Note:
    The 'features_and_labels.csv' file in the 'path' directory should have the same column structure as the existing final dataset.

    """
    new_dataset_path = os.path.join(path, 'features_and_labels.csv') # features and labels of a new dataset
    final_dataset_input_path = os.path.join(os.getcwd(), 'default_dataset.csv') # features and labels of the exising dataset
    final_dataset_path = os.path.join(path, 'final_dataset.csv')

    # Read the CSV files into Pandas DataFrames
    df1 = pd.read_csv(new_dataset_path)
    print(f'NEW DATA:\n{df1}')
    df2 = pd.read_csv(final_dataset_input_path)
    print(f'LOADED FINAL DATASET:\n{df2}')

    # Merge the two DataFrames based on the common columns
    merged_df = pd.concat([df1, df2], ignore_index=True)
    print(f'MERGED FINAL DATASET:\n{merged_df}')

    # save a new file into the final_dataset.csv
    merged_df.to_csv(final_dataset_path, index=False)