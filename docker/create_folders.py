import os
import shutil

def organize_files(folder_path):
    # Get a list of all files in the specified folder
    files = os.listdir(folder_path)

    # Iterate over each file
    for file_name in files:

        # Check for .DS_Store
        if (file_name == '.DS_Store'):
            continue
        
        # Construct the full path of the file
        file_path = os.path.join(folder_path, file_name)

        # Check if the file is a regular file (not a directory)
        if os.path.isfile(file_path):
            # Create a new folder with the same name as the file (without extension)
            folder_name = os.path.splitext(file_name)[0]
            new_folder_path = os.path.join(folder_path, folder_name)
            os.makedirs(new_folder_path, exist_ok=True)

            # Move the file to the new folder
            new_file_path = os.path.join(new_folder_path, file_name)
            shutil.move(file_path, new_file_path)

            print(f"Moved file '{file_name}' to folder '{folder_name}'.")


# Specify the folder path where the files are located
folder_path = os.path.join(os.getcwd(), 'default_dataset')

# Call the function to organize the files
organize_files(folder_path)



