#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import subprocess
import numpy as np
import pandas as pd
import time

#see what we need from these 
import pandas as pd
import numpy as np
import re
import urllib.parse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn import metrics
from sklearn.metrics import accuracy_score 

import matplotlib.pyplot as plt
from sklearn import tree
from six import StringIO  
from IPython.display import Image  
import pydotplus
from matplotlib.colors import LinearSegmentedColormap
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# import the regressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


# In[2]:


#Run to initiate functions 


def CreateFeatureTable(directory):
    
    """
    Creates the table that contains features of each video in the folder
    
    Parameters:
    directory (string): Locaton of the video scenes.
    
    Returns:
    df: The pandas table with scene name and features.
    """
    
    # Create the pandas DataFrame
    df = pd.DataFrame()
    
    filenames = []
    bitrates = [] 
    width = []
    height = []
    fps = []
    
    average_intensity = []

    average_brightness = []
    average_contrast = [] 
    average_sharpness = []
    motion_score = []
    spatial_info = []
    temporal_info = []
    entropy = []


    # iterate over files in the directory
    
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            filenames.append(f.replace('videos/', ''))
            
            b = GetBitrate(f)
            bitrates.append(b) 
            
            properties = get_video_properties(f)
            width.append(properties[0])
            height.append(properties[1])
            fps.append(properties[2])
            
            spatial_info.append(calculate_spatial_information(f))
            temporal_info.append(calculate_temporal_information(f))
            entropy.append(calculate_video_entropy(f))
            

    df = df.assign(Name=filenames,
                   Bitrate=bitrates,
                   width=width,
                   height=height,
                   spatial_info=spatial_info,
                   temporal_info=temporal_info,                   
                   fps=fps,
                   entropy=entropy
                  )
    return df
    
#Find video properties 

def get_video_properties(video_path):
    
    """
    The function retrieves properties of a video file given its path.

    Parameters:

    video_path (str): The file path to the video.
    
    Returns:

    width (int): The width of the video in pixels.
    height (int): The height of the video in pixels.
    fps (float): The frames per second of the video.
    
    """
    
    
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get the video width and height
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get the frames per second (FPS)
    fps = video.get(cv2.CAP_PROP_FPS)

    # Release the video file
    video.release()

    return width, height, fps


#Calculate the bitrate 

def GetBitrate(video_path):

    """
    The function calculates the estimated bitrate of a video file given its path.

    Parameters:
    video_path (str): The file path to the video.
    
    Returns:
    bitrate_mbps (float): The estimated bitrate of the video in megabits per second (Mbps).
   
    """
    
    # Get video file size in bytes
    file_size = os.path.getsize(video_path)

    # Get video duration in seconds using FFprobe
    ffprobe_cmd = f'ffprobe -i "{video_path}" -show_entries format=duration -v quiet -of csv="p=0"'
    result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, shell=True)
    duration = float(result.stdout.strip())

    # Calculate the estimated bitrate in kilobits per second (kbps)
    bitrate = (file_size * 8) / (duration * 1000)
    
    # Convert bitrate to megabits per second (Mbps)
    bitrate_mbps = bitrate / 1000

    return bitrate_mbps

def calculate_spatial_information(video_path):
    """    
    The function calculates the maximum spatial information of a video file given its path. 
    Spatial information is measured based 
    on the standard deviation of the Sobel filter applied to the luminance component of each frame.

    Parameters:
    video_path (str): The file path to the video.
    
    Returns:
    max_std (float): The maximum standard deviation of the Sobel filter applied to the frames.
   
    """    
    video = cv2.VideoCapture(video_path)
    sobel_stds = []

    while True:
        ret, frame = video.read()

        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply the Sobel filter to the luminance component
        sobel_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)

        # Calculate the standard deviation of the Sobel filter
        sobel_std = np.std(sobel)

        sobel_stds.append(sobel_std)

    video.release()

    # Calculate the maximum standard deviation
    max_std = max(sobel_stds)

    return max_std

def calculate_temporal_information(video_path):
    """     
    The function calculates the maximum temporal information of a video file given its path. 
    Temporal information is measured based on the standard deviation of the motion 
    difference between consecutive frames.

    Parameters:
    video_path (str): The file path to the video.
    
    Returns:
    max_std (float): The maximum standard deviation of the motion difference between frames.
    
    """ 
    
    video = cv2.VideoCapture(video_path)
    motion_diffs = []

    ret, prev_frame = video.read()

    while True:
        ret, curr_frame = video.read()

        if not ret:
            break

        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Calculate the motion difference between frames
        motion_diff = curr_gray - prev_gray

        # Calculate the standard deviation of the motion difference
        motion_std = np.std(motion_diff)

        motion_diffs.append(motion_std)

        prev_frame = curr_frame

    video.release()
    # Calculate the maximum standard deviation
    max_std = max(motion_diffs)

    return max_std


def calculate_entropy(image):
    """     
    The function calculate_entropy calculates the entropy of an input image.

    Parameters:
    image (numpy.ndarray): The input image as a NumPy array.
    
    Returns:
    entropy (float): The calculated entropy value of the image.
    
    """ 
    
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist /= hist.sum()  # Normalize histogram

    probabilities = hist.flatten()
    probabilities = probabilities[probabilities != 0]  # Remove zero probabilities

    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy

def calculate_video_entropy(video_path):
    """ 
    The function calculates the average entropy of a video file
    by calculating the entropy for each frame and then computing the mean entropy value.

    Parameters:
    video_path (str): The file path to the video.
    
    Returns:
    average_entropy (float): The average entropy of the video frames.
    
    """ 
    cap = cv2.VideoCapture(video_path)
    entropy_values = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        entropy = calculate_entropy(gray_frame)
        entropy_values.append(entropy)

    cap.release()

    average_entropy = np.mean(entropy_values)
    return average_entropy


# In[ ]:


# assign directory
directory = '/Users/anastasiya/Documents/MASTER/SOSE23/Deep_Encode/scenes'

# Start the timer
start_time = time.time()


df = CreateFeatureTable(directory)

# End the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Table created in: {elapsed_time} seconds")

#show table
df


# In[16]:



#merge tables with features and table with labels
#names should be same
   
path_to_features = 'features.csv'
path_to_labels = 'labels.csv'

#upload 
df_features = pd.read_csv(path_to_features)
df_labels = pd.read_csv(path_to_labels)

#merge tables 
df = pd.merge(df_features, df_labels, on=['Name'])

#raname tables if needed
df.drop('Unnamed: 0_x', axis=1, inplace=True)
df.drop('Unnamed: 0_y', axis=1, inplace=True)
df.rename(columns={'Bitrate_x': 'Bitrate'}, inplace=True)
df.rename(columns={'Bitrate_y': 'label'}, inplace=True)

#delete ows where label -1
df = df[df['label'] != -1]

#save file
# Specify the file path and name for the CSV file
csv_file_path = "features_with_labels.csv"

# Save the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)


# In[ ]:





# In[43]:


#DECISION TREE


df = pd.read_csv("features_with_labels.csv")

# df is a table with features and labels 
#where label is in the last column called Label

#names of the columns with features, 
#can also just take all columns apart from first with name and last with label

list_of_features = ['Bitrate', 'width', 'height', 'spatial_info', 'temporal_info', 'fps', 'entropy']

X = df[list_of_features] #Features
y = df.label #Labels

#split the data into train and test 

#Split the data into training set and test set
# 67% training and 33% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 

# create a regressor object
regressor = DecisionTreeRegressor( random_state = 0) 
  
# fit the regressor with X and Y data
regressor.fit(X_train, y_train)

# Predict
y_predicted = regressor.predict(X_test)

# Calculate the MSE
mse = mean_squared_error(y_test, y_predicted)
print("Mean Squared Error:", mse)


# Get the feature importances
list_of_features = ['Bitrate','width','height','spatial_info', 'temporal_info', 'fps', 'entropy']
feature_importances = regressor.feature_importances_

# Print the feature importances
print("Features importance \n")
for item1, item2 in zip(list_of_features, feature_importances):
    print(f"{item1}: {item2}")
    
#to compare the result we have to set some treshhold 
#like +-5% of abweichung in bitrate for prediction 
compare = pd.DataFrame()
compare= compare.assign(test =y_test, predicted = y_predicted)


# In[44]:


#visualise DT
#optional

from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(regressor, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = list_of_features)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('tree.png')
Image(graph.create_png())


# In[59]:


#table with test set, so that we can identify for which videos to calculate the vmaf

test_table = X_test.copy()
test_table['Name'] = df['Name'].iloc[X_test.index]
test_table['Label'] = y_test
test_table['Prediction'] = y_predicted
#save file
# Specify the file path and name for the CSV file
csv_file_path = "predictions.csv"

# Save the DataFrame to a CSV file
test_table.to_csv(csv_file_path, index=False)


# In[60]:


test_table


# In[ ]:


test_table_filter = test_table[test_table['Label'] > test_table['Prediction']]
test_table_filter = test_table_filter.iloc[1:2]
test_table_filter


# In[ ]:





# In[ ]:





# In[68]:


def extract_vmaf(result_string):
    # Extract the number using regex
    match = re.search(r"VMAF score: (\d+\.\d+)", result_string)

    # Check if a match is found and get the number
    if match:
        vmaf_score = float(match.group(1))
        return vmaf_score 
    else:
        print("VMAF score not found in the string")
        return 101.0
    
    ### INPUT VIDEO META DATA ###
scene_encode_loc = '/Users/anastasiya/Documents/MASTER/SOSE23/Deep_Encode/vmaf/pred'
orig_scenes_loc = '/Users/anastasiya/Documents/MASTER/SOSE23/Deep_Encode/vmaf'

input_vid_pix_fmt = 'yuv420p'
for index, row in test_table_filter.iterrows():
    if 'height' in row and 'width' in row:
        input_vid_res = f"{row['width']}x{row['height']}"
    if 'fps' in row:
        input_vid_fps = f"{row['fps']}"
        
    current_bitrate = row['Prediction']

    #do encode
    qp0_filename = f"{orig_scenes_loc}/{row['Name']}.mp4"
    encoded_filename = f"{scene_encode_loc}/{row['Name']}.mp4"
    encode_command = f"ffmpeg  -v error -f rawvideo -vcodec rawvideo -s {input_vid_res} -r {input_vid_fps} -pix_fmt {input_vid_pix_fmt} -i {orig_scenes_loc}/{qp0_filename} -c:v libx264 -b:v {current_bitrate}M -preset ultrafast -pass 1 -f null /dev/null &&                                    ffmpeg -v error -f rawvideo -vcodec rawvideo -s {input_vid_res} -r {input_vid_fps} -pix_fmt {input_vid_pix_fmt} -i {orig_scenes_loc}/{qp0_filename} -c:v libx264 -b:v {current_bitrate}M -preset ultrafast -pass 2 {encoded_filename}"
    encode_result = subprocess.run(encode_command, capture_output=True, text=True, shell=True)

    vmaf_command = f"ffmpeg -i {encoded_filename} -i {qp0_filename} -filter_complex libvmaf -f null -"
    vmaf_result = subprocess.run(vmaf_command, capture_output=True, text=True, shell=True)
    print('VMAF RESULT: ', vmaf_result)

    vmaf_score=extract_vmaf(str(vmaf_result))
    print('VMAF SCORE: ', vmaf_score)
    #print(f'current last upper candidate vmaf: {last_upper_candidate_vmaf}')


# In[79]:





# TRYNG OTHER OPTION



import subprocess

def compress_video(input_path, output_path, target_bitrate):
    # FFmpeg command to compress video to target bitrate
    ffmpeg_cmd = f'ffmpeg -i "{input_path}" -b:v {target_bitrate} "{output_path}"'

    # Execute the FFmpeg command
    subprocess.run(ffmpeg_cmd, shell=True)

# Example usage
input_file = '/Users/anastasiya/Documents/MASTER/SOSE23/Deep_Encode/vmaf/big_buck_bunny_720p24-Scene-124.mp4'
output_file = '/Users/anastasiya/Documents/MASTER/SOSE23/Deep_Encode/vmaf/big_buck_bunny_720p24-Scene-124-comp.mp4'
target_bitrate = '5000k'  # Set the target bitrate (e.g., 1000 kilobits per second)

compress_video(input_file, output_file, target_bitrate)


# In[80]:


import subprocess

def calculate_vmaf(input_path, compressed_path):
    # FFmpeg command to calculate VMAF score
    ffmpeg_cmd = f'ffmpeg -i "{input_path}" -i "{compressed_path}" -lavfi "libvmaf=psnr=1:log_path=vmaf.log" -f null -'

    # Execute the FFmpeg command
    subprocess.run(ffmpeg_cmd, shell=True)

# Example usage
input_file = '/Users/anastasiya/Documents/MASTER/SOSE23/Deep_Encode/vmaf/big_buck_bunny_720p24-Scene-124.mp4'
compressed_file = '/Users/anastasiya/Documents/MASTER/SOSE23/Deep_Encode/vmaf/big_buck_bunny_720p24-Scene-124-comp.mp4'

calculated_vmaf = calculate_vmaf(input_file, compressed_file)
print(calculated_vmaf)


# In[ ]:




