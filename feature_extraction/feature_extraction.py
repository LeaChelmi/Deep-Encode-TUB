
import cv2
import os
import subprocess
import numpy as np
import pandas as pd
import json

   
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

def CreateFeatureTable(directory):
    
    """
    Creates the table that contains features of each video in the folder
    
    Parameters:
    directory (string): Locaton of the video scenes.
    
    Returns:
    df: The pandas table with scene name and features.
    Also saves the dataframe as csv file.
    """
    
    # Create the pandas DataFrame
    df = pd.DataFrame()
    
    filenames = []
    bitrates = [] 
    width = []
    height = []
    fps = []
    
    spatial_info = []
    temporal_info = []
    entropy = []


    # iterate over files in the directory
    
    #TODO: anpassen for a folder structure
    videos = os.listdir(directory)
    #print("print videos", videos)
    for video in videos:
    
        video_path = os.path.join(directory, video)
        if not os.path.isdir(video_path):
            continue

        scenes_path = os.path.join(video_path, 'scenes')
        scenes = os.listdir(scenes_path)
        for filename in scenes:
            f = os.path.join(scenes_path, filename)

            # checking if it is a file
            if os.path.isfile(f):

                if '.mp4' in f:
                    name = os.path.basename(f)
                    #name = os.path.splitext(name)[0]

                    filenames.append(name)
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
    #save file
    # Specify the file path and name for the CSV file
    csv_file_path = f"{directory}/features.csv"

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)
    return df
 





