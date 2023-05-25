#!/usr/bin/env python
# coding: utf-8

# In[34]:


#collect here all imports and installs
import cv2
import os
import subprocess
import numpy as np
import pandas as pd
import time


# In[ ]:





# In[43]:


# assign directory
directory = 'videos'

# Start the timer
start_time = time.time()
print(CreateFeatureTable(directory))
# End the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Table created in: {elapsed_time} seconds")


# In[ ]:


#without motion score

#with motion score


# In[38]:


#Run to initiate functions 


def CreateFeatureTable(directory):
    #Create Pandas table 
    
    # Start the timer
    start_time = time.time()
    # Create the pandas DataFrame with column name is provided explicitly
    df = pd.DataFrame()
    
    filenames = []
    bitrates = [] 
    average_intensity = []

    average_brightness = []
    average_contrast = [] 
    average_sharpness = []
    motion_score = []


    # iterate over files in
    # that directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            filenames.append(f.replace('videos/', ''))
            b = GetBitrate(f)
            bitrates.append(b)
            average_intensity.append(AverageSpatialValue(f))
            SP = AverageSpatialInformation(f)
            average_brightness.append(SP[0])
            average_contrast.append(SP[1])
            average_sharpness.append(SP[2])
            motion_score.append(calculateMotionScore(f))


      

    df = df.assign(Name=filenames,
                   Bitrate=bitrates,
                  average_intensity=average_intensity,
                   average_brightness = average_brightness,
                   average_contrast = average_contrast,
                   average_sharpness = average_sharpness,
                   motion_score = motion_score
                  )
    # End the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print(f"Table created in: {elapsed_time} seconds")
  
    return df
    

#Calculate the bitrate 

def GetBitrate(video_path):
    
    # Start the timer
    #start_time = time.time()
    
    # Get video file size in bytes
    file_size = os.path.getsize(video_path)

    # Get video duration in seconds using FFprobe
    ffprobe_cmd = f'ffprobe -i "{video_path}" -show_entries format=duration -v quiet -of csv="p=0"'
    result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, shell=True)
    duration = float(result.stdout.strip())

    # Calculate the estimated bitrate in kilobits per second (kbps)
    bitrate = (file_size * 8) / (duration * 1000)
    
    # End the timer
    #end_time = time.time()

    # Calculate the elapsed time
    #elapsed_time = end_time - start_time

    #print(f"Bitrate calculated in: {elapsed_time} seconds")
    
    return bitrate

    #print(f"Video Bitrate: {bitrate:.2f} kbps")
    
#WHAT IT GIVES US?

#Average Spatial Value


import cv2
import numpy as np

def AverageSpatialValue(video_path):
    # Start the timer
    #start_time = time.time()
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is successfully opened
    if not cap.isOpened():
        print("Error opening video file")
        exit()

    # Initialize variables
    frame_count = 0
    total_intensity = 0

    # Read and process frames
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # If the frame was not read successfully, then we have reached the end of the video
        if not ret:
            break

        # Calculate the average pixel intensity of the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        intensity = np.mean(gray_frame)

        # Accumulate the total intensity
        total_intensity += intensity

        # Display the frame with intensity value (optional)
        #cv2.putText(frame, f"Intensity: {intensity:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #cv2.imshow('Frame with Intensity', frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        #    break

        # Increment frame count
        frame_count += 1

    # Calculate the average spatial value
    average_intensity = total_intensity / frame_count

    # Release the video capture object and close any open windows
    cap.release()
    cv2.destroyAllWindows()
    
    # End the timer
    #end_time = time.time()

    # Calculate the elapsed time
    #elapsed_time = end_time - start_time

    #print(f"Average Spatial Value calculated in: {elapsed_time} seconds")
    
    return average_intensity
    #print(f"Average Spatial Value: {average_intensity}")
    
    
    
#Calculate average spatial information


import cv2
def AverageSpatialInformation(video_path):

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize variables to store spatial information
    frame_count = 0
    total_brightness = 0.0
    total_contrast = 0.0
    total_sharpness = 0.0

    # Iterate over the frames
    while True:
        # Read the next frame
        ret, frame = cap.read()

        # If no more frames, break the loop
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate brightness (average pixel intensity)
        brightness = gray_frame.mean()
        total_brightness += brightness
        #print('calculate brightness')
        # Calculate contrast (standard deviation of pixel intensities)
        contrast = gray_frame.std()
        total_contrast += contrast
        #print('calculate contrast')
        # Calculate sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
        total_sharpness += laplacian_var
        #print('calculate sharpness')

        # Increment frame count
        frame_count += 1

    # Release the video capture
    cap.release()

    # Calculate average spatial information
    average_brightness = total_brightness / frame_count
    average_contrast = total_contrast / frame_count
    average_sharpness = total_sharpness / frame_count
    
    return average_brightness, average_contrast, average_sharpness

    #print(frame_count)
    # Print the spatial information measures
    print(f"Average Brightness: {average_brightness}")
    print(f"Average Contrast: {average_contrast}")
    print(f"Average Sharpness: {average_sharpness}")


# In[42]:


#TAKES TOO LONG

#Analyse motion
#Returns video Motion Score


def calculateMotionScore(video_path):

    # Start the timer
    #start_time = time.time()

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, prev_frame = cap.read()

    # Convert the frame to grayscale
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Initialize motion vectors
    motion_vectors = []

    # Set parameters for Farneback optical flow
    params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    # Iterate over the frames
    while True:
        # Read the next frame
        ret, frame = cap.read()

        # If no more frames, break the loop
        if not ret:
            break

        # Convert the frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow using Farneback algorithm
        flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, frame_gray, None, **params)

        # Calculate the magnitude of motion vectors
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

        # Calculate the average magnitude
        average_magnitude = np.mean(magnitude)

        # Store the average magnitude as a motion value
        motion_vectors.append(average_magnitude)

        # Set the current frame as the previous frame for the next iteration
        prev_frame_gray = frame_gray.copy()

    # Release the video capture
    cap.release()

    # Calculate the overall motion score
    motion_score = np.mean(motion_vectors)



    # End the timer
    #end_time = time.time()

    # Calculate the elapsed time
    #elapsed_time = end_time - start_time

    #print(f"Elapsed Time too find motion score: {elapsed_time} seconds")

    #print(f"Motion Score: {motion_score}")
    return motion_score


# In[29]:





# In[ ]:




