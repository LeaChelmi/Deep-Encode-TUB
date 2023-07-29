
import cv2
import os
import subprocess
import pandas as pd
import json


def get_video_properties(video_path):
    
    """
    The function retrieves properties of a video file given its path.

    Parameters:

    video_path (str): The file path to the video.
    
    Returns:

    width (int): The width of the video in pixels.
    height (int): The height of the video in pixels.
    fps (float): The frames per second of the video.
    pix_fmt(str): The pixel format of the video.
    
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
    
    ffprobe_cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=pix_fmt -of json "{video_path}"'

    

    # Execute the ffprobe command
    result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, shell=True)

    print(f'Result:\n\n{result}\n\n')

    # Parse the JSON output
    metadata = json.loads(result.stdout)
    pix_fmt = metadata['streams'][0]['pix_fmt']

    return width, height, fps, pix_fmt


def createMetadataCsv(directory):
    
    # Create the pandas DataFrame
    df = pd.DataFrame()
    
    filenames = []
    resolution = []
    width = []
    height = []
    fps = []
    pixel_format = []

    # iterate over files in the directory
    
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if (filename == '.DS_Store'):
                continue
            name = os.path.basename(f)
            name = os.path.splitext(name)[0]

            filenames.append(name)
            print(f)
            properties = get_video_properties(f)
            res = f"{properties[0]}x{properties[1]}"
            resolution.append(res)
            fps.append(properties[2])
            pixel_format.append(properties[3])

    df = df.assign(VideoName=filenames,
                   Resolution =  resolution,          
                   FPS=fps,
                   PIX_FMT=pixel_format)
    
    #save file
    # Specify the file path and name for the CSV file
    csv_file_path = os.path.join(directory, 'metadata.csv')

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)
    
    return df

