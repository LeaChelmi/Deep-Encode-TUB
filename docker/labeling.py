import os
import subprocess
import pandas as pd
import numpy as np
import re
import math


### PARAMETERS ###

# define minimum acceptable vmaf
minimum_acceptable_vmaf = 92.0

# bitrates from 1 to 64 MB/s possible
bitrate_candidates = np.array(range(1, 65))

##################


# extract VMAF value from shell output
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

# delete test encode after use
def delete_encode(path):
    if os.path.isfile(path):
        # Delete the file
        os.remove(path)
        print(f"The file at {path} has been deleted.")
    else:
        print(f"No file found at {path}.")


def labeling(dataset_path):

    # data structures for saving labels
    labels_df = pd.DataFrame()
    labels = []
    scene_names = []

    #iterate over videos
    videos = os.listdir(dataset_path)
    for video in videos:

        video_path = os.path.join(dataset_path, video)
        if not os.path.isdir(video_path):
            continue

        scenes_path = os.path.join(video_path, 'scenes')
        scenes = os.listdir(scenes_path)

        #create encodes dir
        encodes_path = os.path.join(scenes_path, 'encodes')
        if not os.path.isdir(encodes_path):
            os.makedirs(encodes_path)

        for scene in scenes:

            # path to scene
            scene_path = os.path.join(scenes_path, scene)

            # set initial fallback label
            scene_label=-1

            #init binary search
            low = 0
            high = len(bitrate_candidates) - 1
            mid = 0
    
            #iterations for binary search
            iterations=math.floor(math.log2(len(bitrate_candidates)))

            last_upper_candidate_vmaf = 100.0
            last_upper_candidate=bitrate_candidates[len(bitrate_candidates)-1]

            #BINARY SEARCH OVER BITRATES
            while range(iterations):

                # get current bitrate
                mid = (high + low) // 2
                current_bitrate=bitrate_candidates[mid]
                print(f'CURRENT BITRATE: {current_bitrate}')

                #do encode
                encode_path = os.path.join(encodes_path, f'{current_bitrate}M_{scene}.mp4')
                encode_command = f'ffmpeg  -v error -i {scene_path} -c:v libx264 -b:v {current_bitrate}M -preset ultrafast -pass 1 -f null /dev/null &&    \
                                    ffmpeg -v error -i {scene_path} -c:v libx264 -b:v {current_bitrate}M -preset ultrafast -pass 2 {encode_path}'
                encode_result = subprocess.run(encode_command, capture_output=True, text=True, shell=True)
                #print('ENCODE RESULT: ', encode_result)

                # calc vmaf
                vmaf_command = f'ffmpeg -i {encode_path} -i {scene_path} -filter_complex libvmaf -f null -'
                vmaf_result = subprocess.run(vmaf_command, capture_output=True, text=True, shell=True)
                #print('VMAF RESULT: ', vmaf_result)

                # delete encode
                delete_encode(encode_path)

                vmaf_score=extract_vmaf(str(vmaf_result))
                print('VMAF SCORE: ', vmaf_score)
                #print(f'current last upper candidate vmaf: {last_upper_candidate_vmaf}')

                # final iteration
                if low == high:
                    
                    if vmaf_score > last_upper_candidate_vmaf or vmaf_score < minimum_acceptable_vmaf:
                        if last_upper_candidate_vmaf == 100:
                            print(f'ERROR: CANDIDATE WINDOW NOT FITTING!! DID NOT FIND ENCODE THAT IS ABOVE MINIMUM ACCEPTABLE VAMF. CLOSEST ENCODE FOUND AT BITRATE {current_bitrate} AND VMAF {vmaf_score}')
                            scene_label=current_bitrate
                        else:
                            print(f'CONVERGED: LAST UPPERCANDIDATE IS OPTIMAL. current vmaf_score: {vmaf_score} vs last_upper_candidate_vmaf: {last_upper_candidate_vmaf}')
                            print(f'FINAL BITRATE LABEL FOR SCENE {scene}: {last_upper_candidate}MBit/s')
                            scene_label=last_upper_candidate
                    
                    elif vmaf_score < last_upper_candidate_vmaf and vmaf_score > minimum_acceptable_vmaf:
                        print(f'CONVERGED: CURRENT VMAF IS OPTIMAL. current vmaf_score: {vmaf_score} vs last_upper_candidate_vmaf: {last_upper_candidate_vmaf}')
                        print(f'FINAL BITRATE LABEL FOR SCENE {scene}: {current_bitrate}MBit/s')
                        scene_label=current_bitrate

                    else:
                        print('ERROR DID NOT FIND OPTIMAL BITRATE')

                    break

                elif vmaf_score < minimum_acceptable_vmaf:
                    low = mid + 1
            
                elif vmaf_score > minimum_acceptable_vmaf:
                    high = mid - 1
                    last_upper_candidate=current_bitrate
                    last_upper_candidate_vmaf=vmaf_score
                
                else:
                    print('ERROR DID NOT FIND OPTIMAL BITRATE')

            labels.append(scene_label)
            scene_names.append(scene)

        # remove encodes dir
        os.rmdir(encodes_path)

    labels_df = labels_df.assign(Name=scene_names, Label=labels)

    labels_path = os.path.join(dataset_path, 'labels.csv')
    labels_df.to_csv(labels_path)



