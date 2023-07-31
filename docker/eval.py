import os
import subprocess
import pandas as pd
import shutil
import re

### PATHS ###

#abs_path='/Volumes/T7/deep_encode_dataset/DATASET_DEEP_ENCODE_2/pipeline_dataset'

#location of predictions
#predictions_loc=f'/Volumes/T7/deep_encode_dataset/DATASET_DEEP_ENCODE_2/_csv_pipeline/predictions_pipeline.csv'



#location of evaluation results from both label and prediction
#eval_result_loc_full=f'/Volumes/T7/deep_encode_dataset/DATASET_DEEP_ENCODE_2/_csv_pipeline/eval_full.csv'

#location of evaluation results given by difference between label and prediction
#eval_result_loc_diff=f'/Volumes/T7/deep_encode_dataset/DATASET_DEEP_ENCODE_2/_csv_pipeline/eval_diff.csv'

#location of evaluation results current full
#eval_result_loc_current=f'/Volumes/T7/deep_encode_dataset/DATASET_DEEP_ENCODE_2/_csv_pipeline/eval_current_full'


# HELPER METHODS

def extract_vmaf(result_string):
    # Extract the number using regex
    match = re.search(r"VMAF score: (\d+\.\d+)", result_string)

    # Check if a match is found and get the number
    if match:
        vmaf_score = float(match.group(1))
        return vmaf_score 
    else:
        print("VMAF score not found in the string")
        return -101.0



def run_evaluation(directory, learning_model):

    # locations
    predictions_loc = os.path.join(directory, f'predictions_for_eval_{learning_model}.csv')
    eval_result_loc_full = os.path.join(directory, f'eval_full_{learning_model}.csv')
    eval_result_loc_diff = os.path.join(directory, f'eval_diff_{learning_model}.csv')
    eval_result_loc_current = os.path.join(directory, f'eval_current_full_{learning_model}')

    # read predictions
    predictions=pd.read_csv(predictions_loc)
    print(f'Predictions for {learning_model}:\n{predictions}')

    #init output
    eval_full = pd.DataFrame()
    eval_diff = pd.DataFrame()
    vmaf_label = []
    vmaf_pred = []
    vmaf_diff = [] #pred - label
    scenes = []
    
    for video in os.listdir(directory):

        #check if dir
        if not os.path.isdir(os.path.join(directory, video)):
            continue

        # path to qp0 scenes
        scenes_path = os.path.join(directory, video, 'scenes')

        # path to eval encodes
        eval_encodes_path = os.path.join(directory, video, 'eval_encodes')

        #create dir for encodes
        if not os.path.isdir(eval_encodes_path):
            os.makedirs(eval_encodes_path)


        #iterate over scenes
        for filename in os.listdir(scenes_path):
            print(filename)

            # scene path
            scene_path = os.path.join(scenes_path, filename)

            # CHECK IF SCENE IS PART OF TEST DATA SET
            scene_name=os.path.splitext(filename)[0]
            if not scene_name in predictions['Name'].values:          #TODO check if filename or without .mp4
                print(f'SKIPPED {scene_name}')
                continue

            #get label from dataframe
            label_bitrate=predictions.loc[predictions.Name == f'{scene_name}','Label'].item()

            # if label bitrate valid
            if label_bitrate > 0:
            
                #encode with label bitrate
                label_encoded_filename = os.path.join(eval_encodes_path, f'label_{label_bitrate}M_{filename}')
                if not os.path.isfile(label_encoded_filename):
                    encode_command = f'ffmpeg  -v error -i {scene_path} -c:v libx264 -b:v {label_bitrate}M -preset ultrafast -pass 1 -f null /dev/null &&    \
                                        ffmpeg -v error -i {scene_path} -c:v libx264 -b:v {label_bitrate}M -preset ultrafast -pass 2 {label_encoded_filename}'
                    encode_result = subprocess.run(encode_command, capture_output=True, text=True, shell=True)
                    print(encode_result)


                #get prediction from dataframe
                predicted_bitrate=predictions.loc[predictions.Name == f'{filename}','Prediction'].item()

                #encode with predicted bitrate
                prediction_encoded_filename  = os.path.join(eval_encodes_path, f'predicted_{label_bitrate}M_{filename}')
                if not os.path.isfile(prediction_encoded_filename):
                    encode_command = f'ffmpeg  -v error -i {scene_path} -c:v libx264 -b:v {predicted_bitrate}M -preset ultrafast -pass 1 -f null /dev/null &&    \
                                        ffmpeg -v error -i {scene_path} -c:v libx264 -b:v {predicted_bitrate}M -preset ultrafast -pass 2 {prediction_encoded_filename}'
                    encode_result = subprocess.run(encode_command, capture_output=True, text=True, shell=True)
                    print(encode_result)

                # calc vmaf of label
                vmaf_command = f'ffmpeg -i {label_encoded_filename} -i {scene_path} -filter_complex libvmaf -f null -'
                vmaf_result = subprocess.run(vmaf_command, capture_output=True, text=True, shell=True)
                vmaf_score_label=extract_vmaf(str(vmaf_result))
                print('LABEL VMAF SCORE: ', vmaf_score_label)

                # calc vmaf of prediction
                vmaf_command = f'ffmpeg -i {prediction_encoded_filename} -i {scene_path} -filter_complex libvmaf -f null -'
                vmaf_result = subprocess.run(vmaf_command, capture_output=True, text=True, shell=True)
                vmaf_score_pred=extract_vmaf(str(vmaf_result))
                print('PREDICTION VMAF SCORE: ', vmaf_score_pred)

            # if label invalid, assign invalid vmaf
            else:
                vmaf_score_label = -101.0
                vmaf_score_pred = -101.0

            scenes.append(filename)
            vmaf_label.append(vmaf_score_label)
            vmaf_pred.append(vmaf_score_pred)

            vmaf_score_diff = 0
            if vmaf_score_pred == -101.0 or vmaf_score_label == -101.0:
                vmaf_score_diff = -101.0
            elif vmaf_score_pred != vmaf_score_label:
                vmaf_score_diff = vmaf_score_pred - vmaf_score_label
            vmaf_diff.append(vmaf_score_diff)

        current_full = pd.DataFrame()
        current_full = current_full.assign(Name=scenes, VMAF_LABEL=vmaf_label, VMAF_PRED=vmaf_pred)
        current_full.to_csv(f'{eval_result_loc_current}_{video}.csv')

        # delete eval_encodes
        shutil.rmtree(eval_encodes_path)
    
    eval_full = eval_full.assign(Name=scenes, VMAF_LABEL=vmaf_label, VMAF_PRED=vmaf_pred)
    eval_full.to_csv(eval_result_loc_full)

    eval_diff = eval_diff.assign(Name=scenes, VMAF_DIFF=vmaf_diff)
    eval_diff.to_csv(eval_result_loc_diff)

    print('------------')
    print('NAMES:')
    print(scenes)
    print('------------')
    print('LABEL:')
    print(vmaf_label)
    print('------------')
    print('PREDICTIONS:')
    print(vmaf_pred )

        


